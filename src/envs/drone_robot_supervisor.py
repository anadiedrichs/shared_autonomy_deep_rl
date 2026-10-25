# -*- coding: utf-8 -*-
"""
Base Gymnasium environment and Webots Supervisor controller for the Crazyflie quadrotor.

Adaptation to a continuous action space and 11 continuous observations based on
prior thesis work (Diedrichs, 2025, drone-deep-rl).
"""

import os
import sys
import time
from math import cos, pi, sin
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    gym = None

    class _DummySpaces:
        class Box:
            def __init__(self, low, high, shape, dtype=np.float32):
                self.low = np.full(shape, low, dtype=dtype) if np.isscalar(low) else np.array(low, dtype=dtype)
                self.high = np.full(shape, high, dtype=dtype) if np.isscalar(high) else np.array(high, dtype=dtype)
                self.shape = tuple(shape)
                self.dtype = dtype

            def contains(self, x):
                return bool(np.all(x >= self.low) and np.all(x <= self.high))

    spaces = _DummySpaces

class _DummySupervisor:
    """Dummy class that emulates the minimal Webots Supervisor API for testing outside the simulator."""

    def __init__(self):
        pass

    def getBasicTimeStep(self) -> float:
        return 32.0

    def getTime(self) -> float:
        return 0.0

    def step(self, timestep: int) -> int:
        return 0

    def getFromDef(self, name: str):
        return None

    def getDevice(self, name: str):
        return None

    def simulationReset(self):
        pass

    def simulationResetPhysics(self):
        pass


class _DummyKeyboard:
    """Dummy class that emulates the Webots keyboard device."""

    def __init__(self):
        pass

    def enable(self, timestep: int):
        pass

    def getKey(self) -> int:
        return -1


if os.environ.get("WEBOTS_MOCK", "0") == "1":
    Supervisor = _DummySupervisor
    Keyboard = _DummyKeyboard
else:
    # Auto-detection of WEBOTS_HOME and Webots controller library if not in PYTHONPATH
    _webots_candidates = [
        os.environ.get("WEBOTS_HOME", ""),
        "/usr/local/webots",
        "/snap/webots/current/usr/share/webots",
    ]
    for _candidate in _webots_candidates:
        if _candidate and os.path.exists(_candidate):
            if "WEBOTS_HOME" not in os.environ:
                os.environ["WEBOTS_HOME"] = _candidate
            _webots_py = os.path.join(_candidate, "lib", "controller", "python")
            if os.path.exists(_webots_py) and _webots_py not in sys.path:
                sys.path.append(_webots_py)
            break

    try:
        from controller import Keyboard, Supervisor
    except (ImportError, KeyError, OSError):  # pragma: no cover
        Supervisor = _DummySupervisor
        Keyboard = _DummyKeyboard


from src.utils.pid_controller import pid_velocity_fixed_height_controller
from src.utils.utilities import normalize_to_range


class DroneRobotSupervisor(Supervisor, gym.Env if gym is not None else object):
    """
    Base Gymnasium environment coupled with Webots for the Crazyflie quadrotor.

    Continuous action space (4 dimensions in [-1.0, 1.0]):
        - a[0]: forward longitudinal velocity command vx (normalized)
        - a[1]: lateral displacement velocity command vy (normalized)
        - a[2]: yaw angular velocity command yaw_rate (normalized)
        - a[3]: vertical velocity / altitude variation command (normalized)

    Continuous observation space (11 dimensions in [-1.0, 1.0]):
        - [0]: roll (rad normalized from [-pi, pi])
        - [1]: pitch (rad normalized from [-pi/2, pi/2])
        - [2]: yaw_rate (rad/s normalized from [-pi, pi])
        - [3]: v_x (m/s in drone frame normalized from [-MAX_OBS_VELOCITY, MAX_OBS_VELOCITY])
        - [4]: v_y (m/s in drone frame normalized from [-MAX_OBS_VELOCITY, MAX_OBS_VELOCITY])
        - [5]: v_z (m/s vertical velocity normalized from [-MAX_OBS_VELOCITY, MAX_OBS_VELOCITY])
        - [6]: altitude (meters normalized from [0.1, 5.0])
        - [7]: range_front (mm normalized from [2.0, 2000.0])
        - [8]: range_back (mm normalized from [2.0, 2000.0])
        - [9]: range_right (mm normalized from [2.0, 2000.0])
        - [10]: range_left (mm normalized from [2.0, 2000.0])
    """

    # Physical flight and control bounds
    MAX_FORWARD_VELOCITY = 0.09  # Maximum longitudinal velocity in m/s
    MAX_LATERAL_VELOCITY = 0.09  # Maximum lateral velocity in m/s
    MAX_YAW_RATE = 0.5  # Maximum yaw angular velocity in rad/s (dynamics)
    MAX_VERTICAL_VELOCITY = 0.05  # Maximum ascent/descent velocity in m/s
    MAX_OBS_VELOCITY = 1.0  # Physical velocity range for observation normalization (m/s)

    # Altitude parameters
    MIN_HEIGHT = 0.20  # Minimum allowed altitude in flight (meters)
    MAX_HEIGHT = 0.70  # Target altitude threshold for take-off (meters)
    MAX_FLIGHT_HEIGHT = 1.00  # Maximum allowed altitude in flight (meters)
    HEIGHT_INITIAL = 0.30  # Initial target altitude at the start of take-off (meters)
    HEIGHT_INCREASE = 0.05  # Increment in take-off ramp (meters)
    WAITING_TIME = 3.0  # Stabilization time in seconds after take-off

    def __init__(
        self,
        max_episode_steps: int = 1_000,
        enable_trajectory_logging: bool = False,
        verbose: int = 1,
    ):
        """
        Initializes the Crazyflie robot supervisor.

        Args:
            max_episode_steps: Maximum step limit per episode (default 1000).
            enable_trajectory_logging: Enables saving the CSV trajectory to disk.
            verbose: Console detail level (0: silent for massive RL, 1: key transitions and take-off, 2: extended debug).
        """
        super().__init__()
        self.max_episode_steps = max_episode_steps
        self.enable_trajectory_logging = enable_trajectory_logging
        self.verbose = int(verbose)
        self.render_mode = None

        # Formal definition of Gymnasium spaces
        if spaces is not None:
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(4,), dtype=np.float32
            )
            self.observation_space = spaces.Box(
                low=-1.0, high=1.0, shape=(11,), dtype=np.float32
            )
        if gym is not None:
            self.spec = gym.envs.registration.EnvSpec(
                id="CrazyflieContinuous-v0",
                max_episode_steps=max_episode_steps,
            )

        # State and odometry variables
        self.timestep = int(self.getBasicTimeStep()) if hasattr(self, "getBasicTimeStep") else 32
        self.episode_step = 0
        self.episode_score = 0.0
        self.first_time = True
        self.dt = float(self.timestep) / 1000.0
        self.past_time = 0.0

        # Global positions
        self.x_global = 0.0
        self.y_global = 0.0
        self.z_global = 0.0
        self.past_x_global = 0.0
        self.past_y_global = 0.0
        self.past_z_global = 0.0
        self.x_pos_initial = 0.0
        self.y_pos_initial = 0.0

        # Velocities and orientations
        self.v_x = 0.0
        self.v_y = 0.0
        self.v_z = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.yaw_rate = 0.0
        self.alt = 0.0
        self.lineal_velocity = np.zeros(3, dtype=np.float32)
        self.angular_velocity = np.zeros(3, dtype=np.float32)

        # ToF distance sensors (in millimeters)
        self.dist_front = 2000.0
        self.dist_back = 2000.0
        self.dist_right = 2000.0
        self.dist_left = 2000.0

        # Flight and control state
        self.height_desired = self.HEIGHT_INITIAL
        self.the_drone_took_off = False
        self.timestamp_take_off = 0.0
        self.is_success = False
        self.terminated = False
        self.truncated = False
        self.current_action = np.zeros(4, dtype=np.float32)
        self.prev_action = np.zeros(4, dtype=np.float32)

        # Trajectory logging
        self.drone_trajectory_path = None
        self.drone_trajectory_file = None

        # Controllers and devices
        self.PID_crazyflie = None
        self.motors = [None for _ in range(4)]
        self.imu = None
        self.gps = None
        self.gyro = None
        self.camera = None
        self.range_front = None
        self.range_back = None
        self.range_right = None
        self.range_left = None
        self.robot = None
        self._keyboard = None
        self._rng = np.random.default_rng()

        self._initialization()

    def _initialization(self):
        """
        Initializes devices, sensors, actuators, and the PID controller of the Crazyflie.
        Must be called always after __init__ and reset.
        """
        if hasattr(self, "getBasicTimeStep"):
            self.timestep = int(self.getBasicTimeStep())
        else:
            self.timestep = 32

        self.dt = float(self.timestep) / 1000.0
        self.PID_crazyflie = pid_velocity_fixed_height_controller()

        # Keyboard for optional manual teleoperation
        self._keyboard = Keyboard()
        if hasattr(self._keyboard, "enable"):
            self._keyboard.enable(self.timestep)

        # Sensors integrated into the Crazyflie node
        self.imu = self.getDevice("inertial_unit") if hasattr(self, "getDevice") else None
        if self.imu is not None and hasattr(self.imu, "enable"):
            self.imu.enable(self.timestep)

        self.gps = self.getDevice("gps") if hasattr(self, "getDevice") else None
        if self.gps is not None and hasattr(self.gps, "enable"):
            self.gps.enable(self.timestep)

        self.gyro = self.getDevice("gyro") if hasattr(self, "getDevice") else None
        if self.gyro is not None and hasattr(self.gyro, "enable"):
            self.gyro.enable(self.timestep)

        # self.camera = self.getDevice("camera") if hasattr(self, "getDevice") else None
        # if self.camera is not None and hasattr(self.camera, "enable"):
        #     self.camera.enable(self.timestep)

        self.range_front = self.getDevice("range_front") if hasattr(self, "getDevice") else None
        if self.range_front is not None and hasattr(self.range_front, "enable"):
            self.range_front.enable(self.timestep)

        self.range_left = self.getDevice("range_left") if hasattr(self, "getDevice") else None
        if self.range_left is not None and hasattr(self.range_left, "enable"):
            self.range_left.enable(self.timestep)

        self.range_back = self.getDevice("range_back") if hasattr(self, "getDevice") else None
        if self.range_back is not None and hasattr(self.range_back, "enable"):
            self.range_back.enable(self.timestep)

        self.range_right = self.getDevice("range_right") if hasattr(self, "getDevice") else None
        if self.range_right is not None and hasattr(self.range_right, "enable"):
            self.range_right.enable(self.timestep)

        # Reset state and odometry variables
        self.episode_step = 0
        self.episode_score = 0.0
        self.first_time = True
        self.past_time = 0.0

        self.x_global = 0.0
        self.y_global = 0.0
        self.z_global = 0.0
        self.past_x_global = 0.0
        self.past_y_global = 0.0
        self.past_z_global = 0.0
        self.x_pos_initial = 0.0
        self.y_pos_initial = 0.0

        self.v_x = 0.0
        self.v_y = 0.0
        self.v_z = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.yaw_rate = 0.0
        self.alt = 0.0
        self.lineal_velocity = np.zeros(3, dtype=np.float32)
        self.angular_velocity = np.zeros(3, dtype=np.float32)

        self.dist_front = 2000.0
        self.dist_back = 2000.0
        self.dist_right = 2000.0
        self.dist_left = 2000.0

        self.height_desired = self.HEIGHT_INITIAL
        self.the_drone_took_off = False
        self.timestamp_take_off = 0.0
        self.is_success = False
        self.terminated = False
        self.truncated = False
        self.current_action = np.zeros(4, dtype=np.float32)
        self.prev_action = np.zeros(4, dtype=np.float32)
        self.drone_trajectory_file = None

        # Reference to the Crazyflie robot
        self.robot = self.getFromDef("crazyflie") if hasattr(self, "getFromDef") else None

        # Motors
        self.motors = [None for _ in range(4)]
        self._setup_motors()

    def _setup_motors(self):
        """
        Configures the quadrotor motors in velocity control mode.
        """
        if not hasattr(self, "getDevice"):
            return

        motor_names = ["m1_motor", "m2_motor", "m3_motor", "m4_motor"]
        for i, name in enumerate(motor_names):
            motor = self.getDevice(name)
            self.motors[i] = motor
            if motor is not None and hasattr(motor, "setPosition"):
                motor.setPosition(float("inf"))

        self._setup_motors_velocity([1.0, 1.0, 1.0, 1.0])

    def _setup_motors_velocity(self, motor_power: list[float] | np.ndarray):
        """
        Applies the computed power to each of the 4 Crazyflie motors.
        Motors 0 (m1) and 2 (m3) require sign inversion due to their rotation direction.
        """
        if self.motors[0] is None:
            return

        p = [float(val) for val in motor_power]
        self.motors[0].setVelocity(-p[0])
        self.motors[1].setVelocity(p[1])
        self.motors[2].setVelocity(-p[2])
        self.motors[3].setVelocity(p[3])

    def take_off(self):
        """
        State machine to perform automatic take-off and stabilization of the drone.
        Ascends progressively until exceeding MAX_HEIGHT and waits WAITING_TIME seconds.
        """
        # If no real sensors are available (mock or testing mode), avoid infinite loop
        if self.gps is None or self.imu is None:
            self.the_drone_took_off = True
            self.alt = self.HEIGHT_INITIAL
            self.height_desired = self.HEIGHT_INITIAL
            if self.verbose >= 1:
                print(f"[Crazyflie] Modo testing/mock: despegue simulado a {self.HEIGHT_INITIAL:.2f}m.")
            return

        if self.verbose >= 1:
            print(f"[Crazyflie] Iniciando secuencia de despegue (altitud objetivo: {self.MAX_HEIGHT:.2f}m)...")

        while not self.the_drone_took_off:
            if self.alt < self.MAX_HEIGHT:
                if self.alt > self.height_desired:
                    self.height_desired += self.HEIGHT_INCREASE
                    self.timestamp_take_off = 0.0
            else:
                if self.timestamp_take_off == 0.0:
                    self.timestamp_take_off = self.getTime()
                    if self.verbose >= 1:
                        print(f"[Crazyflie] Altitud {self.alt:.2f}m alcanzada. Esperando estabilización ({self.WAITING_TIME:.1f}s)...")

                if (self.getTime() - self.timestamp_take_off) > self.WAITING_TIME:
                    self.timestamp_take_off = 0.0
                    self.the_drone_took_off = True
                    self.height_desired = self.alt
                    if self.verbose >= 1:
                        print(f"[Crazyflie] CRAZYFLIE TOOK OFF ! (Altitud: {self.alt:.2f}m, estabilizado tras {self.WAITING_TIME:.1f}s)")

            motor_power = self.PID_crazyflie.pid(
                self.dt,
                0.0,
                0.0,
                0.0,
                self.height_desired,
                self.roll,
                self.pitch,
                self.yaw_rate,
                self.alt,
                self.v_x,
                self.v_y,
            )
            self._setup_motors_velocity(motor_power)
            super().step(self.timestep)

            self.get_observations()
            self.past_time = self.getTime()
            self.past_x_global = self.x_global
            self.past_y_global = self.y_global
            self.past_z_global = self.z_global

    def transform_velocity(self, v_x_global: float, v_y_global: float) -> tuple[float, float]:
        """
        Transforms global XY plane velocities to the drone's body-fixed reference frame.
        """
        cos_yaw = cos(self.yaw)
        sin_yaw = sin(self.yaw)
        self.v_x = float(v_x_global * cos_yaw + v_y_global * sin_yaw)
        self.v_y = float(-v_x_global * sin_yaw + v_y_global * cos_yaw)
        return self.v_x, self.v_y

    def update_lineal_velocity(self):
        """
        Updates global coordinates and computes linear velocities along the X, Y, and Z axes.
        """
        if self.gps is not None and hasattr(self.gps, "getValues"):
            pos = self.gps.getValues()
            self.x_global = float(pos[0])
            self.y_global = float(pos[1])
            self.z_global = float(pos[2])
            self.alt = self.z_global
        else:
            return

        dt = self.dt if self.dt > 0 else (float(self.timestep) / 1000.0)
        v_x_global = (self.x_global - self.past_x_global) / dt
        v_y_global = (self.y_global - self.past_y_global) / dt
        v_z_global = (self.z_global - self.past_z_global) / dt

        self.past_x_global = self.x_global
        self.past_y_global = self.y_global
        self.past_z_global = self.z_global

        self.transform_velocity(v_x_global, v_y_global)
        self.v_z = float(v_z_global)
        self.lineal_velocity = np.array([self.v_x, self.v_y, self.v_z], dtype=np.float32)

    def get_observations(self) -> np.ndarray:
        """
        Reads and normalizes sensor data to the range [-1.0, 1.0].

        Returns:
            np.ndarray: 11-element continuous vector:
                [roll, pitch, yaw_rate, v_x, v_y, v_z, altitude, range_front, range_back, range_right, range_left]
        """
        if self.first_time:
            if self.gps is not None and hasattr(self.gps, "getValues"):
                vals = self.gps.getValues()
                self.past_x_global = float(vals[0])
                self.past_y_global = float(vals[1])
                self.past_z_global = float(vals[2])
            self.past_time = self.getTime() if hasattr(self, "getTime") else 0.0
            if self.gyro is not None and hasattr(self.gyro, "getValues"):
                self.angular_velocity = np.array(self.gyro.getValues(), dtype=np.float32)
            self.first_time = False

        curr_time = self.getTime() if hasattr(self, "getTime") else (self.past_time + 0.032)
        self.dt = curr_time - self.past_time
        if self.dt <= 0:
            self.dt = float(self.timestep) / 1000.0

        # Read IMU orientation
        if self.imu is not None and hasattr(self.imu, "getRollPitchYaw"):
            rpy = self.imu.getRollPitchYaw()
            self.roll = float(rpy[0])
            self.pitch = float(rpy[1])
            self.yaw = float(rpy[2])

        # Read Gyro angular velocity
        if self.gyro is not None and hasattr(self.gyro, "getValues"):
            gyro_vals = self.gyro.getValues()
            self.yaw_rate = float(gyro_vals[2])
            self.angular_velocity = np.array(gyro_vals, dtype=np.float32)

        # Update linear velocities
        self.update_lineal_velocity()

        # ToF distance sensors
        if self.range_front is not None and hasattr(self.range_front, "getValue"):
            self.dist_front = float(self.range_front.getValue())
        if self.range_back is not None and hasattr(self.range_back, "getValue"):
            self.dist_back = float(self.range_back.getValue())
        if self.range_right is not None and hasattr(self.range_right, "getValue"):
            self.dist_right = float(self.range_right.getValue())
        if self.range_left is not None and hasattr(self.range_left, "getValue"):
            self.dist_left = float(self.range_left.getValue())

        # Normalize variables to [-1.0, 1.0]
        roll_norm = normalize_to_range(self.roll, -pi, pi, -1.0, 1.0, clip=True)
        pitch_norm = normalize_to_range(self.pitch, -pi / 2.0, pi / 2.0, -1.0, 1.0, clip=True)
        yaw_rate_norm = normalize_to_range(self.yaw_rate, -pi, pi, -1.0, 1.0, clip=True)

        vx_norm = normalize_to_range(self.v_x, -self.MAX_OBS_VELOCITY, self.MAX_OBS_VELOCITY, -1.0, 1.0, clip=True)
        vy_norm = normalize_to_range(self.v_y, -self.MAX_OBS_VELOCITY, self.MAX_OBS_VELOCITY, -1.0, 1.0, clip=True)
        vz_norm = normalize_to_range(self.v_z, -self.MAX_OBS_VELOCITY, self.MAX_OBS_VELOCITY, -1.0, 1.0, clip=True)

        alt_norm = normalize_to_range(self.alt, 0.1, 5.0, -1.0, 1.0, clip=True)

        rf_norm = normalize_to_range(self.dist_front, 2.0, 2000.0, -1.0, 1.0, clip=True)
        rb_norm = normalize_to_range(self.dist_back, 2.0, 2000.0, -1.0, 1.0, clip=True)
        rr_norm = normalize_to_range(self.dist_right, 2.0, 2000.0, -1.0, 1.0, clip=True)
        rl_norm = normalize_to_range(self.dist_left, 2.0, 2000.0, -1.0, 1.0, clip=True)

        obs = np.array(
            [
                roll_norm,
                pitch_norm,
                yaw_rate_norm,
                vx_norm,
                vy_norm,
                vz_norm,
                alt_norm,
                rf_norm,
                rb_norm,
                rr_norm,
                rl_norm,
            ],
            dtype=np.float32,
        )
        return obs

    def get_default_observation(self) -> np.ndarray:
        """Returns a zero vector with the dimension of the observation space."""
        dim = self.observation_space.shape[0] if hasattr(self, "observation_space") else 11
        return np.zeros(dim, dtype=np.float32)

    def _denormalize_action(self, action: np.ndarray) -> tuple[float, float, float, float]:
        """
        Converts the normalized action vector [-1.0, 1.0]^4 into physical commands:
        - vx_cmd (m/s): desired longitudinal velocity
        - vy_cmd (m/s): desired lateral velocity
        - yaw_rate_cmd (rad/s): desired yaw angular velocity
        - desired_altitude (m): updated altitude according to integrated vertical velocity
        """
        act = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        vx_cmd = float(act[0] * self.MAX_FORWARD_VELOCITY)
        vy_cmd = float(act[1] * self.MAX_LATERAL_VELOCITY)
        yaw_rate_cmd = float(act[2] * self.MAX_YAW_RATE)

        # Relative altitude variation (Option 2: vertical velocity integrated with dt)
        delta_h = float(act[3] * self.MAX_VERTICAL_VELOCITY * self.dt)
        self.height_desired = float(
            np.clip(self.height_desired + delta_h, self.MIN_HEIGHT, self.MAX_FLIGHT_HEIGHT)
        )

        return vx_cmd, vy_cmd, yaw_rate_cmd, self.height_desired

    def step(self, action: np.ndarray):
        """
        Advances the simulation by one step applying the specified continuous action.

        Args:
            action (np.ndarray): 4-component continuous vector in [-1.0, 1.0].

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        self.episode_step += 1
        self.current_action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        vx_cmd, vy_cmd, yaw_rate_cmd, height_cmd = self._denormalize_action(action)

        # Compute motor power with the velocity and altitude PID
        if self.PID_crazyflie is not None:
            motor_power = self.PID_crazyflie.pid(
                self.dt,
                vx_cmd,
                vy_cmd,
                yaw_rate_cmd,
                height_cmd,
                self.roll,
                self.pitch,
                self.yaw_rate,
                self.alt,
                self.v_x,
                self.v_y,
            )
            self._setup_motors_velocity(motor_power)

        super().step(self.timestep)

        # Read new states
        obs = self.get_observations()
        self.past_time = self.getTime() if hasattr(self, "getTime") else (self.past_time + self.dt)

        # Evaluate episode termination and reward
        reward = self.get_reward()
        self.episode_score += reward
        terminated = self.is_done()
        truncated = self.episode_step >= self.max_episode_steps
        info = self.get_info()

        # Optional trajectory logging to CSV
        if self.enable_trajectory_logging and self.drone_trajectory_file is not None:
            self.record_trajectory()

        # Store current action as previous for the next step
        self.prev_action = np.copy(self.current_action)

        return obs, reward, terminated, truncated, info

    def reset(self, seed: int | None = None, options: dict | None = None):
        """
        Resets the Webots simulation and the internal state of the Gymnasium environment.

        Returns:
            tuple: (observation, info)
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if hasattr(self, "simulationResetPhysics"):
            self.simulationResetPhysics()
        if hasattr(self, "simulationReset"):
            self.simulationReset()

        super().step(self.timestep)

        self._initialization()
        super().step(self.timestep)

        self.take_off()

        # Initialize trajectory file if logging is enabled
        if self.enable_trajectory_logging and self.drone_trajectory_path is not None:
            timestamp = int(time.time())
            file_name = f"trajectory_x_{self.x_global:.2f}_y_{self.y_global:.2f}_t_{timestamp}.csv"
            self.set_trajectory_file_name(file_name)

        self.x_pos_initial = self.x_global
        self.y_pos_initial = self.y_global

        obs = self.get_observations()
        info = self.get_info()
        return obs, info

    def set_trajectory_path(self, path_dir: str):
        """Sets the directory path for saving CSV trajectory files."""
        self.drone_trajectory_path = path_dir
        if not os.path.exists(path_dir):
            os.makedirs(path_dir, exist_ok=True)

    def set_trajectory_file_name(self, file_name: str):
        """Configures the CSV file and writes its header."""
        if self.drone_trajectory_path is None:
            return
        self.drone_trajectory_file = os.path.join(self.drone_trajectory_path, file_name)
        with open(self.drone_trajectory_file, "w", encoding="utf-8") as f:
            f.write("time,x,y,z\n")

    def record_trajectory(self):
        """Records the current position (time, x, y, z) in the configured CSV file."""
        if self.drone_trajectory_file is None:
            return
        curr_time = self.getTime() if hasattr(self, "getTime") else 0.0
        with open(self.drone_trajectory_file, "a", encoding="utf-8") as f:
            f.write(f"{curr_time},{self.x_global:.4f},{self.y_global:.4f},{self.z_global:.4f}\n")

    def get_info(self) -> dict:
        """Additional information dictionary for the current step."""
        return {
            "is_success": self.is_success,
            "altitude": self.alt,
            "desired_altitude": self.height_desired,
            "episode_step": self.episode_step,
        }

    def get_reward(self) -> float:
        """
        Base reward. Overridden in specific environments such as CornerEnvContinuous.
        """
        return 0.0

    def is_done(self) -> bool:
        """
        Base stopping condition. Overridden in specific environments such as CornerEnvContinuous.
        """
        return False

    def render(self, mode: str = "human"):
        """Passive rendering managed by the Webots GUI."""
        pass

    def close(self):
        """Cleanly closes environment resources."""
        pass
