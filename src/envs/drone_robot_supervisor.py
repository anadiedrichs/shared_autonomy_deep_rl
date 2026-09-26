# -*- coding: utf-8 -*-
"""
Entorno base de Gymnasium y controlador Webots Supervisor para cuadricóptero Crazyflie.

Adaptación a espacio de acciones continuo y 11 observaciones continuas basada en el
trabajo previo de tesis (Diedrichs, 2025, drone-deep-rl).
"""

import os
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

# Importación desacoplada del Supervisor de Webots para permitir pruebas unitarias y CI sin GUI
try:
    from controller import Keyboard, Supervisor
except ImportError:  # pragma: no cover
    class Supervisor:
        """Clase dummy que emula la API mínima de Webots Supervisor para pruebas fuera del simulador."""

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

    class Keyboard:
        """Clase dummy que emula el dispositivo de teclado de Webots."""

        def __init__(self):
            pass

        def enable(self, timestep: int):
            pass

        def getKey(self) -> int:
            return -1


from src.utils.pid_controller import pid_velocity_fixed_height_controller
from src.utils.utilities import normalize_to_range


class DroneRobotSupervisor(Supervisor, gym.Env if gym is not None else object):
    """
    Entorno base Gymnasium acoplado a Webots para el cuadricóptero Crazyflie.

    Espacio de acciones continuo (4 dimensiones en [-1.0, 1.0]):
        - a[0]: comando de avance longitudinal vx (normalizado)
        - a[1]: comando de desplazamiento lateral vy (normalizado)
        - a[2]: comando de velocidad angular de guiñada yaw_rate (normalizado)
        - a[3]: comando de velocidad vertical / variación de altura (normalizado)

    Espacio de observaciones continuo (11 dimensiones en [-1.0, 1.0]):
        - [0]: roll (rad normalizado desde [-pi, pi])
        - [1]: pitch (rad normalizado desde [-pi/2, pi/2])
        - [2]: yaw_rate (rad/s normalizado desde [-pi, pi])
        - [3]: v_x (m/s en marco del dron normalizado desde [-MAX_OBS_VELOCITY, MAX_OBS_VELOCITY])
        - [4]: v_y (m/s en marco del dron normalizado desde [-MAX_OBS_VELOCITY, MAX_OBS_VELOCITY])
        - [5]: v_z (m/s velocidad vertical normalizada desde [-MAX_OBS_VELOCITY, MAX_OBS_VELOCITY])
        - [6]: altitude (metros normalizado desde [0.1, 5.0])
        - [7]: range_front (mm normalizado desde [2.0, 2000.0])
        - [8]: range_back (mm normalizado desde [2.0, 2000.0])
        - [9]: range_right (mm normalizado desde [2.0, 2000.0])
        - [10]: range_left (mm normalizado desde [2.0, 2000.0])
    """

    # Límites físicos de vuelo y control
    MAX_FORWARD_VELOCITY = 0.09  # Velocidad longitudinal máxima en m/s
    MAX_LATERAL_VELOCITY = 0.09  # Velocidad lateral máxima en m/s
    MAX_YAW_RATE = 0.5  # Velocidad angular de guiñada máxima en rad/s (dinámica)
    MAX_VERTICAL_VELOCITY = 0.05  # Velocidad máxima de ascenso/descenso en m/s
    MAX_OBS_VELOCITY = 1.0  # Rango físico de velocidad para normalización de observaciones (m/s)

    # Parámetros de altitud
    MIN_HEIGHT = 0.20  # Altura mínima permitida en vuelo (metros)
    MAX_HEIGHT = 0.70  # Altura máxima permitida en vuelo (metros)
    HEIGHT_INITIAL = 0.30  # Altura objetivo tras despegue (metros)
    HEIGHT_INCREASE = 0.05  # Incremento en rampa de despegue (metros)
    WAITING_TIME = 5.0  # Tiempo de estabilización en segundos tras despegue

    def __init__(
        self,
        max_episode_steps: int = 10_000,
        enable_trajectory_logging: bool = False,
        verbose: int = 1,
    ):
        """
        Inicializa el supervisor del robot Crazyflie.

        Args:
            max_episode_steps: Límite máximo de pasos por episodio.
            enable_trajectory_logging: Habilita el guardado en disco de la trayectoria CSV.
            verbose: Nivel de detalle en consola (0: silencioso para RL masivo, 1: transiciones clave y despegue, 2: depuración extendida).
        """
        super().__init__()
        self.max_episode_steps = max_episode_steps
        self.enable_trajectory_logging = enable_trajectory_logging
        self.verbose = int(verbose)
        self.render_mode = None

        # Definición formal de espacios Gymnasium
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

        # Variables de estado y odometría
        self.timestep = int(self.getBasicTimeStep()) if hasattr(self, "getBasicTimeStep") else 32
        self.episode_step = 0
        self.episode_score = 0.0
        self.first_time = True
        self.dt = float(self.timestep) / 1000.0
        self.past_time = 0.0

        # Posiciones globales
        self.x_global = 0.0
        self.y_global = 0.0
        self.z_global = 0.0
        self.past_x_global = 0.0
        self.past_y_global = 0.0
        self.past_z_global = 0.0
        self.x_pos_initial = 0.0
        self.y_pos_initial = 0.0

        # Velocidades y orientaciones
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

        # Sensores de distancia ToF (en milímetros)
        self.dist_front = 2000.0
        self.dist_back = 2000.0
        self.dist_right = 2000.0
        self.dist_left = 2000.0

        # Estado de vuelo y control
        self.height_desired = self.HEIGHT_INITIAL
        self.the_drone_took_off = False
        self.timestamp_take_off = 0.0
        self.is_success = False
        self.terminated = False
        self.truncated = False

        # Registro de trayectorias
        self.drone_trajectory_path = None
        self.drone_trajectory_file = None

        # Controladores y dispositivos
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
        Inicializa dispositivos, sensores, actuadores y controlador PID del Crazyflie.
        """
        if hasattr(self, "getBasicTimeStep"):
            self.timestep = int(self.getBasicTimeStep())
        else:
            self.timestep = 32

        self.dt = float(self.timestep) / 1000.0
        self.PID_crazyflie = pid_velocity_fixed_height_controller()

        # Teclado para eventual teleoperación manual
        self._keyboard = Keyboard()
        if hasattr(self._keyboard, "enable"):
            self._keyboard.enable(self.timestep)

        # Sensores integrados en el nodo Crazyflie
        self.imu = self.getDevice("inertial_unit") if hasattr(self, "getDevice") else None
        if self.imu is not None and hasattr(self.imu, "enable"):
            self.imu.enable(self.timestep)

        self.gps = self.getDevice("gps") if hasattr(self, "getDevice") else None
        if self.gps is not None and hasattr(self.gps, "enable"):
            self.gps.enable(self.timestep)

        self.gyro = self.getDevice("gyro") if hasattr(self, "getDevice") else None
        if self.gyro is not None and hasattr(self.gyro, "enable"):
            self.gyro.enable(self.timestep)

        self.camera = self.getDevice("camera") if hasattr(self, "getDevice") else None
        if self.camera is not None and hasattr(self.camera, "enable"):
            self.camera.enable(self.timestep)

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

        # Referencia al robot Crazyflie
        self.robot = self.getFromDef("crazyflie") if hasattr(self, "getFromDef") else None

        # Motores
        self._setup_motors()

    def _setup_motors(self):
        """
        Configura los motores del cuadricóptero en modo control de velocidad.
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
        Aplica la potencia calculada a cada uno de los 4 motores del Crazyflie.
        Los motores 0 (m1) y 2 (m3) requieren inversión de signo por su sentido de giro.
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
        Máquina de estados para realizar el despegue automático y estabilización del dron.
        Asciende progresivamente hasta superar MAX_HEIGHT y espera WAITING_TIME segundos.
        """
        # Si no hay sensores reales disponibles (modo mock o testing), evitar bucle infinito
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
        Transforma las velocidades globales del plano XY al marco de referencia del cuerpo del dron (body-fixed frame).
        """
        cos_yaw = cos(self.yaw)
        sin_yaw = sin(self.yaw)
        self.v_x = float(v_x_global * cos_yaw + v_y_global * sin_yaw)
        self.v_y = float(-v_x_global * sin_yaw + v_y_global * cos_yaw)
        return self.v_x, self.v_y

    def update_lineal_velocity(self):
        """
        Actualiza las coordenadas globales y calcula las velocidades lineales en los ejes X, Y y Z.
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
        Lee y normaliza los datos de los sensores al rango [-1.0, 1.0].

        Returns:
            np.ndarray: Vector de 11 valores continuos:
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

        # Lectura de orientación IMU
        if self.imu is not None and hasattr(self.imu, "getRollPitchYaw"):
            rpy = self.imu.getRollPitchYaw()
            self.roll = float(rpy[0])
            self.pitch = float(rpy[1])
            self.yaw = float(rpy[2])

        # Lectura de velocidad angular Gyro
        if self.gyro is not None and hasattr(self.gyro, "getValues"):
            gyro_vals = self.gyro.getValues()
            self.yaw_rate = float(gyro_vals[2])
            self.angular_velocity = np.array(gyro_vals, dtype=np.float32)

        # Actualización de velocidades lineales
        self.update_lineal_velocity()

        # Sensores de distancia ToF
        if self.range_front is not None and hasattr(self.range_front, "getValue"):
            self.dist_front = float(self.range_front.getValue())
        if self.range_back is not None and hasattr(self.range_back, "getValue"):
            self.dist_back = float(self.range_back.getValue())
        if self.range_right is not None and hasattr(self.range_right, "getValue"):
            self.dist_right = float(self.range_right.getValue())
        if self.range_left is not None and hasattr(self.range_left, "getValue"):
            self.dist_left = float(self.range_left.getValue())

        # Normalización de variables a [-1.0, 1.0]
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
        """Devuelve un vector de ceros con la dimensión del espacio de observación."""
        dim = self.observation_space.shape[0] if hasattr(self, "observation_space") else 11
        return np.zeros(dim, dtype=np.float32)

    def _denormalize_action(self, action: np.ndarray) -> tuple[float, float, float, float]:
        """
        Convierte el vector de acción normalizado [-1.0, 1.0]^4 a comandos físicos reales:
        - vx_cmd (m/s): velocidad longitudinal deseada
        - vy_cmd (m/s): velocidad lateral deseada
        - yaw_rate_cmd (rad/s): velocidad angular de guiñada deseada
        - desired_altitude (m): altitud actualizada según velocidad vertical integrada
        """
        act = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        vx_cmd = float(act[0] * self.MAX_FORWARD_VELOCITY)
        vy_cmd = float(act[1] * self.MAX_LATERAL_VELOCITY)
        yaw_rate_cmd = float(act[2] * self.MAX_YAW_RATE)

        # Variación relativa de altura (Opción 2: velocidad vertical integrada con dt)
        delta_h = float(act[3] * self.MAX_VERTICAL_VELOCITY * self.dt)
        self.height_desired = float(
            np.clip(self.height_desired + delta_h, self.MIN_HEIGHT, self.MAX_HEIGHT)
        )

        return vx_cmd, vy_cmd, yaw_rate_cmd, self.height_desired

    def step(self, action: np.ndarray):
        """
        Avanza la simulación un paso aplicando la acción continua indicada.

        Args:
            action (np.ndarray): Vector continuo de 4 componentes en [-1.0, 1.0].

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        self.episode_step += 1

        vx_cmd, vy_cmd, yaw_rate_cmd, height_cmd = self._denormalize_action(action)

        # Cálculo de potencias con el PID de velocidad y altitud
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

        # Lectura de nuevos estados
        obs = self.get_observations()
        self.past_time = self.getTime() if hasattr(self, "getTime") else (self.past_time + self.dt)

        # Evaluación de fin de episodio y recompensa
        reward = self.get_reward()
        self.episode_score += reward
        terminated = self.is_done()
        truncated = self.episode_step >= self.max_episode_steps
        info = self.get_info()

        # Registro opcional de trayectoria en CSV
        if self.enable_trajectory_logging and self.drone_trajectory_file is not None:
            self.record_trajectory()

        return obs, reward, terminated, truncated, info

    def reset(self, seed: int | None = None, options: dict | None = None):
        """
        Reinicia la simulación de Webots y el estado interno del entorno Gymnasium.

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

        self.episode_step = 0
        self.episode_score = 0.0
        self.the_drone_took_off = False
        self.first_time = True
        self.is_success = False
        self.terminated = False
        self.truncated = False
        self.height_desired = self.HEIGHT_INITIAL

        self._initialization()
        super().step(self.timestep)

        self.take_off()

        # Inicialización de archivo de trayectoria si el logging está activado
        if self.enable_trajectory_logging and self.drone_trajectory_path is not None:
            timestamp = int(time.time())
            file_name = f"trajectory_x_{self.x_global:.2f}_y_{self.y_global:.2f}_t_{timestamp}.csv"
            self.set_trajectory_file_name(file_name)

        obs = self.get_observations()
        info = self.get_info()
        return obs, info

    def set_trajectory_path(self, path_dir: str):
        """Establece el directorio de guardado para los archivos de trayectoria CSV."""
        self.drone_trajectory_path = path_dir
        if not os.path.exists(path_dir):
            os.makedirs(path_dir, exist_ok=True)

    def set_trajectory_file_name(self, file_name: str):
        """Configura el archivo CSV y escribe su encabezado."""
        if self.drone_trajectory_path is None:
            return
        self.drone_trajectory_file = os.path.join(self.drone_trajectory_path, file_name)
        with open(self.drone_trajectory_file, "w", encoding="utf-8") as f:
            f.write("time,x,y,z\n")

    def record_trajectory(self):
        """Registra la posición actual (tiempo, x, y, z) en el archivo CSV configurado."""
        if self.drone_trajectory_file is None:
            return
        curr_time = self.getTime() if hasattr(self, "getTime") else 0.0
        with open(self.drone_trajectory_file, "a", encoding="utf-8") as f:
            f.write(f"{curr_time},{self.x_global:.4f},{self.y_global:.4f},{self.z_global:.4f}\n")

    def get_info(self) -> dict:
        """Diccionario de información adicional del paso actual."""
        return {
            "is_success": self.is_success,
            "altitude": self.alt,
            "desired_altitude": self.height_desired,
            "episode_step": self.episode_step,
        }

    def get_reward(self) -> float:
        """
        Recompensa base. Se sobreescribe en entornos específicos como CornerEnvContinuous.
        """
        return 0.0

    def is_done(self) -> bool:
        """
        Condición de parada base. Se sobreescribe en entornos específicos como CornerEnvContinuous.
        """
        return False

    def render(self, mode: str = "human"):
        """Renderizado pasivo administrado por la GUI de Webots."""
        pass

    def close(self):
        """Cierre ordenado de recursos del entorno."""
        pass
