# -*- coding: utf-8 -*-
"""
Heuristic-based deterministic pilot for the CornerEnvContinuous environment.

Supports two main operating approaches depending on the 'mode' parameter:
  1. 'open_loop' (or 'enfoque1', 'fixed'):
     Emits a constant fixed open-loop action toward the target corner.
     Ideal for direct kinematic validation and straight-line experiments.
  2. 'proportional' (or 'enfoque2', 'closed_loop'):
     Closed-loop proportional controller (P-Controller) over 3D position.
     Computes the Cartesian error vector toward the cone, projects it onto the drone's
     body frame, and modulates approach speed with automatic braking inside the
     success radius to maintain stable hover.
"""

from math import cos, sin, sqrt
import numpy as np

from src.pilots.base_pilot import BasePilot


class HeuristicPilot(BasePilot):
    """
    Deterministic heuristic pilot with multiple control modalities.
    """

    SUPPORTED_MODES = (
        "open_loop",
        "proportional",
        "enfoque1",
        "enfoque2",
        "fixed",
        "closed_loop",
    )

    # Coordinates of the 4 cones in the 2x2m room (reference altitude: 0.55m)
    CORNER_COORDINATES = {
        "cone_1": np.array([0.9, 0.9, 0.55], dtype=np.float32),
        "cone_2": np.array([-0.9, 0.9, 0.55], dtype=np.float32),
        "cone_3": np.array([-0.9, -0.9, 0.55], dtype=np.float32),
        "cone_4": np.array([0.9, -0.9, 0.55], dtype=np.float32),
    }

    def __init__(
        self,
        mode: str = "proportional",
        corner_name: str = "cone_1",
        target_position: np.ndarray | tuple | list | None = None,
        env: object | None = None,
        fixed_action: np.ndarray | tuple | list | None = None,
        kp_xy: float = 2.5,
        kp_z: float = 3.0,
        target_altitude: float = 0.55,
        target_radius: float = 0.15,
        target_speed_ratio: float = 0.90,
    ):
        """
        Initializes the deterministic heuristic pilot.

        Args:
            mode: Operating mode ('open_loop'/'enfoque1' or 'proportional'/'enfoque2').
            corner_name: Target cone identifier ("cone_1", "cone_2", "cone_3", "cone_4").
            target_position: Optional Cartesian target position [x, y, z] (overrides corner_name if specified).
            env: Optional reference to the CornerEnvContinuous environment for direct odometry/GPS reading.
            fixed_action: Continuous action vector [vx, vy, yaw_rate, vz] for open_loop mode.
            kp_xy: Proportional gain for horizontal velocity in closed loop.
            kp_z: Proportional gain for altitude variation in closed loop.
            target_altitude: Target altitude in meters (default 0.55 m to exceed the 0.50 m threshold).
            target_radius: Radius in meters within which deceleration is modulated (default 0.15 m).
            target_speed_ratio: Fraction of maximum speed in open loop (in [0.0, 1.0]).
        """
        mode_lower = mode.lower()
        if mode_lower not in self.SUPPORTED_MODES:
            raise ValueError(
                f"Modo '{mode}' no soportado. Opciones válidas: {self.SUPPORTED_MODES}"
            )

        self.mode = mode_lower
        self.is_open_loop = self.mode in ("open_loop", "enfoque1", "fixed")
        self.corner_name = corner_name
        self.env = env
        self.kp_xy = float(kp_xy)
        self.kp_z = float(kp_z)
        self.target_altitude = float(target_altitude)
        self.target_radius = float(target_radius)
        self.target_speed_ratio = float(np.clip(target_speed_ratio, 0.1, 1.0))

        # Configure Cartesian target position [x, y, z]
        if target_position is not None:
            self.target_position = np.asarray(target_position, dtype=np.float32)
        else:
            self.target_position = self.CORNER_COORDINATES.get(
                corner_name, np.array([0.9, 0.9, self.target_altitude], dtype=np.float32)
            ).copy()
        # Ensure desired target altitude
        self.target_position[2] = self.target_altitude

        # Configure fixed action for Approach 1 (open_loop)
        if fixed_action is not None:
            self.fixed_action = np.clip(np.asarray(fixed_action, dtype=np.float32), -1.0, 1.0)
        else:
            self.fixed_action = self._calculate_default_fixed_action()

    def _calculate_default_fixed_action(self) -> np.ndarray:
        """
        Computes the default constant open-loop action toward the target corner.
        Accounts for the drone's initial orientation in the Webots world (yaw ≈ π/2,
        where the drone's front (local +X axis) points north (global +Y) and its
        right side (local -Y axis) points east (global +X)).
        """
        dx = float(self.target_position[0])
        dy = float(self.target_position[1])

        # In the Webots world crazyflie.wbt, the drone starts rotated ~90° (π/2 rad) toward north (global +Y).
        # Its front (+X body axis) points toward global +Y, and its right side (-Y body axis) toward global +X (+east).
        yaw_init = np.pi / 2.0
        if self.env is not None and getattr(self.env, "the_drone_took_off", False):
            yaw_init = float(getattr(self.env, "yaw", np.pi / 2.0))
        cos_y = cos(yaw_init)
        sin_y = sin(yaw_init)

        dx_body = dx * cos_y + dy * sin_y
        dy_body = -dx * sin_y + dy * cos_y

        vx_sign = 1.0 if dx_body >= 0.0 else -1.0
        vy_sign = 1.0 if dy_body >= 0.0 else -1.0

        # vz = 0.60 to generate steady climb from 0.30m to 0.55m during flight
        return np.array(
            [
                vx_sign * self.target_speed_ratio,
                vy_sign * self.target_speed_ratio,
                0.0,
                0.60,
            ],
            dtype=np.float32,
        )

    def _get_open_loop_action(self, observation: np.ndarray) -> np.ndarray:
        """
        Approach 1: Constant fixed open-loop action.
        """
        return np.copy(self.fixed_action)

    def _get_proportional_action(
        self, observation: np.ndarray, env: object | None = None
    ) -> np.ndarray:
        """
        Approach 2: Closed-loop proportional controller with braking and altitude hold.
        """
        active_env = env if env is not None else self.env

        # 1. Retrieve global drone odometry
        if active_env is not None and hasattr(active_env, "x_global"):
            curr_x = float(active_env.x_global)
            curr_y = float(active_env.y_global)
            curr_z = float(getattr(active_env, "alt", getattr(active_env, "z_global", 0.30)))
            curr_yaw = float(getattr(active_env, "yaw", 0.0))
        else:
            # Estimate from observation if the environment is not provided
            # obs[6] = normalized altitude in [0.1, 5.0]
            obs = np.asarray(observation, dtype=np.float32)
            alt_norm = float(obs[6]) if len(obs) > 6 else 0.0
            curr_z = 0.1 + (alt_norm + 1.0) / 2.0 * (5.0 - 0.1)
            curr_x = 0.0
            curr_y = 0.0
            curr_yaw = 0.0

        # 2. Cartesian errors toward the target
        err_x = float(self.target_position[0] - curr_x)
        err_y = float(self.target_position[1] - curr_y)
        err_z = float(self.target_position[2] - curr_z)
        dist_xy = float(sqrt(err_x * err_x + err_y * err_y))

        # 3. Transform horizontal error to the drone's body reference frame (body frame)
        cos_yaw = cos(curr_yaw)
        sin_yaw = sin(curr_yaw)
        err_x_body = float(err_x * cos_yaw + err_y * sin_yaw)
        err_y_body = float(-err_x * sin_yaw + err_y * cos_yaw)

        # 4. Proportional modulation of horizontal commands
        if dist_xy > self.target_radius:
            # In free flight toward the cone: proportional command bounded to [-1.0, 1.0]
            cmd_vx = float(np.clip(self.kp_xy * err_x_body, -1.0, 1.0))
            cmd_vy = float(np.clip(self.kp_xy * err_y_body, -1.0, 1.0))
        else:
            # Within the target radius: smooth braking zone for stable hover
            brake_scale = max(dist_xy / self.target_radius, 0.1)
            cmd_vx = float(np.clip(self.kp_xy * err_x_body * brake_scale, -0.25, 0.25))
            cmd_vy = float(np.clip(self.kp_xy * err_y_body * brake_scale, -0.25, 0.25))

        # 5. Proportional vertical command toward the target altitude (0.55m)
        cmd_vz = float(np.clip(self.kp_z * err_z, -1.0, 1.0))

        # 6. Neutral yaw (maintains stable forward heading without sharp turns)
        cmd_yaw = 0.0

        return np.array([cmd_vx, cmd_vy, cmd_yaw, cmd_vz], dtype=np.float32)

    def get_action(
        self, observation: np.ndarray, env: object | None = None
    ) -> np.ndarray:
        """
        Computes the heuristic pilot action for the current observation.

        Args:
            observation: 11-dimensional continuous observation vector.
            env: Optional reference to the environment for reading position and yaw in closed loop.

        Returns:
            np.ndarray: Continuous action vector [vx, vy, yaw_rate, vz] in [-1.0, 1.0]^4.
        """
        if self.is_open_loop:
            return self._get_open_loop_action(observation)
        return self._get_proportional_action(observation, env=env)

    def choose_action(
        self, observation: np.ndarray, env: object | None = None
    ) -> tuple[np.ndarray, list]:
        """
        Compatibility alias for the synthetic pilots API from the thesis (Diedrichs, 2025).

        Returns:
            tuple: (selected action, empty list of auxiliary states).
        """
        return self.get_action(observation, env=env), []
