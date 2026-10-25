# -*- coding: utf-8 -*-
"""
Continuous Crazyflie environment in a 2x2m room with a corner goal (CornerEnv).

Continuous port of the SimpleCornerEnvRS10 class from the thesis (Diedrichs, 2025, drone-deep-rl).
"""

from math import sqrt
import numpy as np

from src.envs.drone_robot_supervisor import DroneRobotSupervisor
from src.rewards.reward_goal_directed import compute_goal_directed_reward


class CornerEnvContinuous(DroneRobotSupervisor):
    """
    Continuous navigation environment toward a target cone in a corner of a 2x2m room.
    """

    # Target identifier and scenario thresholds
    CORNER_NAME = "cone_1"
    DISTANCE_THRESHOLD = 0.15  # Minimum Euclidean distance to the target (15 cm)
    MIN_DIST_OBSTACLES = 100  # Minimum distance to walls in mm (100 mm = 0.10 m)
    MIN_EPISODE_SCORE = -1_000.0  # Minimum cumulative score before truncation
    MAX_EPISODE_SCORE = 1_000.0  # Maximum cumulative score before truncation
    HEIGHT_INITIAL = 0.50  # Target altitude after take-off (meters)

    # Spatial bounds of the room (2x2 meters centered at [0, 0])
    X_MIN, X_MAX = -1.0, 1.0
    Y_MIN, Y_MAX = -1.0, 1.0

    def __init__(
        self,
        max_episode_steps: int = 1_000,
        corner_name: str = "cone_1",
        enable_trajectory_logging: bool = False,
        verbose: int = 1,
    ):
        """
        Initializes the continuous corner room environment.

        Args:
            max_episode_steps: Maximum number of steps per episode (default 1000).
            corner_name: DEF identifier of the target node in the Webots world (default "cone_1").
            enable_trajectory_logging: If True, enables saving the CSV trajectory to disk.
            verbose: Console detail level (0: silent for massive RL, 1: key transitions and take-off, 2: extended debug).
        """
        self.corner_name = corner_name
        self.target_node = None
        self.dist_min_target = 0.0
        self.prev_dist_min_target = 0.0
        self.corner = None
        self.x_init = 0.0
        self.y_init = 0.0

        # Default target position for unit testing without Webots
        self._target_pos_fallback = np.array([0.8, 0.8, 0.0], dtype=np.float32)

        super().__init__(
            max_episode_steps=max_episode_steps,
            enable_trajectory_logging=enable_trajectory_logging,
            verbose=verbose,
        )

        self._init_targets()
        self._initialization()

    def _init_targets(self):
        """
        Obtains the reference to the target cone node in Webots via its DEF identifier.
        """
        if hasattr(self, "getFromDef"):
            self.target_node = self.getFromDef(self.corner_name)

    def put_drone_in_the_center(self):
        """
        Positions the drone at the center of the arena (coordinates [0, 0, z]).
        """
        self.x_init = 0.0
        self.y_init = 0.0

        if self.robot is not None and hasattr(self.robot, "getField"):
            pos_field = self.robot.getField("translation")
            if pos_field is not None and hasattr(pos_field, "getSFVec3f"):
                current_z = pos_field.getSFVec3f()[2]
                pos_field.setSFVec3f([0.0, 0.0, current_z])

    def _initialization(self):
        """
        Resets the environment state variables and positions the drone at the center of the arena.
        Must be called always after __init__ and reset.
        """
        super()._initialization()
        self.dist_min_target = 0.0
        self.prev_dist_min_target = 0.0
        self.corner = None
        self.put_drone_in_the_center()

    def is_out_of_bounds(self) -> bool:
        """
        Checks whether the drone has left the geometric boundaries of the room (2x2 meters).

        Returns:
            bool: True if the drone crossed any wall, False if it remains inside the enclosure.
        """
        return (
            self.x_global < self.X_MIN
            or self.x_global > self.X_MAX
            or self.y_global < self.Y_MIN
            or self.y_global > self.Y_MAX
        )

    def get_distance_to_target(self) -> float:
        """
        Computes the horizontal plane (XY) Euclidean distance between the drone and the target.

        Returns:
            float: Distance in meters to the target.
        """
        if self.target_node is not None and hasattr(self.target_node, "getField"):
            target_field = self.target_node.getField("translation")
            if target_field is not None and hasattr(target_field, "getSFVec3f"):
                target_pos = target_field.getSFVec3f()
                dx = self.x_global - target_pos[0]
                dy = self.y_global - target_pos[1]
                return float(sqrt(dx * dx + dy * dy))

        # Fallback calculation for unit tests / CI without Webots
        dx = self.x_global - self._target_pos_fallback[0]
        dy = self.y_global - self._target_pos_fallback[1]
        return float(sqrt(dx * dx + dy * dy))

    def achieve_goal(self) -> bool:
        """
        Determines whether the drone successfully reached the target cone while airborne.
        Requires being within DISTANCE_THRESHOLD (15 cm) and at an altitude greater than 0.20 meters.

        Returns:
            bool: True if the goal was validly reached in flight.
        """
        self.dist_min_target = self.get_distance_to_target()
        return bool(self.dist_min_target <= self.DISTANCE_THRESHOLD and self.alt > 0.20)

    def is_done(self) -> bool:
        """
        Evaluates episode termination conditions:
        1. Crash to the ground (altitude < 0.10 m)
        2. Leaving the scenario boundaries
        3. Success upon reaching the target cone in safe flight
        4. Truncation due to cumulative score bounds
        5. Episode step limit exceeded

        Returns:
            bool: True if the episode has terminated or must be truncated.
        """
        # 1. Quadrotor crash / fall
        if self.alt < 0.10:
            self.terminated = True
            self.truncated = False
            self.is_success = False
            self.corner = "fall"
            if self.verbose >= 1:
                print(f"[CornerEnv] Caída detectada (altitud: {self.alt:.2f}m). Episodio terminado.")
            return True

        # 2. Leaving room boundaries
        if self.is_out_of_bounds():
            self.terminated = True
            self.truncated = False
            self.is_success = False
            self.corner = "out_of_bounds"
            if self.verbose >= 1:
                print(
                    f"[CornerEnv] Fuera de límites detectado ([x={self.x_global:.2f}, y={self.y_global:.2f}]). "
                    "Episodio terminado."
                )
            return True

        # 3. Success upon reaching the target at an appropriate altitude
        if self.achieve_goal():
            self.terminated = True
            self.truncated = False
            self.is_success = True
            self.corner = self.corner_name
            if self.verbose >= 1:
                print(
                    f"[CornerEnv] ¡Objetivo alcanzado con éxito en '{self.corner_name}'! "
                    f"(Distancia: {self.dist_min_target:.2f}m, Altitud: {self.alt:.2f}m)"
                )
            return True

        # 4. Truncation due to extreme cumulative score
        if (
            self.episode_score <= self.MIN_EPISODE_SCORE
            or self.episode_score >= self.MAX_EPISODE_SCORE
        ):
            self.terminated = True
            self.truncated = False
            self.is_success = False
            self.corner = "score_limit"
            if self.verbose >= 1:
                print(
                    f"[CornerEnv] Límite de puntaje acumulado alcanzado ({self.episode_score:.2f}). "
                    "Episodio terminado."
                )
            return True

        return False

    def get_reward(self) -> float:
        """
        Computes the reward for the current step via compute_goal_directed_reward.

        Returns:
            float: Unnormalized continuous reward.
        """
        self.dist_min_target = self.get_distance_to_target()

        # Minimum distance to obstacles in meters
        d_min = min(
            self.dist_front, self.dist_back, self.dist_right, self.dist_left
        ) / 1000.0

        # Evaluation of terminal events
        reached = self.achieve_goal()
        collided = bool(
            self.alt < 0.10
            or self.is_out_of_bounds()
            or (self.episode_score <= self.MIN_EPISODE_SCORE)
            or (self.episode_score >= self.MAX_EPISODE_SCORE)
        )

        reward = compute_goal_directed_reward(
            d_prev=self.prev_dist_min_target,
            d_now=self.dist_min_target,
            action=self.current_action,
            prev_action=self.prev_action,
            ang_vel=self.angular_velocity,
            d_min=d_min,
            collided=collided,
            reached=reached,
            progress_weight=30.0,
            obstacle_weight=0.5,
            obstacle_decay=0.10,
            action_smoothness_weight=0.05,
            angular_velocity_weight=0.01,
            time_penalty=0.01,
            reached_bonus=50.0,
            collision_penalty=50.0,
        )

        self.prev_dist_min_target = self.dist_min_target
        return reward

    def reset(self, seed: int | None = None, options: dict | None = None):
        """
        Resets the environment, repositions the drone at the center, and resumes take-off.

        Returns:
            tuple: (observation, info)
        """
        obs, info = super().reset(seed=seed, options=options)
        self.dist_min_target = self.get_distance_to_target()
        self.prev_dist_min_target = self.dist_min_target
        info = self.get_info()
        return obs, info

    def get_info(self) -> dict:
        """
        Returns the dictionary with detailed information and metrics for the current step.
        """
        info = super().get_info()
        info.update(
            {
                "x_init": self.x_init,
                "y_init": self.y_init,
                "is_success": self.is_success,
                "corner": self.corner,
                "dist_min_target": self.dist_min_target,
            }
        )
        return info
