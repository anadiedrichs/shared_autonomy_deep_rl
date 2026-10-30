# -*- coding: utf-8 -*-
"""
Shared autonomy environment for assisted teleoperation of Crazyflie quadrotors.

Composes a pilot (human or synthetic) and a residual copilot following the formulation
of Schaff & Walter (2020):
    a = clip(a_h + a_r, -1.0, 1.0)
    a_h ~ pi_h(s)      # goal-directed
    a_r ~ pi_r(s, a_h) # goal-agnostic residual correction

Operates both in residual policy training mode (receiving a_r in step())
and in autonomous evaluation/inference mode (with an injected ResidualPolicy copilot).
"""

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    gym = None
    from src.envs.drone_robot_supervisor import spaces

from src.envs.corner_env_continuous import CornerEnvContinuous
from src.rewards.reward_general_safety import general_safety_reward


class SharedAutonomyEnv(CornerEnvContinuous):
    """
    Wraps CornerEnvContinuous to implement continuous shared autonomy.
    """

    def __init__(
        self,
        pilot,
        copilot=None,
        max_episode_steps: int = 1_000,
        corner_name: str = "cone_1",
        reward_mode: str = "safety",
        lambda_intervention: float = 0.10,
        residual_scale: float = 1.0,
        include_pilot_action_in_obs: bool = True,
        enable_trajectory_logging: bool = False,
        action_smoothness_weight: float = 0.05,
        obstacle_weight: float = 1.0,
        obstacle_safe_distance_m: float = 0.35,
        altitude_weight: float = 1.0,
        safe_altitude_m: float = 0.30,
        ground_threshold: float = 0.10,
        proximity_mode: str = "quadratic",
        verbose: int = 1,
    ):
        """
        Initializes the shared autonomy environment.

        Args:
            pilot: Pilot object with method get_action(observation, **kwargs) -> np.ndarray(4,).
            copilot: Optional copilot object with method get_action(observation, pilot_action) -> np.ndarray(4,).
            max_episode_steps: Maximum step limit per episode.
            corner_name: Target cone name in Webots.
            reward_mode: Reward function used ("safety", "goal_directed", "combined").
            lambda_intervention: Intervention penalty coefficient ||a_r||^2.
            residual_scale: Attenuation factor for the residual (in (0, 1]).
            include_pilot_action_in_obs: If True, the observation space for the copilot
                is 15-dimensional ([drone_obs(11), pilot_action(4)]).
            enable_trajectory_logging: Save physical trajectory to disk.
            action_smoothness_weight: Penalty weight for residual action variations ||a_r(t) - a_r(t-1)||^2.
            obstacle_weight: Obstacle proximity penalty weight.
            obstacle_safe_distance_m: Distance threshold below which obstacle penalty activates (default 0.35 m).
            altitude_weight: Ground proximity penalty weight for copilot safety reward.
            safe_altitude_m: Altitude threshold below which ground penalty activates (default 0.30 m).
            ground_threshold: Altitude below which crash occurs (default 0.10 m).
            proximity_mode: 'quadratic' (dead zone in safe flight) or 'exponential'.
            verbose: Console detail level.
        """
        self.pilot = pilot
        self.copilot = copilot
        self.reward_mode = str(reward_mode).lower()
        self.lambda_intervention = float(lambda_intervention)
        self.residual_scale = float(residual_scale)
        self.include_pilot_action_in_obs = bool(include_pilot_action_in_obs)
        self.action_smoothness_weight = float(action_smoothness_weight)
        self.obstacle_weight = float(obstacle_weight)
        self.obstacle_safe_distance_m = float(obstacle_safe_distance_m)
        self.altitude_weight = float(altitude_weight)
        self.safe_altitude_m = float(safe_altitude_m)
        self.ground_threshold = float(ground_threshold)
        self.proximity_mode = str(proximity_mode).lower()

        # Internal traceability variables for the current step
        self.current_drone_obs = np.zeros(11, dtype=np.float32)
        self.current_pilot_action = np.zeros(4, dtype=np.float32)
        self.current_residual_action = np.zeros(4, dtype=np.float32)
        self.prev_residual_action = np.zeros(4, dtype=np.float32)
        self.current_composed_action = np.zeros(4, dtype=np.float32)
        self.last_safety_reward = 0.0
        self.last_goal_reward = 0.0

        super().__init__(
            max_episode_steps=max_episode_steps,
            corner_name=corner_name,
            enable_trajectory_logging=enable_trajectory_logging,
            verbose=verbose,
        )

        # Redefine formal Gymnasium spaces
        obs_dim = 15 if self.include_pilot_action_in_obs else 11
        if spaces is not None:
            self.observation_space = spaces.Box(
                low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
            )
            # The action space for the RL agent learning the copilot is the residual a_r
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(4,), dtype=np.float32
            )

    def _get_pilot_action(self, drone_obs: np.ndarray) -> np.ndarray:
        """Obtains the action proposed by the pilot for the current physical observation."""
        if self.pilot is None:
            return np.zeros(4, dtype=np.float32)
        try:
            act = self.pilot.get_action(drone_obs, env=self)
        except TypeError:
            act = self.pilot.get_action(drone_obs)
        return np.clip(np.asarray(act, dtype=np.float32), -1.0, 1.0)

    def _build_copilot_observation(
        self, drone_obs: np.ndarray, pilot_action: np.ndarray
    ) -> np.ndarray:
        """Builds the extended observation for the copilot [obs, a_h]."""
        if self.include_pilot_action_in_obs:
            return np.concatenate([drone_obs, pilot_action], dtype=np.float32)
        return np.asarray(drone_obs, dtype=np.float32)

    def reset(self, seed: int | None = None, options: dict | None = None):
        """
        Resets the environment, pilot, and copilot, and returns the initial observation for the copilot.
        """
        drone_obs, info = super().reset(seed=seed, options=options)
        self.current_drone_obs = drone_obs

        # Reset internal pilot states if they implement memory
        if hasattr(self.pilot, "reset"):
            self.pilot.reset()
        if self.copilot is not None and hasattr(self.copilot, "reset"):
            self.copilot.reset()

        # Compute initial pilot action
        self.current_pilot_action = self._get_pilot_action(drone_obs)
        self.current_residual_action = np.zeros(4, dtype=np.float32)
        self.prev_residual_action = np.zeros(4, dtype=np.float32)
        self.current_composed_action = np.clip(self.current_pilot_action, -1.0, 1.0)
        self.last_safety_reward = 0.0
        self.last_goal_reward = 0.0

        copilot_obs = self._build_copilot_observation(drone_obs, self.current_pilot_action)
        info.update(self._get_shared_autonomy_info())
        return copilot_obs, info

    def step(self, action: np.ndarray | None = None):
        """
        Executes a step in the shared autonomy environment.

        Args:
            action: If provided, interpreted as the residual action a_r computed by the
                RL algorithm. If None, queries self.copilot if present, or assumes 0.

        Returns:
            tuple: (copilot_obs, reward, terminated, truncated, info)
        """
        if action is None:
            if self.copilot is not None:
                a_r = self.copilot.get_action(self.current_drone_obs, self.current_pilot_action)
            else:
                a_r = np.zeros(4, dtype=np.float32)
        else:
            a_r = np.asarray(action, dtype=np.float32)

        # Apply scale factor and clipping to the residual action
        a_r = np.clip(a_r * self.residual_scale, -1.0, 1.0)
        self.current_residual_action = a_r

        # Additive composition: a = clip(a_h + a_r, -1, 1)
        a_composed = np.clip(self.current_pilot_action + a_r, -1.0, 1.0)
        self.current_composed_action = a_composed

        # Execute the underlying physical step in Webots
        drone_obs, goal_reward, terminated, truncated, info = super().step(a_composed)
        self.current_drone_obs = drone_obs
        self.last_goal_reward = float(goal_reward)

        # Compute general safety reward (goal-agnostic)
        d_min = (
            min(self.dist_front, self.dist_back, self.dist_right, self.dist_left) / 1000.0
        )
        collided = bool(
            self.alt < 0.10
            or self.is_out_of_bounds()
            or (self.episode_score <= self.MIN_EPISODE_SCORE)
            or (self.episode_score >= self.MAX_EPISODE_SCORE)
        )
        safety_reward = general_safety_reward(
            distance_to_obstacle_m=d_min,
            residual_action=a_r,
            prev_residual_action=self.prev_residual_action,
            collided=collided,
            lambda_intervention=self.lambda_intervention,
            action_smoothness_weight=self.action_smoothness_weight,
            obstacle_weight=self.obstacle_weight,
            obstacle_safe_distance_m=self.obstacle_safe_distance_m,
            proximity_mode=self.proximity_mode,
            altitude_m=self.alt,
            ground_threshold=self.ground_threshold,
            safe_altitude_m=self.safe_altitude_m,
            altitude_weight=self.altitude_weight,
        )
        self.last_safety_reward = float(safety_reward)
        self.prev_residual_action = a_r.copy()

        # Select reward signal according to configured mode
        if self.reward_mode == "safety":
            step_reward = safety_reward
        elif self.reward_mode == "goal_directed":
            step_reward = goal_reward
        elif self.reward_mode == "combined":
            step_reward = goal_reward + safety_reward
        else:
            step_reward = safety_reward

        # Next pilot action for the following step and construction of the new observation
        next_pilot_action = self._get_pilot_action(drone_obs)
        self.current_pilot_action = next_pilot_action
        next_copilot_obs = self._build_copilot_observation(drone_obs, next_pilot_action)

        info.update(self._get_shared_autonomy_info())
        return next_copilot_obs, step_reward, terminated, truncated, info

    def _get_shared_autonomy_info(self) -> dict:
        """Returns the dictionary with shared autonomy system metrics."""
        return {
            "pilot_action": self.current_pilot_action.copy(),
            "residual_action": self.current_residual_action.copy(),
            "composed_action": self.current_composed_action.copy(),
            "residual_norm": float(np.linalg.norm(self.current_residual_action)),
            "safety_reward": float(self.last_safety_reward),
            "goal_reward": float(self.last_goal_reward),
            "altitude": float(self.alt),
        }
