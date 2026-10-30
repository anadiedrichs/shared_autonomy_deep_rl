# -*- coding: utf-8 -*-
"""
Goal-agnostic reward (R_general) for training the copilot residual policy.

Following the scheme of Schaff & Walter (2020), the copilot is trained to minimize its
own intervention while preventing collisions and safety violations, without knowledge of
the target cone pursued by the human or synthetic pilot.

Components:
    1. Exponential penalty for obstacle/wall proximity.
    2. Residual action magnitude penalty: -lambda * ||a_r||^2 (minimal intervention).
    3. Severe penalty for collision or fall: terminal event (-50.0).
    4. Optional survival bonus per step in safe flight.
"""

import numpy as np

from src.rewards.reward_goal_directed import obstacle_proximity_penalization


def obstacle_proximity_quadratic_penalization(
    distance_to_obstacle_m: float,
    safe_distance_m: float = 0.35,
    amplitude: float = 1.0,
) -> float:
    """
    Quadratic penalty with dead zone for obstacle/wall proximity.
    Returns 0.0 when distance >= safe_distance_m (no spurious gradients in open space).

    Args:
        distance_to_obstacle_m: Distance to the obstacle in meters.
        safe_distance_m: Threshold below which the penalty activates (default 0.35 m).
        amplitude: Multiplicative weight factor (default 1.0).

    Returns:
        float: Non-positive penalty in [-amplitude, 0.0].
    """
    d = float(distance_to_obstacle_m)
    d_safe = float(safe_distance_m)
    if d >= d_safe or d_safe <= 0.0:
        return 0.0
    pen = max(0.0, (d_safe - max(d, 0.0)) / d_safe) ** 2
    return -float(amplitude) * float(pen)


def ground_proximity_penalization(
    altitude_m: float,
    ground_threshold: float = 0.10,
    safe_altitude_m: float = 0.30,
    amplitude: float = 1.0,
    decay: float = 0.10,
    mode: str = "quadratic",
) -> float:
    """
    Penalty for proximity to the ground (crash plane at ground_threshold).
    Supports 'quadratic' (dead zone above safe_altitude_m) and 'exponential' modes.

    Args:
        altitude_m: Current altitude of the drone in meters.
        ground_threshold: Collision altitude threshold in meters (default 0.10 m).
        safe_altitude_m: Altitude threshold above which penalty is strictly 0.0 (default 0.30 m).
        amplitude: Penalty magnitude at ground contact (default 1.0).
        decay: Characteristic decay distance in meters for exponential mode (default 0.10 m).
        mode: Penalty shape ('quadratic' with dead zone or 'exponential').

    Returns:
        float: Non-positive penalty value in [-amplitude, 0.0].
    """
    alt = float(altitude_m)
    z_col = float(ground_threshold)
    if mode == "quadratic":
        z_safe = float(safe_altitude_m)
        if alt >= z_safe or z_safe <= z_col:
            return 0.0
        pen = max(0.0, (z_safe - max(alt, z_col)) / (z_safe - z_col)) ** 2
        return -float(amplitude) * float(pen)
    else:
        d_ground = max(alt - z_col, 0.0)
        return -float(amplitude) * float(np.exp(-d_ground / float(decay)))


def general_safety_reward(
    distance_to_obstacle_m: float,
    residual_action: np.ndarray | list[float],
    prev_residual_action: np.ndarray | list[float] | None = None,
    collided: bool = False,
    lambda_intervention: float = 0.50,
    action_smoothness_weight: float = 0.05,
    obstacle_weight: float = 1.0,
    obstacle_safe_distance_m: float = 0.35,
    obstacle_decay: float = 0.15,
    proximity_mode: str = "quadratic",
    collision_penalty: float = 50.0,
    alive_bonus: float = 0.0,
    altitude_m: float | None = None,
    ground_threshold: float = 0.10,
    safe_altitude_m: float = 0.30,
    altitude_weight: float = 1.0,
    altitude_decay: float = 0.10,
) -> float:
    """
    Computes the continuous goal-agnostic safety reward for the residual copilot.

    Args:
        distance_to_obstacle_m: Minimum detected distance to obstacles/walls in meters.
        residual_action: Residual action vector a_r in [-1.0, 1.0]^4.
        prev_residual_action: Residual action from the previous step for smoothness penalty.
        collided: True if the drone collided with a wall, fell to the ground, or went out of bounds.
        lambda_intervention: Intervention penalty coefficient (weight on ||a_r||^2).
        action_smoothness_weight: Weight for penalizing rapid changes in residual action.
        obstacle_weight: Amplitude of obstacle proximity penalty.
        obstacle_safe_distance_m: Distance threshold for quadratic dead zone (default 0.35 m).
        obstacle_decay: Characteristic decay distance in meters (for exponential mode).
        proximity_mode: 'quadratic' (dead zone in open space) or 'exponential'.
        collision_penalty: Penalty applied on a terminal collision event (default 50.0).
        alive_bonus: Bonus per step in safe flight without collision (default 0.0).
        altitude_m: Current altitude in meters (optional). When provided, adds ground proximity shaping.
        ground_threshold: Altitude at which crash occurs in meters (default 0.10 m).
        safe_altitude_m: Altitude above which ground penalty is strictly 0.0 (default 0.30 m).
        altitude_weight: Amplitude of ground proximity penalty (default 1.0).
        altitude_decay: Characteristic decay distance for ground proximity (default 0.10 m).

    Returns:
        float: Scalar reward for the copilot at the current step.
    """
    # 1. Penalty for proximity to walls or obstacles
    if proximity_mode == "quadratic":
        r_obstacle = obstacle_proximity_quadratic_penalization(
            distance_to_obstacle_m,
            safe_distance_m=obstacle_safe_distance_m,
            amplitude=obstacle_weight,
        )
    else:
        r_obstacle = obstacle_proximity_penalization(
            distance_to_obstacle_m, amplitude=obstacle_weight, decay=obstacle_decay
        )

    # 1b. Penalty for proximity to ground / fall risk
    if altitude_m is not None:
        r_altitude = ground_proximity_penalization(
            altitude_m=altitude_m,
            ground_threshold=ground_threshold,
            safe_altitude_m=safe_altitude_m,
            amplitude=altitude_weight,
            decay=altitude_decay,
            mode=proximity_mode,
        )
    else:
        r_altitude = 0.0

    # 2. Penalty for residual magnitude ||a_r||^2 to promote minimal intervention
    a_r = np.asarray(residual_action, dtype=np.float32)
    intervention_magnitude = float(np.dot(a_r, a_r))
    r_intervention = -float(lambda_intervention) * intervention_magnitude

    # 2b. Penalty for residual jerk / action smoothness ||a_r(t) - a_r(t-1)||^2
    if prev_residual_action is not None and action_smoothness_weight > 0.0:
        prev_a_r = np.asarray(prev_residual_action, dtype=np.float32)
        diff_act = a_r - prev_a_r
        r_smoothness = -float(action_smoothness_weight) * float(np.dot(diff_act, diff_act))
    else:
        r_smoothness = 0.0

    # 3. Base reward
    reward = r_obstacle + r_altitude + r_intervention + r_smoothness + float(alive_bonus)

    # 4. Terminal collision / fall event
    if collided:
        reward -= float(collision_penalty)

    return float(reward)


def compute_general_safety_reward(
    distance_to_obstacle_m: float | None = None,
    residual_action: np.ndarray | list[float] | None = None,
    prev_residual_action: np.ndarray | list[float] | None = None,
    collided: bool = False,
    dist_sensors_mm: list[float] | tuple[float, ...] | np.ndarray | None = None,
    lambda_intervention: float = 0.50,
    action_smoothness_weight: float = 0.05,
    obstacle_weight: float = 1.0,
    obstacle_safe_distance_m: float = 0.35,
    obstacle_decay: float = 0.15,
    proximity_mode: str = "quadratic",
    collision_penalty: float = 50.0,
    alive_bonus: float = 0.0,
    altitude_m: float | None = None,
    ground_threshold: float = 0.10,
    safe_altitude_m: float = 0.30,
    altitude_weight: float = 1.0,
    altitude_decay: float = 0.10,
) -> float:
    """
    Utility function to compute the safety reward from ToF sensor readings or direct distance.

    Args:
        distance_to_obstacle_m: Distance in meters to the nearest obstacle.
        residual_action: Residual action vector a_r.
        prev_residual_action: Residual action from the previous step.
        collided: True if a collision or fall occurred.
        dist_sensors_mm: ToF readings in millimeters [front, back, right, left] (optional).
        lambda_intervention: Intervention coefficient.
        action_smoothness_weight: Smoothness penalty weight on action jerk.
        obstacle_weight: Obstacle penalty amplitude.
        obstacle_safe_distance_m: Safe threshold for obstacle quadratic penalty (default 0.35 m).
        obstacle_decay: Obstacle spatial decay for exponential mode.
        proximity_mode: 'quadratic' (dead zone in safe flight) or 'exponential'.
        collision_penalty: Collision penalty.
        alive_bonus: Survival bonus.
        altitude_m: Current altitude in meters (optional).
        ground_threshold: Altitude threshold in meters for crash (default 0.10 m).
        safe_altitude_m: Safe altitude threshold for quadratic penalty (default 0.30 m).
        altitude_weight: Ground penalty amplitude.
        altitude_decay: Ground penalty spatial decay for exponential mode.

    Returns:
        float: Computed scalar reward.
    """
    if distance_to_obstacle_m is None:
        if dist_sensors_mm is not None and len(dist_sensors_mm) > 0:
            distance_to_obstacle_m = float(min(dist_sensors_mm)) / 1000.0
        else:
            distance_to_obstacle_m = 2.0  # No nearby obstacles by default

    act_r = residual_action if residual_action is not None else np.zeros(4, dtype=np.float32)

    return general_safety_reward(
        distance_to_obstacle_m=distance_to_obstacle_m,
        residual_action=act_r,
        prev_residual_action=prev_residual_action,
        collided=collided,
        lambda_intervention=lambda_intervention,
        action_smoothness_weight=action_smoothness_weight,
        obstacle_weight=obstacle_weight,
        obstacle_safe_distance_m=obstacle_safe_distance_m,
        obstacle_decay=obstacle_decay,
        proximity_mode=proximity_mode,
        collision_penalty=collision_penalty,
        alive_bonus=alive_bonus,
        altitude_m=altitude_m,
        ground_threshold=ground_threshold,
        safe_altitude_m=safe_altitude_m,
        altitude_weight=altitude_weight,
        altitude_decay=altitude_decay,
    )
