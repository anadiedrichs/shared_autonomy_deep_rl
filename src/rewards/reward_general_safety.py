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


def general_safety_reward(
    distance_to_obstacle_m: float,
    residual_action: np.ndarray | list[float],
    collided: bool = False,
    lambda_intervention: float = 0.10,
    obstacle_weight: float = 1.0,
    obstacle_decay: float = 0.15,
    collision_penalty: float = 50.0,
    alive_bonus: float = 0.0,
) -> float:
    """
    Computes the continuous goal-agnostic safety reward for the residual copilot.

    Args:
        distance_to_obstacle_m: Minimum detected distance to obstacles/walls in meters.
        residual_action: Residual action vector a_r in [-1.0, 1.0]^4.
        collided: True if the drone collided with a wall, fell to the ground, or went out of bounds.
        lambda_intervention: Intervention penalty coefficient (weight on ||a_r||^2).
        obstacle_weight: Amplitude of the exponential obstacle proximity penalty.
        obstacle_decay: Characteristic decay distance in meters (default 0.15 m).
        collision_penalty: Penalty applied on a terminal collision event (default 50.0).
        alive_bonus: Bonus per step in safe flight without collision (default 0.0).

    Returns:
        float: Scalar reward for the copilot at the current step.
    """
    # 1. Penalty for proximity to walls or obstacles (smooth proximity)
    r_obstacle = obstacle_proximity_penalization(
        distance_to_obstacle_m, amplitude=obstacle_weight, decay=obstacle_decay
    )

    # 2. Penalty for residual magnitude ||a_r||^2 to promote minimal intervention
    a_r = np.asarray(residual_action, dtype=np.float32)
    intervention_magnitude = float(np.dot(a_r, a_r))
    r_intervention = -float(lambda_intervention) * intervention_magnitude

    # 3. Base reward
    reward = r_obstacle + r_intervention + float(alive_bonus)

    # 4. Terminal collision / fall event
    if collided:
        reward -= float(collision_penalty)

    return float(reward)


def compute_general_safety_reward(
    distance_to_obstacle_m: float | None = None,
    residual_action: np.ndarray | list[float] | None = None,
    collided: bool = False,
    dist_sensors_mm: list[float] | tuple[float, ...] | np.ndarray | None = None,
    lambda_intervention: float = 0.10,
    obstacle_weight: float = 1.0,
    obstacle_decay: float = 0.15,
    collision_penalty: float = 50.0,
    alive_bonus: float = 0.0,
) -> float:
    """
    Utility function to compute the safety reward from ToF sensor readings or direct distance.

    Args:
        distance_to_obstacle_m: Distance in meters to the nearest obstacle.
        residual_action: Residual action vector a_r.
        collided: True if a collision or fall occurred.
        dist_sensors_mm: ToF readings in millimeters [front, back, right, left] (optional).
        lambda_intervention: Intervention coefficient.
        obstacle_weight: Obstacle penalty amplitude.
        obstacle_decay: Obstacle spatial decay.
        collision_penalty: Collision penalty.
        alive_bonus: Survival bonus.

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
        collided=collided,
        lambda_intervention=lambda_intervention,
        obstacle_weight=obstacle_weight,
        obstacle_decay=obstacle_decay,
        collision_penalty=collision_penalty,
        alive_bonus=alive_bonus,
    )
