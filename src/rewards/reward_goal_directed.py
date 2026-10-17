# -*- coding: utf-8 -*-
"""
Goal-directed reward for the Crazyflie under continuous control.

Implements the calibrated multi-component reward shaping formulation:
    - Differential progress: w_prog * (d_prev - d_now)
    - Obstacle proximity: -w_obs * exp(-d_min / decay_obs)
    - Action smoothness: -w_act * ||action - prev_action||^2
    - Angular stability: -w_omega * ||ang_vel||^2
    - Step cost (anti-hovering): -w_time

Terminal events:
    - +50.0 upon reaching the target at a safe altitude
    - -50.0 upon colliding or falling to the ground
"""

import numpy as np


def obstacle_proximity_penalization(
    distance_to_obstacle_m: float,
    amplitude: float = 0.5,
    decay: float = 0.10,
) -> float:
    """
    Exponential penalty for proximity to walls or obstacles.

    Args:
        distance_to_obstacle_m: Minimum detected distance to obstacles in meters.
        amplitude: Penalty magnitude at contact (default 0.5).
        decay: Characteristic decay distance in meters (default 0.10 m = 10 cm).

    Returns:
        float: Non-positive penalty value in [-amplitude, 0.0].
    """
    d = max(float(distance_to_obstacle_m), 0.0)
    return -float(amplitude) * float(np.exp(-d / float(decay)))


def walls_proximity_penalization(distance_to_obstacle_m: float) -> float:
    """
    Equation 3.1 from the thesis (kept for backwards compatibility): linear proximity penalty.
    """
    return 50.0 * float(distance_to_obstacle_m) - 10.0


def penalization_distance_to_target(distance_to_target_m: float) -> float:
    """
    Equation 3.2 from the thesis (kept for backwards compatibility): 1 / x.
    """
    dist = max(float(distance_to_target_m), 0.05)
    return 1.0 / dist


def compute_goal_directed_reward(
    d_prev: float | None = None,
    d_now: float | None = None,
    action: np.ndarray | list[float] | None = None,
    prev_action: np.ndarray | list[float] | None = None,
    ang_vel: np.ndarray | list[float] | None = None,
    d_min: float = 1.0,
    collided: bool = False,
    reached: bool = False,
    state_pos: np.ndarray | list[float] | None = None,
    prev_state_pos: np.ndarray | list[float] | None = None,
    goal_pos: np.ndarray | list[float] | None = None,
    progress_weight: float = 30.0,
    obstacle_weight: float = 0.5,
    obstacle_decay: float = 0.10,
    action_smoothness_weight: float = 0.05,
    angular_velocity_weight: float = 0.01,
    time_penalty: float = 0.01,
    reached_bonus: float = 50.0,
    collision_penalty: float = 50.0,
) -> float:
    """
    Computes the unnormalized continuous reward for the Crazyflie.

    Args:
        d_prev: Previous Euclidean distance to the goal (in meters).
        d_now: Current Euclidean distance to the goal (in meters).
        action: Current 4-component continuous action vector in [-1.0, 1.0].
        prev_action: Continuous action vector from the previous step.
        ang_vel: Angular velocity in rad/s (3-component gyroscope vector).
        d_min: Minimum detected distance to obstacles in meters.
        collided: True if the drone collided with a wall, fell to the ground, or went out of bounds.
        reached: True if the drone reached the target cone at a safe altitude.
        state_pos: Current position of the drone (optional if d_now is provided).
        prev_state_pos: Previous position of the drone (optional if d_prev is provided).
        goal_pos: Target cone position (optional if d_prev and d_now are provided).
        progress_weight: Multiplicative factor for differential progress (default 30.0).
        obstacle_weight: Obstacle penalty amplitude factor (default 0.5).
        obstacle_decay: Decay radius in meters for obstacles (default 0.10 m).
        action_smoothness_weight: Weight for penalizing abrupt action changes (default 0.05).
        angular_velocity_weight: Weight for penalizing high angular velocities (default 0.01).
        time_penalty: Fixed cost per step to prevent inactive hovering (default 0.01).
        reached_bonus: Bonus for reaching the target (default +50.0).
        collision_penalty: Penalty for collision / fall (default -50.0).

    Returns:
        float: Scalar reward for the step.
    """
    # If vector positions are passed instead of scalar distances
    if d_prev is None and prev_state_pos is not None and goal_pos is not None:
        p_prev = np.asarray(prev_state_pos, dtype=np.float32)
        g = np.asarray(goal_pos, dtype=np.float32)
        d_prev = float(np.linalg.norm(p_prev - g))

    if d_now is None and state_pos is not None and goal_pos is not None:
        p_now = np.asarray(state_pos, dtype=np.float32)
        g = np.asarray(goal_pos, dtype=np.float32)
        d_now = float(np.linalg.norm(p_now - g))

    d_prev_val = float(d_prev) if d_prev is not None else 0.0
    d_now_val = float(d_now) if d_now is not None else 0.0

    act = np.asarray(action, dtype=np.float32) if action is not None else np.zeros(4, dtype=np.float32)
    prev_act = (
        np.asarray(prev_action, dtype=np.float32)
        if prev_action is not None
        else np.zeros(4, dtype=np.float32)
    )
    omega = np.asarray(ang_vel, dtype=np.float32) if ang_vel is not None else np.zeros(3, dtype=np.float32)

    # 1. Differential progress toward the goal
    r = float(progress_weight) * (d_prev_val - d_now_val)

    # 2. Obstacle proximity (smooth exponential)
    r += obstacle_proximity_penalization(d_min, amplitude=obstacle_weight, decay=obstacle_decay)

    # 3. Action smoothness (penalizes abrupt changes between consecutive steps)
    act_diff = act - prev_act
    r -= float(action_smoothness_weight) * float(np.dot(act_diff, act_diff))

    # 4. Angular velocity (penalizes sharp turns and encourages stability)
    r -= float(angular_velocity_weight) * float(np.dot(omega, omega))

    # 5. Per-step time cost (eliminates reward farming through inaction)
    r -= float(time_penalty)

    # 6. Terminal events
    if reached:
        r += float(reached_bonus)
    if collided:
        r -= float(collision_penalty)

    return float(r)
