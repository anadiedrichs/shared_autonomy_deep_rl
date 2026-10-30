# -*- coding: utf-8 -*-
"""
Unit tests for the general_safety_reward and compute_general_safety_reward functions.
"""

import os
import sys
import numpy as np

# Ensure the project root is in PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.rewards.reward_general_safety import (
    compute_general_safety_reward,
    general_safety_reward,
)


def test_zero_intervention_far_from_obstacles():
    """Validate that without residual intervention and far from obstacles the reward is close to zero."""
    residual = np.zeros(4, dtype=np.float32)
    # Far from obstacles (2.0 meters)
    r = general_safety_reward(
        distance_to_obstacle_m=2.0,
        residual_action=residual,
        collided=False,
        lambda_intervention=0.1,
    )
    # Exponential proximity at 2m with decay 0.15 is negligible (< 1e-5)
    assert np.isclose(r, 0.0, atol=1e-3)


def test_intervention_penalty_scaling():
    """Validate that the residual magnitude penalty scales as -lambda * ||a_r||^2."""
    residual = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)  # ||a_r||^2 = 1.0
    lambda_val = 0.20
    r = general_safety_reward(
        distance_to_obstacle_m=2.0,
        residual_action=residual,
        collided=False,
        lambda_intervention=lambda_val,
    )
    expected_intervention = -lambda_val * 1.0
    assert np.isclose(r, expected_intervention, atol=1e-3)

    # With maximum magnitude action ||[1, 1, 1, 1]||^2 = 4.0
    residual_max = np.ones(4, dtype=np.float32)
    r_max = general_safety_reward(
        distance_to_obstacle_m=2.0,
        residual_action=residual_max,
        collided=False,
        lambda_intervention=lambda_val,
    )
    assert np.isclose(r_max, -lambda_val * 4.0, atol=1e-3)


def test_obstacle_proximity_penalty():
    """Validate that proximity to obstacles penalizes appropriately according to distance in exponential mode."""
    residual = np.zeros(4, dtype=np.float32)
    # Right in contact with the obstacle (d = 0.0 m)
    r_contact = general_safety_reward(
        distance_to_obstacle_m=0.0,
        residual_action=residual,
        collided=False,
        obstacle_weight=1.5,
        obstacle_decay=0.15,
        proximity_mode="exponential",
    )
    # exp(0) = 1 -> -1.5
    assert np.isclose(r_contact, -1.5, atol=1e-3)

    # At 0.15m (1 tau distance): -1.5 * exp(-1) = -1.5 * 0.367879 = -0.5518
    r_tau = general_safety_reward(
        distance_to_obstacle_m=0.15,
        residual_action=residual,
        collided=False,
        obstacle_weight=1.5,
        obstacle_decay=0.15,
        proximity_mode="exponential",
    )
    assert np.isclose(r_tau, -1.5 * np.exp(-1.0), atol=1e-3)


def test_collision_terminal_penalty():
    """Validate that a terminal collision event applies the collision_penalty."""
    residual = np.zeros(4, dtype=np.float32)
    collision_pen = 50.0
    r_normal = general_safety_reward(
        distance_to_obstacle_m=0.05,
        residual_action=residual,
        collided=False,
        collision_penalty=collision_pen,
    )
    r_collided = general_safety_reward(
        distance_to_obstacle_m=0.05,
        residual_action=residual,
        collided=True,
        collision_penalty=collision_pen,
    )
    assert np.isclose(r_collided, r_normal - collision_pen, atol=1e-3)


def test_compute_general_safety_reward_with_sensors():
    """Validate the helper function with ToF sensor readings in millimeters."""
    # Sensors: front=1000mm, back=150mm (closest), right=500mm, left=800mm
    sensors_mm = [1000.0, 150.0, 500.0, 800.0]
    residual = np.array([0.1, 0.0, 0.0, 0.0], dtype=np.float32)

    r_helper = compute_general_safety_reward(
        dist_sensors_mm=sensors_mm,
        residual_action=residual,
        collided=False,
        lambda_intervention=0.1,
        obstacle_weight=1.0,
        obstacle_decay=0.15,
    )

    r_direct = general_safety_reward(
        distance_to_obstacle_m=0.15,
        residual_action=residual,
        collided=False,
        lambda_intervention=0.1,
        obstacle_weight=1.0,
        obstacle_decay=0.15,
    )

    assert np.isclose(r_helper, r_direct, atol=1e-4)


def test_altitude_proximity_penalty():
    """Validate that proximity to the ground penalizes appropriately according to altitude."""
    residual = np.zeros(4, dtype=np.float32)

    # 1. At ground collision threshold (alt = 0.10 m) with exponential mode
    r_ground = general_safety_reward(
        distance_to_obstacle_m=2.0,
        residual_action=residual,
        collided=False,
        altitude_m=0.10,
        ground_threshold=0.10,
        altitude_weight=1.5,
        altitude_decay=0.10,
        proximity_mode="exponential",
    )
    # exp(0) = 1 -> -1.5 (obstacle proximity at 2m is negligible < 1e-5)
    assert np.isclose(r_ground, -1.5, atol=1e-3)

    # 2. At 0.20m altitude (d_ground = 0.10m = 1 tau): -1.5 * exp(-1.0)
    r_tau = general_safety_reward(
        distance_to_obstacle_m=2.0,
        residual_action=residual,
        collided=False,
        altitude_m=0.20,
        ground_threshold=0.10,
        altitude_weight=1.5,
        altitude_decay=0.10,
        proximity_mode="exponential",
    )
    assert np.isclose(r_tau, -1.5 * np.exp(-1.0), atol=1e-3)

    # 3. At safe flight altitude (alt = 0.60 m): negligible penalty
    r_safe = general_safety_reward(
        distance_to_obstacle_m=2.0,
        residual_action=residual,
        collided=False,
        altitude_m=0.60,
        ground_threshold=0.10,
        altitude_weight=1.5,
        altitude_decay=0.10,
        proximity_mode="exponential",
    )
    assert abs(r_safe) < 0.02

    # 4. Backward compatibility: when altitude_m is None, r_altitude = 0.0
    r_none = general_safety_reward(
        distance_to_obstacle_m=2.0,
        residual_action=residual,
        collided=False,
        altitude_m=None,
    )
    assert np.isclose(r_none, 0.0, atol=1e-3)


def test_quadratic_dead_zone_penalties():
    """Validate quadratic penalty with strict 0.0 dead zone in safe flight."""
    residual = np.zeros(4, dtype=np.float32)

    # Safe space: obstacle at 1.0 m (> 0.35 m safe threshold), altitude 0.6 m (> 0.30 m)
    r_safe = general_safety_reward(
        distance_to_obstacle_m=1.0,
        residual_action=residual,
        collided=False,
        altitude_m=0.60,
        proximity_mode="quadratic",
        obstacle_safe_distance_m=0.35,
        safe_altitude_m=0.30,
    )
    # Must be EXACTLY zero in safe air
    assert np.isclose(r_safe, 0.0, atol=1e-6)

    # Obstacle inside dead zone: d = 0.175 m (halfway to 0.35 m) -> ((0.35 - 0.175) / 0.35)^2 = 0.5^2 = 0.25
    r_obs = general_safety_reward(
        distance_to_obstacle_m=0.175,
        residual_action=residual,
        collided=False,
        altitude_m=0.60,
        proximity_mode="quadratic",
        obstacle_safe_distance_m=0.35,
        obstacle_weight=1.0,
        safe_altitude_m=0.30,
    )
    assert np.isclose(r_obs, -0.25, atol=1e-3)

    # Altitude inside dead zone: alt = 0.20 m (halfway between 0.10 and 0.30) -> 0.5^2 = 0.25
    r_alt = general_safety_reward(
        distance_to_obstacle_m=1.0,
        residual_action=residual,
        collided=False,
        altitude_m=0.20,
        ground_threshold=0.10,
        safe_altitude_m=0.30,
        altitude_weight=1.0,
        proximity_mode="quadratic",
        obstacle_safe_distance_m=0.35,
    )
    assert np.isclose(r_alt, -0.25, atol=1e-3)


def test_action_smoothness_jerk_penalty():
    """Validate penalty on residual action variation ||a_r(t) - a_r(t-1)||^2."""
    a_r_curr = np.array([0.2, -0.1, 0.0, 0.0], dtype=np.float32)
    a_r_prev = np.array([0.0, 0.1, 0.0, 0.0], dtype=np.float32)
    # diff = [0.2, -0.2, 0.0, 0.0], diff^2 = 0.04 + 0.04 = 0.08
    # smoothness_weight = 0.10 -> penalty = -0.008
    w_smooth = 0.10
    r = general_safety_reward(
        distance_to_obstacle_m=2.0,
        residual_action=a_r_curr,
        prev_residual_action=a_r_prev,
        collided=False,
        lambda_intervention=0.0,
        action_smoothness_weight=w_smooth,
        altitude_m=0.60,
        proximity_mode="quadratic",
    )
    assert np.isclose(r, -w_smooth * 0.08, atol=1e-4)


def test_compute_general_safety_reward_with_altitude():
    """Validate compute_general_safety_reward forwards altitude properly."""
    residual = np.zeros(4, dtype=np.float32)
    r_helper = compute_general_safety_reward(
        distance_to_obstacle_m=2.0,
        residual_action=residual,
        collided=False,
        altitude_m=0.15,
        ground_threshold=0.10,
        safe_altitude_m=0.30,
        altitude_weight=1.0,
        proximity_mode="quadratic",
    )
    r_direct = general_safety_reward(
        distance_to_obstacle_m=2.0,
        residual_action=residual,
        collided=False,
        altitude_m=0.15,
        ground_threshold=0.10,
        safe_altitude_m=0.30,
        altitude_weight=1.0,
        proximity_mode="quadratic",
    )
    assert np.isclose(r_helper, r_direct, atol=1e-4)


if __name__ == "__main__":
    test_zero_intervention_far_from_obstacles()
    test_intervention_penalty_scaling()
    test_obstacle_proximity_penalty()
    test_collision_terminal_penalty()
    test_compute_general_safety_reward_with_sensors()
    test_altitude_proximity_penalty()
    test_compute_general_safety_reward_with_altitude()
    test_quadratic_dead_zone_penalties()
    test_action_smoothness_jerk_penalty()
    print("Todos los tests de recompensa de seguridad general pasaron exitosamente.")
