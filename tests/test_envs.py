# -*- coding: utf-8 -*-
"""
Unit tests for DroneRobotSupervisor and environment utilities.
"""

import os
import sys

# Ensure the project root is in PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

os.environ["WEBOTS_MOCK"] = "1"
import numpy as np
try:
    import pytest
except ImportError:
    pytest = None
from src.envs.drone_robot_supervisor import DroneRobotSupervisor
from src.utils.utilities import normalize_to_range


def test_normalize_to_range():
    """Validate the linear normalization function with and without clipping."""
    # Simple mapping from [0, 10] to [-1, 1]
    assert np.isclose(normalize_to_range(5.0, 0.0, 10.0, -1.0, 1.0), 0.0)
    assert np.isclose(normalize_to_range(0.0, 0.0, 10.0, -1.0, 1.0), -1.0)
    assert np.isclose(normalize_to_range(10.0, 0.0, 10.0, -1.0, 1.0), 1.0)

    # Clipping with clip=True
    assert np.isclose(normalize_to_range(15.0, 0.0, 10.0, -1.0, 1.0, clip=True), 1.0)
    assert np.isclose(normalize_to_range(-5.0, 0.0, 10.0, -1.0, 1.0, clip=True), -1.0)


def test_drone_robot_supervisor_spaces():
    """Validate that action and observation spaces have the expected dimensions and types."""
    env = DroneRobotSupervisor()

    # Continuous action space: 4 dimensions in [-1, 1]
    assert env.action_space.shape == (4,)
    assert env.action_space.dtype == np.float32
    assert np.all(env.action_space.low == -1.0)
    assert np.all(env.action_space.high == 1.0)

    # Continuous observation space: 11 dimensions in [-1, 1]
    assert env.observation_space.shape == (11,)
    assert env.observation_space.dtype == np.float32
    assert np.all(env.observation_space.low == -1.0)
    assert np.all(env.observation_space.high == 1.0)


def test_denormalize_action():
    """Validate physical conversion of the continuous action vector."""
    env = DroneRobotSupervisor()
    env.dt = 0.032
    env.height_desired = 0.30

    # Zero action: zero velocities and unchanged altitude
    action_zero = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    vx, vy, yaw_rate, h = env._denormalize_action(action_zero)
    assert np.isclose(vx, 0.0)
    assert np.isclose(vy, 0.0)
    assert np.isclose(yaw_rate, 0.0)
    assert np.isclose(h, 0.30)

    # Maximum action: configured maximum limits
    action_max = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    vx, vy, yaw_rate, h = env._denormalize_action(action_max)
    assert np.isclose(vx, env.MAX_FORWARD_VELOCITY)
    assert np.isclose(vy, env.MAX_LATERAL_VELOCITY)
    assert np.isclose(yaw_rate, env.MAX_YAW_RATE)
    assert h > 0.30  # Altitude must have increased


def test_drone_robot_supervisor_reset_and_step():
    """Validate that reset() and step() work properly in decoupled / mock mode."""
    env = DroneRobotSupervisor()

    # Reset
    obs, info = env.reset(seed=42)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (11,)
    assert isinstance(info, dict)
    assert "is_success" in info

    # Step with valid action within bounds
    action = np.array([0.5, -0.5, 0.1, 0.0], dtype=np.float32)
    next_obs, reward, terminated, truncated, info = env.step(action)

    assert next_obs.shape == (11,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    assert env.episode_step == 1


def test_reward_goal_directed():
    """Validate the goal-directed reward formulas."""
    from src.rewards.reward_goal_directed import (
        walls_proximity_penalization,
        penalization_distance_to_target,
        obstacle_proximity_penalization,
        compute_goal_directed_reward,
    )

    # Legacy Equation 3.1: 50x - 10
    assert np.isclose(walls_proximity_penalization(0.10), -5.0)
    assert np.isclose(walls_proximity_penalization(0.0), -10.0)

    # Legacy Equation 3.2: 1 / x
    assert np.isclose(penalization_distance_to_target(1.0), 1.0)
    assert np.isclose(penalization_distance_to_target(0.5), 2.0)

    # Exponential obstacle penalty (-0.5 * exp(-d / 0.10))
    assert np.isclose(obstacle_proximity_penalization(0.0, amplitude=0.5, decay=0.10), -0.5)
    assert np.isclose(
        obstacle_proximity_penalization(0.10, amplitude=0.5, decay=0.10), -0.5 * np.exp(-1.0)
    )

    # Goal reached (+50.0) with positive progress
    r_goal = compute_goal_directed_reward(
        d_prev=0.2,
        d_now=0.1,
        d_min=1.0,
        reached=True,
        collided=False,
    )
    # Progress: 30*(0.2-0.1)=3.0, Obstacle (d=1.0) ~0.0, Time: -0.01, Reached: +50.0 => ~52.99
    assert r_goal >= 50.0

    # Failure / collision (-50.0)
    r_fail = compute_goal_directed_reward(
        d_prev=1.0,
        d_now=1.0,
        d_min=0.0,
        reached=False,
        collided=True,
    )
    # Collision: -50.0 - 0.5 (contact) - 0.01 (time) => -50.51
    assert r_fail <= -50.0

    # Static hovering in the center (time cost without progress)
    r_hover = compute_goal_directed_reward(
        d_prev=1.0,
        d_now=1.0,
        d_min=1.0,
    )
    assert np.isclose(r_hover, -0.01, atol=1e-3)

    # Penalty for abrupt action change (smoothness)
    r_smooth = compute_goal_directed_reward(
        d_prev=1.0,
        d_now=1.0,
        action=np.array([1.0, 0.0, 0.0, 0.0]),
        prev_action=np.array([0.0, 0.0, 0.0, 0.0]),
        d_min=1.0,
    )
    # -0.01 (time) - 0.05 * 1.0 (smoothness) = -0.06
    assert np.isclose(r_smooth, -0.06, atol=1e-3)


def test_corner_env_continuous():
    """Validate the CornerEnvContinuous environment and its logical conditions."""
    from src.envs.corner_env_continuous import CornerEnvContinuous

    env = CornerEnvContinuous(corner_name="cone_1")

    # Reset
    obs, info = env.reset(seed=10)
    assert obs.shape == (11,)
    assert "corner" in info
    assert "dist_min_target" in info

    # Step
    action = np.array([0.1, 0.1, 0.0, 0.0], dtype=np.float32)
    next_obs, reward, terminated, truncated, info = env.step(action)
    assert next_obs.shape == (11,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)

    # Geometric bounds test
    env.x_global = 1.5
    assert env.is_out_of_bounds() is True
    env.x_global = 0.0
    assert env.is_out_of_bounds() is False

    # Ground crash test (alt < 0.10)
    env.alt = 0.05
    assert env.is_done() is True
    assert env.corner == "fall"

    # Goal reached test in safe flight (close to target at [0.8, 0.8])
    env.alt = 0.55
    env.x_global = 0.80
    env.y_global = 0.80
    assert env.achieve_goal() is True

    # Altitude threshold alt > 0.20
    env.alt = 0.25
    assert env.achieve_goal() is True
    env.alt = 0.15
    assert env.achieve_goal() is False


def test_verbose_mode():
    """Validate verbose mode configuration in environments."""
    from src.envs.corner_env_continuous import CornerEnvContinuous

    env_silent = CornerEnvContinuous(verbose=0)
    assert env_silent.verbose == 0

    env_verbose = CornerEnvContinuous(verbose=1)
    assert env_verbose.verbose == 1


if __name__ == "__main__":
    test_normalize_to_range()
    test_drone_robot_supervisor_spaces()
    test_denormalize_action()
    test_drone_robot_supervisor_reset_and_step()
    test_reward_goal_directed()
    test_corner_env_continuous()
    test_verbose_mode()
    print("Todos los tests pasaron exitosamente.")
