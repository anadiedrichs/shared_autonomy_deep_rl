# -*- coding: utf-8 -*-
"""
Unit tests for the SharedAutonomyEnv class (Phase 4: Shared Autonomy).
"""

import os
import sys
import numpy as np

# Ensure the project root is in PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

os.environ["WEBOTS_MOCK"] = "1"

from src.copilot.residual_policy import ResidualPolicy
from src.envs.shared_autonomy_env import SharedAutonomyEnv
from src.pilots.base_pilot import BasePilot


class ConstantPilot(BasePilot):
    """Pilot that returns a constant action for testing."""

    def __init__(self, action: np.ndarray):
        self.action = np.asarray(action, dtype=np.float32)
        self.reset_count = 0

    def get_action(self, observation: np.ndarray, **kwargs) -> np.ndarray:
        return self.action

    def reset(self) -> None:
        self.reset_count += 1


def test_shared_autonomy_env_spaces():
    """Validate observation and action spaces in SharedAutonomyEnv."""
    pilot = ConstantPilot(np.array([0.5, 0.0, 0.0, 0.2], dtype=np.float32))

    # With extended observation (default: 11 drone obs + 4 pilot action = 15)
    env = SharedAutonomyEnv(pilot=pilot, include_pilot_action_in_obs=True, verbose=0)
    assert env.observation_space.shape == (15,)
    assert env.observation_space.dtype == np.float32
    assert env.action_space.shape == (4,)
    assert env.action_space.dtype == np.float32

    # With standard observation (11 drone dimensions only)
    env_simple = SharedAutonomyEnv(pilot=pilot, include_pilot_action_in_obs=False, verbose=0)
    assert env_simple.observation_space.shape == (11,)


def test_shared_autonomy_env_reset():
    """Validate that reset() initializes variables and returns the extended observation."""
    pilot = ConstantPilot(np.array([0.4, -0.3, 0.1, 0.0], dtype=np.float32))
    env = SharedAutonomyEnv(pilot=pilot, include_pilot_action_in_obs=True, verbose=0)

    obs, info = env.reset(seed=42)

    assert isinstance(obs, np.ndarray)
    assert obs.shape == (15,)
    # The last 4 elements must match the pilot's initial action
    assert np.allclose(obs[11:15], pilot.action)
    assert pilot.reset_count == 1
    assert "pilot_action" in info
    assert "residual_action" in info
    assert "composed_action" in info


def test_shared_autonomy_env_step_training_mode():
    """Validate step(action=a_r) where the RL algorithm supplies the residual action."""
    pilot_act = np.array([0.6, -0.4, 0.0, 0.2], dtype=np.float32)
    pilot = ConstantPilot(pilot_act)
    env = SharedAutonomyEnv(
        pilot=pilot,
        copilot=None,
        reward_mode="safety",
        lambda_intervention=0.1,
        residual_scale=1.0,
        verbose=0,
    )
    env.reset(seed=42)

    # The RL agent proposes a residual correction a_r
    residual_action = np.array([-0.2, 0.1, 0.0, -0.1], dtype=np.float32)
    next_obs, reward, terminated, truncated, info = env.step(residual_action)

    assert next_obs.shape == (15,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)

    # Expected composition: pilot_act + residual_action
    expected_composed = pilot_act + residual_action
    assert np.allclose(info["composed_action"], expected_composed)
    assert np.allclose(info["residual_action"], residual_action)
    assert np.isclose(info["residual_norm"], np.linalg.norm(residual_action))


def test_shared_autonomy_env_step_with_copilot_instance():
    """Validate step(action=None) when a copilot object is provided (inference/evaluation mode)."""
    pilot_act = np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32)
    pilot = ConstantPilot(pilot_act)

    def copilot_fn(copilot_obs):
        return np.array([-0.3, -0.3, 0.0, 0.0], dtype=np.float32)

    copilot = ResidualPolicy(model=copilot_fn, backend="dummy")
    env = SharedAutonomyEnv(pilot=pilot, copilot=copilot, verbose=0)
    env.reset(seed=42)

    # step without arguments
    next_obs, reward, terminated, truncated, info = env.step()

    expected_composed = np.array([0.2, 0.2, 0.0, 0.0], dtype=np.float32)
    assert np.allclose(info["composed_action"], expected_composed)


def test_shared_autonomy_env_action_saturation():
    """Validate that action composition is strictly clipped to [-1.0, 1.0]."""
    pilot_act = np.array([0.9, -0.9, 0.5, -0.5], dtype=np.float32)
    pilot = ConstantPilot(pilot_act)
    env = SharedAutonomyEnv(pilot=pilot, verbose=0)
    env.reset(seed=42)

    residual_overflow = np.array([0.5, -0.5, 0.8, -0.8], dtype=np.float32)
    _, _, _, _, info = env.step(residual_overflow)

    # [0.9+0.5=1.4->1.0, -0.9-0.5=-1.4->-1.0, 0.5+0.8=1.3->1.0, -0.5-0.8=-1.3->-1.0]
    assert np.allclose(info["composed_action"], np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32))


def test_shared_autonomy_env_reward_modes():
    """Validate the 'safety', 'goal_directed', and 'combined' reward modes."""
    pilot = ConstantPilot(np.zeros(4, dtype=np.float32))

    for mode in ("safety", "goal_directed", "combined"):
        env = SharedAutonomyEnv(pilot=pilot, reward_mode=mode, verbose=0)
        env.reset(seed=42)
        _, reward, _, _, info = env.step(np.zeros(4, dtype=np.float32))

        safety_r = info["safety_reward"]
        goal_r = info["goal_reward"]

        if mode == "safety":
            assert np.isclose(reward, safety_r)
        elif mode == "goal_directed":
            assert np.isclose(reward, goal_r)
        elif mode == "combined":
            assert np.isclose(reward, safety_r + goal_r)


if __name__ == "__main__":
    test_shared_autonomy_env_spaces()
    test_shared_autonomy_env_reset()
    test_shared_autonomy_env_step_training_mode()
    test_shared_autonomy_env_step_with_copilot_instance()
    test_shared_autonomy_env_action_saturation()
    test_shared_autonomy_env_reward_modes()
    print("Todos los tests de SharedAutonomyEnv pasaron exitosamente.")
