# -*- coding: utf-8 -*-
"""
Unit tests for the ResidualPolicy and AssistedPilot classes.
"""

import os
import sys
import numpy as np

# Ensure the project root is in PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.copilot.residual_policy import AssistedPilot, ResidualPolicy
from src.pilots.base_pilot import BasePilot


class MockBasePilot(BasePilot):
    """Simulated pilot for testing."""

    def __init__(self, action: np.ndarray):
        self.action = np.asarray(action, dtype=np.float32)
        self.reset_called = False

    def get_action(self, observation: np.ndarray, **kwargs) -> np.ndarray:
        return self.action

    def reset(self) -> None:
        self.reset_called = True


def test_residual_policy_dummy_default():
    """Validate that a ResidualPolicy without a model returns zeros by default."""
    copilot = ResidualPolicy(backend="dummy")
    obs = np.zeros(11, dtype=np.float32)
    p_act = np.array([0.5, -0.5, 0.0, 0.0], dtype=np.float32)

    a_r = copilot.get_action(obs, p_act)
    assert a_r.shape == (4,)
    assert np.allclose(a_r, 0.0)


def test_residual_policy_with_callable():
    """Validate inference with an in-memory callable function/model."""
    # Return a constant 4-dimensional vector after checking input dimension
    def custom_model(copilot_obs):
        assert copilot_obs.shape == (15,)  # 11 from obs + 4 from pilot_action
        return np.ones(4, dtype=np.float32) * 0.4

    copilot = ResidualPolicy(model=custom_model, backend="dummy", residual_scale=1.0)
    obs = np.zeros(11, dtype=np.float32)
    p_act = np.zeros(4, dtype=np.float32)

    a_r = copilot.get_action(obs, p_act)
    assert np.allclose(a_r, 0.4)


def test_residual_policy_scaling_and_clipping():
    """Validate the residual scale factor and clipping to [-1, 1]."""
    def saturated_model(copilot_obs):
        return np.array([2.0, -3.0, 0.5, -0.5], dtype=np.float32)

    # With scale 0.5
    copilot = ResidualPolicy(model=saturated_model, backend="dummy", residual_scale=0.5)
    obs = np.zeros(11, dtype=np.float32)
    p_act = np.zeros(4, dtype=np.float32)

    a_r = copilot.get_action(obs, p_act)
    # 2.0 * 0.5 = 1.0; -3.0 * 0.5 = -1.5 -> clip to -1.0; 0.5 * 0.5 = 0.25; -0.5 * 0.5 = -0.25
    assert np.isclose(a_r[0], 1.0)
    assert np.isclose(a_r[1], -1.0)
    assert np.isclose(a_r[2], 0.25)
    assert np.isclose(a_r[3], -0.25)


def test_backend_detection_from_extension():
    """Validate backend auto-detection based on checkpoint extension."""
    assert ResidualPolicy._detect_backend("models/copilot.zip") == "sb3_ppo"
    assert ResidualPolicy._detect_backend("models/copilot_sac.zip") == "sb3_sac"
    assert ResidualPolicy._detect_backend("models/copilot.h") == "rltools_sac"
    assert ResidualPolicy._detect_backend(None) == "dummy"


def test_assisted_pilot_composition():
    """Validate that AssistedPilot additively composes pilot and residual actions."""
    p_act = np.array([0.7, -0.2, 0.0, 0.3], dtype=np.float32)
    base_pilot = MockBasePilot(p_act)

    def residual_fn(copilot_obs):
        return np.array([0.5, 0.1, 0.0, -0.1], dtype=np.float32)

    copilot = ResidualPolicy(model=residual_fn, backend="dummy")
    assisted = AssistedPilot(pilot=base_pilot, copilot=copilot)

    obs = np.zeros(11, dtype=np.float32)
    a_composed = assisted.get_action(obs)

    # Composition: [0.7+0.5=1.2->1.0, -0.2+0.1=-0.1, 0.0+0.0=0.0, 0.3-0.1=0.2]
    assert np.isclose(a_composed[0], 1.0)
    assert np.isclose(a_composed[1], -0.1)
    assert np.isclose(a_composed[2], 0.0)
    assert np.isclose(a_composed[3], 0.2)

    # Validate choose_action alias
    act, states = assisted.choose_action(obs)
    assert np.allclose(act, a_composed)
    assert states == []

    # Validate reset propagation
    assisted.reset()
    assert base_pilot.reset_called is True


if __name__ == "__main__":
    test_residual_policy_dummy_default()
    test_residual_policy_with_callable()
    test_residual_policy_scaling_and_clipping()
    test_backend_detection_from_extension()
    test_assisted_pilot_composition()
    print("Todos los tests de ResidualPolicy y AssistedPilot pasaron exitosamente.")
