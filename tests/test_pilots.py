# -*- coding: utf-8 -*-
"""
Unit tests for synthetic pilots (NoisyPilot and LaggyPilot).
"""

import os
import sys
import numpy as np

# Ensure the project root is in PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    import pytest
except ImportError:
    pytest = None
from src.pilots.laggy_pilot import LaggyPilot
from src.pilots.noisy_pilot import NoisyPilot


class DummyOptimalPilot:
    """Simulated pilot for unit testing."""

    def __init__(self, fixed_action: np.ndarray | None = None):
        if fixed_action is None:
            self.fixed_action = np.array([0.5, -0.2, 0.1, 0.0], dtype=np.float32)
        else:
            self.fixed_action = np.asarray(fixed_action, dtype=np.float32)
        self.call_count = 0

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        self.call_count += 1
        # Generate an action that depends slightly on the first element of obs to distinguish steps
        return self.fixed_action + float(observation[0]) * 0.01


def test_noisy_pilot_zero_noise():
    """Validate that with sigma=0.0 NoisyPilot returns the exact action of the optimal pilot."""
    base_pilot = DummyOptimalPilot()
    noisy_pilot = NoisyPilot(optimal_pilot=base_pilot, sigma=0.0, seed=42)

    obs = np.zeros(11, dtype=np.float32)
    action = noisy_pilot.get_action(obs)

    assert np.allclose(action, base_pilot.fixed_action)
    # Test choose_action alias
    action_alias, states = noisy_pilot.choose_action(obs)
    assert np.allclose(action_alias, base_pilot.fixed_action)
    assert states == []


def test_noisy_pilot_with_noise_and_clipping():
    """Validate that noise alters the action while respecting the [-1.0, 1.0] range."""
    # Base action near the upper limit
    base_pilot = DummyOptimalPilot(fixed_action=np.array([0.98, -0.98, 0.0, 0.5], dtype=np.float32))
    noisy_pilot = NoisyPilot(optimal_pilot=base_pilot, sigma=0.5, seed=123)

    obs = np.zeros(11, dtype=np.float32)
    actions = [noisy_pilot.get_action(obs) for _ in range(50)]

    for act in actions:
        assert act.shape == (4,)
        assert np.all(act >= -1.0)
        assert np.all(act <= 1.0)

    # Verify that actions vary due to noise
    action_array = np.array(actions)
    assert np.std(action_array[:, 0]) > 0.05


def test_laggy_pilot_always_repeat():
    """Validate that with p_repeat=1.0 it always repeats the first obtained action."""
    base_pilot = DummyOptimalPilot()
    laggy_pilot = LaggyPilot(optimal_pilot=base_pilot, p_repeat=1.0, seed=42)

    obs1 = np.array([1.0] + [0.0] * 10, dtype=np.float32)
    first_action = laggy_pilot.get_action(obs1)

    # Second and third steps with different observations
    obs2 = np.array([5.0] + [0.0] * 10, dtype=np.float32)
    obs3 = np.array([10.0] + [0.0] * 10, dtype=np.float32)

    assert np.allclose(laggy_pilot.get_action(obs2), first_action)
    assert np.allclose(laggy_pilot.get_action(obs3), first_action)
    assert base_pilot.call_count == 1  # Must not query the optimal pilot again

    # Upon calling reset(), it must take a new action
    laggy_pilot.reset()
    assert laggy_pilot.last_action is None
    new_action = laggy_pilot.get_action(obs2)
    assert base_pilot.call_count == 2
    assert np.allclose(laggy_pilot.last_action, new_action)


def test_laggy_pilot_never_repeat():
    """Validate that with p_repeat=0.0 it always queries the base pilot on every step."""
    base_pilot = DummyOptimalPilot()
    laggy_pilot = LaggyPilot(optimal_pilot=base_pilot, p_repeat=0.0, seed=42)

    obs = np.zeros(11, dtype=np.float32)
    for _ in range(5):
        laggy_pilot.get_action(obs)

    assert base_pilot.call_count == 5


def test_laggy_pilot_invalid_probability():
    """Validate that it raises ValueError for invalid probabilities outside [0, 1]."""
    base_pilot = DummyOptimalPilot()
    try:
        LaggyPilot(optimal_pilot=base_pilot, p_repeat=-0.1)
        assert False, "Debería haber lanzado ValueError para p_repeat < 0"
    except ValueError:
        pass

    try:
        LaggyPilot(optimal_pilot=base_pilot, p_repeat=1.5)
        assert False, "Debería haber lanzado ValueError para p_repeat > 1"
    except ValueError:
        pass


def test_optimal_pilot_dummy():
    """Validate unified inference with dummy backend and custom functions."""
    from src.pilots.optimal_pilot import OptimalPilot

    # Default dummy pilot (neutral hover: [0, 0, 0, 0])
    pilot_default = OptimalPilot(backend="dummy")
    obs = np.zeros(11, dtype=np.float32)
    act = pilot_default.get_action(obs)
    assert np.allclose(act, np.zeros(4, dtype=np.float32))

    # Dummy pilot with custom function
    custom_func = lambda o: np.array([0.2, -0.3, 0.1, 0.0], dtype=np.float32)
    pilot_custom = OptimalPilot(backend="dummy", model_instance=custom_func)
    act_custom = pilot_custom.get_action(obs)
    assert np.allclose(act_custom, np.array([0.2, -0.3, 0.1, 0.0], dtype=np.float32))

    # Test choose_action alias
    act_alias, states = pilot_custom.choose_action(obs)
    assert np.allclose(act_alias, act_custom)
    assert states == []


def test_optimal_pilot_unsupported_backend():
    """Validate that it raises ValueError for unrecognized backends."""
    from src.pilots.optimal_pilot import OptimalPilot

    try:
        OptimalPilot(backend="backend_inexistente")
        assert False, "Debería haber lanzado ValueError ante backend no soportado"
    except ValueError:
        pass


def test_optimal_pilot_relative_path_resolution():
    """Validate that OptimalPilot resolves relative paths by searching in the project root."""
    from src.pilots.optimal_pilot import OptimalPilot

    # File that does not exist anywhere must raise FileNotFoundError
    try:
        OptimalPilot(backend="sb3_ppo", checkpoint_path="archivo_inexistente.zip")
        assert False, "Debería haber lanzado FileNotFoundError"
    except FileNotFoundError:
        pass


def test_base_pilot_inheritance_and_interface():
    """Validate that all pilots inherit from BasePilot and fulfill its common contract."""
    from src.pilots.base_pilot import BasePilot, Pilot
    from src.pilots.heuristic_pilot import HeuristicPilot
    from src.pilots.laggy_pilot import LaggyPilot
    from src.pilots.noisy_pilot import NoisyPilot
    from src.pilots.optimal_pilot import OptimalPilot

    assert Pilot is BasePilot

    # 1. Cannot instantiate BasePilot directly without implementing get_action
    try:
        class IncompletePilot(BasePilot):
            pass
        IncompletePilot()
        assert False, "Debería fallar instanciación de clase abstracta"
    except TypeError:
        pass

    # 2. Pilot instances inherit from BasePilot
    heuristic = HeuristicPilot(mode="open_loop")
    optimal = OptimalPilot(backend="dummy")
    noisy = NoisyPilot(optimal_pilot=heuristic, sigma=0.0)
    laggy = LaggyPilot(optimal_pilot=heuristic, p_repeat=0.0)

    for p in (heuristic, optimal, noisy, laggy):
        assert isinstance(p, BasePilot)
        assert isinstance(p, Pilot)

        obs = np.zeros(11, dtype=np.float32)
        act = p.get_action(obs)
        assert act.shape == (4,)

        # Validate direct __call__ invocation
        act_call = p(obs)
        assert np.allclose(act, act_call)

        # Validate choose_action contract
        act_choose, aux = p.choose_action(obs)
        assert np.allclose(act, act_choose)
        assert aux == []

        # Validate reset
        p.reset()


if __name__ == "__main__":
    test_base_pilot_inheritance_and_interface()
    test_noisy_pilot_zero_noise()
    test_noisy_pilot_with_noise_and_clipping()
    test_laggy_pilot_always_repeat()
    test_laggy_pilot_never_repeat()
    test_laggy_pilot_invalid_probability()
    test_optimal_pilot_dummy()
    test_optimal_pilot_unsupported_backend()
    test_optimal_pilot_relative_path_resolution()
    print("Todos los tests de pilotos pasaron exitosamente.")
