# -*- coding: utf-8 -*-
"""
Tests unitarios para los pilotos sintéticos (NoisyPilot y LaggyPilot).
"""

import numpy as np
try:
    import pytest
except ImportError:
    pytest = None
from src.pilots.laggy_pilot import LaggyPilot
from src.pilots.noisy_pilot import NoisyPilot


class DummyOptimalPilot:
    """Piloto simulado para pruebas unitarias."""

    def __init__(self, fixed_action: np.ndarray | None = None):
        if fixed_action is None:
            self.fixed_action = np.array([0.5, -0.2, 0.1, 0.0], dtype=np.float32)
        else:
            self.fixed_action = np.asarray(fixed_action, dtype=np.float32)
        self.call_count = 0

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        self.call_count += 1
        # Genera una acción que depende ligeramente del primer elemento de obs para distinguir pasos
        return self.fixed_action + float(observation[0]) * 0.01


def test_noisy_pilot_zero_noise():
    """Valida que con sigma=0.0 el NoisyPilot devuelva la acción exacta del piloto óptimo."""
    base_pilot = DummyOptimalPilot()
    noisy_pilot = NoisyPilot(optimal_pilot=base_pilot, sigma=0.0, seed=42)

    obs = np.zeros(11, dtype=np.float32)
    action = noisy_pilot.get_action(obs)

    assert np.allclose(action, base_pilot.fixed_action)
    # Probar alias choose_action
    action_alias, states = noisy_pilot.choose_action(obs)
    assert np.allclose(action_alias, base_pilot.fixed_action)
    assert states == []


def test_noisy_pilot_with_noise_and_clipping():
    """Valida que el ruido altere la acción pero respete el rango [-1.0, 1.0]."""
    # Acción base en el límite superior
    base_pilot = DummyOptimalPilot(fixed_action=np.array([0.98, -0.98, 0.0, 0.5], dtype=np.float32))
    noisy_pilot = NoisyPilot(optimal_pilot=base_pilot, sigma=0.5, seed=123)

    obs = np.zeros(11, dtype=np.float32)
    actions = [noisy_pilot.get_action(obs) for _ in range(50)]

    for act in actions:
        assert act.shape == (4,)
        assert np.all(act >= -1.0)
        assert np.all(act <= 1.0)

    # Verificar que las acciones varían por el ruido
    action_array = np.array(actions)
    assert np.std(action_array[:, 0]) > 0.05


def test_laggy_pilot_always_repeat():
    """Valida que con p_repeat=1.0 repita siempre la primera acción obtenida."""
    base_pilot = DummyOptimalPilot()
    laggy_pilot = LaggyPilot(optimal_pilot=base_pilot, p_repeat=1.0, seed=42)

    obs1 = np.array([1.0] + [0.0] * 10, dtype=np.float32)
    first_action = laggy_pilot.get_action(obs1)

    # Segundo y tercer paso con distintas observaciones
    obs2 = np.array([5.0] + [0.0] * 10, dtype=np.float32)
    obs3 = np.array([10.0] + [0.0] * 10, dtype=np.float32)

    assert np.allclose(laggy_pilot.get_action(obs2), first_action)
    assert np.allclose(laggy_pilot.get_action(obs3), first_action)
    assert base_pilot.call_count == 1  # No debe consultar al piloto óptimo de nuevo

    # Al llamar a reset(), debe tomar una nueva acción
    laggy_pilot.reset()
    assert laggy_pilot.last_action is None
    new_action = laggy_pilot.get_action(obs2)
    assert base_pilot.call_count == 2
    assert np.allclose(laggy_pilot.last_action, new_action)


def test_laggy_pilot_never_repeat():
    """Valida que con p_repeat=0.0 consulte siempre al piloto base en cada paso."""
    base_pilot = DummyOptimalPilot()
    laggy_pilot = LaggyPilot(optimal_pilot=base_pilot, p_repeat=0.0, seed=42)

    obs = np.zeros(11, dtype=np.float32)
    for _ in range(5):
        laggy_pilot.get_action(obs)

    assert base_pilot.call_count == 5


def test_laggy_pilot_invalid_probability():
    """Valida que lance ValueError con probabilidades inválidas fuera de [0, 1]."""
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
    """Valida la inferencia unificada con backend dummy y funciones personalizadas."""
    from src.pilots.optimal_pilot import OptimalPilot

    # Piloto dummy por defecto (hover neutro: [0, 0, 0, 0])
    pilot_default = OptimalPilot(backend="dummy")
    obs = np.zeros(11, dtype=np.float32)
    act = pilot_default.get_action(obs)
    assert np.allclose(act, np.zeros(4, dtype=np.float32))

    # Piloto dummy con función personalizada
    custom_func = lambda o: np.array([0.2, -0.3, 0.1, 0.0], dtype=np.float32)
    pilot_custom = OptimalPilot(backend="dummy", model_instance=custom_func)
    act_custom = pilot_custom.get_action(obs)
    assert np.allclose(act_custom, np.array([0.2, -0.3, 0.1, 0.0], dtype=np.float32))

    # Probar alias choose_action
    act_alias, states = pilot_custom.choose_action(obs)
    assert np.allclose(act_alias, act_custom)
    assert states == []


def test_optimal_pilot_unsupported_backend():
    """Valida que lance ValueError ante backends no reconocidos."""
    from src.pilots.optimal_pilot import OptimalPilot

    try:
        OptimalPilot(backend="backend_inexistente")
        assert False, "Debería haber lanzado ValueError ante backend no soportado"
    except ValueError:
        pass


if __name__ == "__main__":
    test_noisy_pilot_zero_noise()
    test_noisy_pilot_with_noise_and_clipping()
    test_laggy_pilot_always_repeat()
    test_laggy_pilot_never_repeat()
    test_laggy_pilot_invalid_probability()
    test_optimal_pilot_dummy()
    test_optimal_pilot_unsupported_backend()
    print("Todos los tests de pilotos pasaron exitosamente.")
