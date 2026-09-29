# -*- coding: utf-8 -*-
"""
Tests unitarios para la clase HeuristicPilot (Enfoque 1: open_loop, Enfoque 2: proportional).
"""

import numpy as np
try:
    import pytest
except ImportError:
    pytest = None

from src.pilots.heuristic_pilot import HeuristicPilot
from src.pilots.laggy_pilot import LaggyPilot
from src.pilots.noisy_pilot import NoisyPilot
from src.pilots.optimal_pilot import OptimalPilot


class MockDroneEnv:
    """Mock ligero de CornerEnvContinuous para pruebas de lazo cerrado."""

    def __init__(self, x=0.0, y=0.0, z=0.30, yaw=0.0):
        self.x_global = x
        self.y_global = y
        self.alt = z
        self.z_global = z
        self.yaw = yaw


def test_heuristic_pilot_open_loop_default():
    """Valida la generación de acciones constantes en lazo abierto (Enfoque 1)."""
    # Cuadrante 1: cone_1 (+x, +y)
    pilot1 = HeuristicPilot(mode="open_loop", corner_name="cone_1")
    obs = np.zeros(11, dtype=np.float32)
    act1 = pilot1.get_action(obs)
    assert act1.shape == (4,)
    assert act1[0] > 0.0  # vx > 0
    assert act1[1] > 0.0  # vy > 0
    assert act1[2] == 0.0  # yaw = 0
    assert act1[3] > 0.0  # vz > 0 (ascenso)

    # Cuadrante 3: cone_3 (-x, -y)
    pilot3 = HeuristicPilot(mode="enfoque1", corner_name="cone_3")
    act3 = pilot3.get_action(obs)
    assert act3[0] < 0.0  # vx < 0
    assert act3[1] < 0.0  # vy < 0
    assert act3[3] > 0.0


def test_heuristic_pilot_open_loop_custom_action():
    """Valida que una acción fija personalizada se respete y recorte a [-1, 1]."""
    custom_act = [0.75, -0.65, 0.10, 0.40]
    pilot = HeuristicPilot(mode="fixed", fixed_action=custom_act)
    obs = np.zeros(11, dtype=np.float32)
    act = pilot.get_action(obs)
    assert np.allclose(act, custom_act)


def test_heuristic_pilot_proportional_climbing_and_direction():
    """Valida el cálculo de errores y saturación en lazo cerrado (Enfoque 2)."""
    env_mock = MockDroneEnv(x=0.0, y=0.0, z=0.30, yaw=0.0)
    pilot = HeuristicPilot(
        mode="proportional",
        corner_name="cone_1",
        env=env_mock,
        target_altitude=0.55,
    )

    obs = np.zeros(11, dtype=np.float32)
    act = pilot.get_action(obs)

    # Lejos del objetivo en (0.9, 0.9): saturación a velocidad máxima hacia adelante y derecha
    assert np.isclose(act[0], 1.0)
    assert np.isclose(act[1], 1.0)
    assert act[2] == 0.0
    # Por debajo de 0.55m: debe comandar ascenso
    assert act[3] > 0.0


def test_heuristic_pilot_proportional_hover_at_target():
    """Valida que desacelere y haga hover estable al alcanzar la posición objetivo."""
    # Posicionado exactamente sobre el cono_1 a altitud 0.55m
    env_mock = MockDroneEnv(x=0.9, y=0.9, z=0.55, yaw=0.0)
    pilot = HeuristicPilot(
        mode="enfoque2",
        corner_name="cone_1",
        env=env_mock,
        target_altitude=0.55,
    )

    obs = np.zeros(11, dtype=np.float32)
    act = pilot.get_action(obs)

    # Dentro de la zona objetivo y en altura deseada: comandos cercanos a cero
    assert np.isclose(act[0], 0.0, atol=1e-3)
    assert np.isclose(act[1], 0.0, atol=1e-3)
    assert np.isclose(act[3], 0.0, atol=1e-3)


def test_heuristic_pilot_body_frame_rotation():
    """Valida que la transformación al marco del dron (body frame) rote adecuadamente."""
    # Dron rotado 90 grados (pi / 2 rad) hacia la izquierda
    # Para ir a (0.9, 0.0), en body frame eso está a su DERECHA (-y_body)
    env_mock = MockDroneEnv(x=0.0, y=0.0, z=0.55, yaw=np.pi / 2.0)
    pilot = HeuristicPilot(
        mode="proportional",
        target_position=np.array([0.9, 0.0, 0.55]),
        env=env_mock,
    )

    obs = np.zeros(11, dtype=np.float32)
    act = pilot.get_action(obs)

    # Con yaw = 90 deg: el avance hacia +X global requiere desplazamiento lateral a la derecha (-vy en body frame)
    assert np.isclose(act[0], 0.0, atol=1e-2)
    assert act[1] < -0.5


def test_optimal_pilot_integration_with_heuristic():
    """Valida que OptimalPilot pueda instanciar y usar HeuristicPilot como backend."""
    pilot = OptimalPilot(backend="heuristic", mode="open_loop", corner_name="cone_1")
    obs = np.zeros(11, dtype=np.float32)
    act = pilot.get_action(obs)
    assert act.shape == (4,)
    assert act[0] > 0.0
    action_alias, _ = pilot.choose_action(obs)
    assert np.allclose(act, action_alias)


def test_noisy_and_laggy_with_heuristic_base():
    """Valida que NoisyPilot y LaggyPilot acepten HeuristicPilot como piloto base."""
    base_pilot = HeuristicPilot(mode="open_loop", corner_name="cone_1")
    noisy = NoisyPilot(optimal_pilot=base_pilot, sigma=0.05, seed=42)
    laggy = LaggyPilot(optimal_pilot=base_pilot, p_repeat=0.5, seed=42)

    obs = np.zeros(11, dtype=np.float32)
    act_noisy = noisy.get_action(obs)
    act_laggy = laggy.get_action(obs)

    assert act_noisy.shape == (4,)
    assert act_laggy.shape == (4,)
    assert np.all(act_noisy >= -1.0) and np.all(act_noisy <= 1.0)
    assert np.all(act_laggy >= -1.0) and np.all(act_laggy <= 1.0)


if __name__ == "__main__":
    test_heuristic_pilot_open_loop_default()
    test_heuristic_pilot_open_loop_custom_action()
    test_heuristic_pilot_proportional_climbing_and_direction()
    test_heuristic_pilot_proportional_hover_at_target()
    test_heuristic_pilot_body_frame_rotation()
    test_optimal_pilot_integration_with_heuristic()
    test_noisy_and_laggy_with_heuristic_base()
    print("Todos los tests de HeuristicPilot pasaron exitosamente.")
