# -*- coding: utf-8 -*-
"""
Unit tests for the HeuristicPilot class (Approach 1: open_loop, Approach 2: proportional).
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

from src.pilots.heuristic_pilot import HeuristicPilot
from src.pilots.laggy_pilot import LaggyPilot
from src.pilots.noisy_pilot import NoisyPilot
from src.pilots.optimal_pilot import OptimalPilot


class MockDroneEnv:
    """Lightweight mock of CornerEnvContinuous for closed-loop testing."""

    def __init__(self, x=0.0, y=0.0, z=0.30, yaw=0.0):
        self.x_global = x
        self.y_global = y
        self.alt = z
        self.z_global = z
        self.yaw = yaw


def test_heuristic_pilot_open_loop_default():
    """Validate constant action generation in open loop (Approach 1)."""
    # Quadrant 1: cone_1 (+x, +y in global -> forward and right in body frame)
    pilot1 = HeuristicPilot(mode="open_loop", corner_name="cone_1")
    obs = np.zeros(11, dtype=np.float32)
    act1 = pilot1.get_action(obs)
    assert act1.shape == (4,)
    assert act1[0] > 0.0  # vx > 0 (forward)
    assert act1[1] < 0.0  # vy < 0 (right)
    assert act1[2] == 0.0  # yaw = 0
    assert act1[3] > 0.0  # vz > 0 (climb)

    # Quadrant 2: cone_2 (-x, +y in global -> forward and left in body frame)
    pilot2 = HeuristicPilot(mode="open_loop", corner_name="cone_2")
    act2 = pilot2.get_action(obs)
    assert act2[0] > 0.0  # vx > 0 (forward)
    assert act2[1] > 0.0  # vy > 0 (left)

    # Quadrant 3: cone_3 (-x, -y in global -> backward and left in body frame)
    pilot3 = HeuristicPilot(mode="enfoque1", corner_name="cone_3")
    act3 = pilot3.get_action(obs)
    assert act3[0] < 0.0  # vx < 0 (backward)
    assert act3[1] > 0.0  # vy > 0 (left)
    assert act3[3] > 0.0


def test_heuristic_pilot_open_loop_custom_action():
    """Validate that a custom fixed action is respected and clipped to [-1, 1]."""
    custom_act = [0.75, -0.65, 0.10, 0.40]
    pilot = HeuristicPilot(mode="fixed", fixed_action=custom_act)
    obs = np.zeros(11, dtype=np.float32)
    act = pilot.get_action(obs)
    assert np.allclose(act, custom_act)


def test_heuristic_pilot_proportional_climbing_and_direction():
    """Validate error computation and saturation in closed loop (Approach 2)."""
    env_mock = MockDroneEnv(x=0.0, y=0.0, z=0.30, yaw=0.0)
    pilot = HeuristicPilot(
        mode="proportional",
        corner_name="cone_1",
        env=env_mock,
        target_altitude=0.55,
    )

    obs = np.zeros(11, dtype=np.float32)
    act = pilot.get_action(obs)

    # Far from target at (0.9, 0.9): saturation to maximum forward and right velocity
    assert np.isclose(act[0], 1.0)
    assert np.isclose(act[1], 1.0)
    assert act[2] == 0.0
    # Below 0.55m: must command climb
    assert act[3] > 0.0


def test_heuristic_pilot_proportional_hover_at_target():
    """Validate that it decelerates and hovers stably upon reaching the target position."""
    # Positioned exactly over cone_1 at altitude 0.55m
    env_mock = MockDroneEnv(x=0.9, y=0.9, z=0.55, yaw=0.0)
    pilot = HeuristicPilot(
        mode="enfoque2",
        corner_name="cone_1",
        env=env_mock,
        target_altitude=0.55,
    )

    obs = np.zeros(11, dtype=np.float32)
    act = pilot.get_action(obs)

    # Inside the target zone and at desired altitude: commands close to zero
    assert np.isclose(act[0], 0.0, atol=1e-3)
    assert np.isclose(act[1], 0.0, atol=1e-3)
    assert np.isclose(act[3], 0.0, atol=1e-3)


def test_heuristic_pilot_body_frame_rotation():
    """Validate that the transformation to the drone's body frame rotates properly."""
    # Drone rotated 90 degrees (pi / 2 rad) to the left
    # To go to (0.9, 0.0), in body frame that is to its RIGHT (-y_body)
    env_mock = MockDroneEnv(x=0.0, y=0.0, z=0.55, yaw=np.pi / 2.0)
    pilot = HeuristicPilot(
        mode="proportional",
        target_position=np.array([0.9, 0.0, 0.55]),
        env=env_mock,
    )

    obs = np.zeros(11, dtype=np.float32)
    act = pilot.get_action(obs)

    # With yaw = 90 deg: advancing toward global +X requires lateral movement to the right (-vy in body frame)
    assert np.isclose(act[0], 0.0, atol=1e-2)
    assert act[1] < -0.5


def test_optimal_pilot_integration_with_heuristic():
    """Validate that OptimalPilot can instantiate and use HeuristicPilot as a backend."""
    pilot = OptimalPilot(backend="heuristic", mode="open_loop", corner_name="cone_1")
    obs = np.zeros(11, dtype=np.float32)
    act = pilot.get_action(obs)
    assert act.shape == (4,)
    assert act[0] > 0.0
    action_alias, _ = pilot.choose_action(obs)
    assert np.allclose(act, action_alias)


def test_noisy_and_laggy_with_heuristic_base():
    """Validate that NoisyPilot and LaggyPilot accept HeuristicPilot as a base pilot."""
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
