# -*- coding: utf-8 -*-
"""
Tests unitarios para la función de recompensa general_safety_reward y compute_general_safety_reward.
"""

import os
import sys
import numpy as np

# Asegurar que la raíz del proyecto esté en el PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.rewards.reward_general_safety import (
    compute_general_safety_reward,
    general_safety_reward,
)


def test_zero_intervention_far_from_obstacles():
    """Valida que sin intervención residual y lejos de obstáculos la recompensa sea cercana a cero."""
    residual = np.zeros(4, dtype=np.float32)
    # Lejos de obstáculos (2.0 metros)
    r = general_safety_reward(
        distance_to_obstacle_m=2.0,
        residual_action=residual,
        collided=False,
        lambda_intervention=0.1,
    )
    # Proximidad exponencial a 2m con decay 0.15 es despreciable (< 1e-5)
    assert np.isclose(r, 0.0, atol=1e-3)


def test_intervention_penalty_scaling():
    """Valida que la penalización por magnitud del residual escale como -lambda * ||a_r||^2."""
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

    # Con acción de magnitud máxima ||[1, 1, 1, 1]||^2 = 4.0
    residual_max = np.ones(4, dtype=np.float32)
    r_max = general_safety_reward(
        distance_to_obstacle_m=2.0,
        residual_action=residual_max,
        collided=False,
        lambda_intervention=lambda_val,
    )
    assert np.isclose(r_max, -lambda_val * 4.0, atol=1e-3)


def test_obstacle_proximity_penalty():
    """Valida que la cercanía a obstáculos penalice adecuadamente según la distancia."""
    residual = np.zeros(4, dtype=np.float32)
    # Justo en contacto con el obstáculo (d = 0.0 m)
    r_contact = general_safety_reward(
        distance_to_obstacle_m=0.0,
        residual_action=residual,
        collided=False,
        obstacle_weight=1.5,
        obstacle_decay=0.15,
    )
    # exp(0) = 1 -> -1.5
    assert np.isclose(r_contact, -1.5, atol=1e-3)

    # A 0.15m (1 tau de distancia): -1.5 * exp(-1) = -1.5 * 0.367879 = -0.5518
    r_tau = general_safety_reward(
        distance_to_obstacle_m=0.15,
        residual_action=residual,
        collided=False,
        obstacle_weight=1.5,
        obstacle_decay=0.15,
    )
    assert np.isclose(r_tau, -1.5 * np.exp(-1.0), atol=1e-3)


def test_collision_terminal_penalty():
    """Valida que un evento terminal de colisión aplique la penalización collision_penalty."""
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
    """Valida la función helper con lecturas de sensores ToF en milímetros."""
    # Sensores: front=1000mm, back=150mm (más cercano), right=500mm, left=800mm
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


if __name__ == "__main__":
    test_zero_intervention_far_from_obstacles()
    test_intervention_penalty_scaling()
    test_obstacle_proximity_penalty()
    test_collision_terminal_penalty()
    test_compute_general_safety_reward_with_sensors()
    print("Todos los tests de recompensa de seguridad general pasaron exitosamente.")
