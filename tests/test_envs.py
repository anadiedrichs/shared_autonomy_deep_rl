# -*- coding: utf-8 -*-
"""
Tests unitarios para DroneRobotSupervisor y utilidades del entorno.
"""

import numpy as np
try:
    import pytest
except ImportError:
    pytest = None
from src.envs.drone_robot_supervisor import DroneRobotSupervisor
from src.utils.utilities import normalize_to_range


def test_normalize_to_range():
    """Valida la función de normalización lineal con y sin recorte."""
    # Mapeo simple de [0, 10] a [-1, 1]
    assert np.isclose(normalize_to_range(5.0, 0.0, 10.0, -1.0, 1.0), 0.0)
    assert np.isclose(normalize_to_range(0.0, 0.0, 10.0, -1.0, 1.0), -1.0)
    assert np.isclose(normalize_to_range(10.0, 0.0, 10.0, -1.0, 1.0), 1.0)

    # Recorte con clip=True
    assert np.isclose(normalize_to_range(15.0, 0.0, 10.0, -1.0, 1.0, clip=True), 1.0)
    assert np.isclose(normalize_to_range(-5.0, 0.0, 10.0, -1.0, 1.0, clip=True), -1.0)


def test_drone_robot_supervisor_spaces():
    """Valida que los espacios de acción y observación tengan las dimensiones y tipos acordados."""
    env = DroneRobotSupervisor()

    # Espacio de acción continuo: 4 dimensiones en [-1, 1]
    assert env.action_space.shape == (4,)
    assert env.action_space.dtype == np.float32
    assert np.all(env.action_space.low == -1.0)
    assert np.all(env.action_space.high == 1.0)

    # Espacio de observación continuo: 11 dimensiones en [-1, 1]
    assert env.observation_space.shape == (11,)
    assert env.observation_space.dtype == np.float32
    assert np.all(env.observation_space.low == -1.0)
    assert np.all(env.observation_space.high == 1.0)


def test_denormalize_action():
    """Valida la conversión física del vector de acción continuo."""
    env = DroneRobotSupervisor()
    env.dt = 0.032
    env.height_desired = 0.30

    # Acción cero: velocidades nulas y altura sin cambio
    action_zero = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    vx, vy, yaw_rate, h = env._denormalize_action(action_zero)
    assert np.isclose(vx, 0.0)
    assert np.isclose(vy, 0.0)
    assert np.isclose(yaw_rate, 0.0)
    assert np.isclose(h, 0.30)

    # Acción máxima: límites máximos configurados
    action_max = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    vx, vy, yaw_rate, h = env._denormalize_action(action_max)
    assert np.isclose(vx, env.MAX_FORWARD_VELOCITY)
    assert np.isclose(vy, env.MAX_LATERAL_VELOCITY)
    assert np.isclose(yaw_rate, env.MAX_YAW_RATE)
    assert h > 0.30  # Altura debe haber incrementado


def test_drone_robot_supervisor_reset_and_step():
    """Valida que reset() y step() funcionen correctamente en modo desacoplado / mock."""
    env = DroneRobotSupervisor()

    # Reset
    obs, info = env.reset(seed=42)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (11,)
    assert isinstance(info, dict)
    assert "is_success" in info

    # Step con acción válida dentro de los límites
    action = np.array([0.5, -0.5, 0.1, 0.0], dtype=np.float32)
    next_obs, reward, terminated, truncated, info = env.step(action)

    assert next_obs.shape == (11,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    assert env.episode_step == 1


def test_reward_goal_directed():
    """Valida las fórmulas de recompensa orientada a la meta."""
    from src.rewards.reward_goal_directed import (
        walls_proximity_penalization,
        penalization_distance_to_target,
        compute_goal_directed_reward,
    )

    # Ecuación 3.1: 50x - 10
    assert np.isclose(walls_proximity_penalization(0.10), -5.0)
    assert np.isclose(walls_proximity_penalization(0.0), -10.0)

    # Ecuación 3.2: 1 / x
    assert np.isclose(penalization_distance_to_target(1.0), 1.0)
    assert np.isclose(penalization_distance_to_target(0.5), 2.0)

    # Meta alcanzada (+10 normalizado a +1.0)
    r_goal = compute_goal_directed_reward(
        goal_reached=True,
        is_terminal_failure=False,
        min_obstacle_dist_m=0.5,
        dist_to_target_m=0.1,
    )
    assert np.isclose(r_goal, 1.0)

    # Falla terminal (-10 normalizado a -1.0)
    r_fail = compute_goal_directed_reward(
        goal_reached=False,
        is_terminal_failure=True,
        min_obstacle_dist_m=0.5,
        dist_to_target_m=1.0,
    )
    assert np.isclose(r_fail, -1.0)


def test_corner_env_continuous():
    """Valida el entorno CornerEnvContinuous y sus condiciones lógicas."""
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
    assert -1.0 <= reward <= 1.0
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)

    # Prueba de límites geométricos
    env.x_global = 1.5
    assert env.is_out_of_bounds() is True
    env.x_global = 0.0
    assert env.is_out_of_bounds() is False

    # Prueba de caída al suelo (alt < 0.10)
    env.alt = 0.05
    assert env.is_done() is True
    assert env.corner == "fall"

    # Prueba de objetivo alcanzado en vuelo seguro (cerca del objetivo a [0.8, 0.8])
    env.alt = 0.55
    env.x_global = 0.80
    env.y_global = 0.80
    assert env.achieve_goal() is True


def test_verbose_mode():
    """Valida la configuración del modo verbose en los entornos."""
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
