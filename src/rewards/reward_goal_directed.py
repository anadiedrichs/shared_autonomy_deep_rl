# -*- coding: utf-8 -*-
"""
Recompensa orientada a la meta para el entrenamiento del piloto.

Port conceptual de las ecuaciones 3.1 y 3.2 de la tesis (Diedrichs, 2025):
    - walls_proximity_penalization(x) = 50x - 10  (x = distancia mínima a obstáculos en metros)
    - penalization_distance_to_target(x) = 1 / x   (x = distancia al objetivo en metros)

Eventos terminales:
    - +10 al alcanzar el objetivo en altitud segura
    - -10 al caer al suelo, salir de límites o exceder puntajes/pasos
    - Normalización final al rango [-1.0, 1.0]
"""

from src.utils.utilities import normalize_to_range


def walls_proximity_penalization(distance_to_obstacle_m: float) -> float:
    """
    Ecuación 3.1 de la tesis: penalización lineal por proximidad a paredes u obstáculos.

    Args:
        distance_to_obstacle_m: Distancia al obstáculo más cercano en metros (x <= 0.10 m).

    Returns:
        float: Valor de penalización en el rango [-10.0, -5.0].
    """
    return 50.0 * float(distance_to_obstacle_m) - 10.0


def penalization_distance_to_target(distance_to_target_m: float) -> float:
    """
    Ecuación 3.2 de la tesis: recompensa inversamente proporcional a la distancia a la meta.

    Args:
        distance_to_target_m: Distancia euclídea al objetivo en metros.

    Returns:
        float: Recompensa positiva (acotada para evitar división por cero).
    """
    dist = max(float(distance_to_target_m), 0.05)
    return 1.0 / dist


def compute_goal_directed_reward(
    goal_reached: bool,
    is_terminal_failure: bool,
    min_obstacle_dist_m: float,
    dist_to_target_m: float,
    min_dist_threshold_m: float = 0.10,
    min_reward_raw: float = -10.0,
    max_reward_raw: float = 10.0,
) -> float:
    """
    Calcula la recompensa normalizada orientada a la meta para el Crazyflie.

    Args:
        goal_reached: True si el dron alcanzó el cono objetivo a altura de vuelo segura.
        is_terminal_failure: True si el dron cayó, salió del área o acumuló puntaje límite.
        min_obstacle_dist_m: Distancia mínima detectada por los sensores ToF (en metros).
        dist_to_target_m: Distancia euclídea actual a la esquina objetivo (en metros).
        min_dist_threshold_m: Umbral de distancia a pared para aplicar penalización (por defecto 0.10 m = 100 mm).
        min_reward_raw: Límite inferior para normalización (por defecto -10.0).
        max_reward_raw: Límite superior para normalización (por defecto 10.0).

    Returns:
        float: Recompensa escalar normalizada en [-1.0, 1.0].
    """
    if goal_reached:
        raw_reward = 10.0
    elif is_terminal_failure:
        raw_reward = -10.0
    else:
        raw_reward = 0.0
        # Penalización por proximidad a paredes si está dentro del umbral crítico
        if min_obstacle_dist_m <= min_dist_threshold_m:
            raw_reward += walls_proximity_penalization(min_obstacle_dist_m)
        # Recompensa por acercamiento al cono objetivo
        raw_reward += penalization_distance_to_target(dist_to_target_m)

    # Normalización al rango [-1.0, 1.0] con recorte
    return normalize_to_range(
        raw_reward, min_reward_raw, max_reward_raw, -1.0, 1.0, clip=True
    )
