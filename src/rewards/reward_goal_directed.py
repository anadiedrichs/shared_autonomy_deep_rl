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
    Ecuación 3.2 de la tesis: recompensa inversamente proporcional a la distancia a la meta (conservada por compatibilidad).

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
    prev_dist_to_target_m: float | None = None,
    min_dist_threshold_m: float = 0.10,
    goal_reward: float = 100.0,
    failure_reward: float = -10.0,
    progress_weight: float = 30.0,
    time_penalty: float = 0.01,
) -> float:
    """
    Calcula la recompensa orientada a la meta para el Crazyflie.

    Usa potential-based reward shaping diferencial (CornerEnvShaped20) con penalización
    por tiempo para evitar el 'reward farming' por hovering inactivo.

    Args:
        goal_reached: True si el dron alcanzó el cono objetivo a altura de vuelo segura (+100.0).
        is_terminal_failure: True si el dron cayó o salió del área delimitada (-10.0).
        min_obstacle_dist_m: Distancia mínima detectada por los sensores ToF (en metros).
        dist_to_target_m: Distancia euclídea actual a la esquina objetivo (en metros).
        prev_dist_to_target_m: Distancia euclídea en el paso anterior (para calcular avance diferencial).
        min_dist_threshold_m: Umbral de distancia a pared para aplicar penalización (por defecto 0.10 m = 100 mm).
        goal_reward: Recompensa por alcanzar la meta (por defecto 100.0).
        failure_reward: Penalización por caída o salida de límites (por defecto -10.0).
        progress_weight: Factor multiplicativo para el avance diferencial hacia el objetivo (por defecto 30.0).
        time_penalty: Coste por cada paso transcurrido para desincentivar el hovering estático (por defecto 0.01).

    Returns:
        float: Recompensa escalar del paso.
    """
    if goal_reached:
        return float(goal_reward)

    if is_terminal_failure:
        return float(failure_reward)

    # Coste temporal por paso
    reward = -float(time_penalty)

    # Penalización por proximidad a paredes si está dentro del umbral crítico
    if min_obstacle_dist_m <= min_dist_threshold_m:
        reward += walls_proximity_penalization(min_obstacle_dist_m)

    # Recompensa por avance diferencial hacia el cono (Potential-Based Shaping)
    if prev_dist_to_target_m is not None:
        progress = float(prev_dist_to_target_m) - float(dist_to_target_m)
        reward += float(progress_weight) * progress

    return float(reward)
