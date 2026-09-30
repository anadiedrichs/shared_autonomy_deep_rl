# -*- coding: utf-8 -*-
"""
Recompensa goal-agnostic (R_general) para entrenar la política residual del copiloto.

Siguiendo el esquema de Schaff & Walter (2020), el copiloto se entrena para minimizar su
propia intervención mientras previene colisiones y violaciones de seguridad, sin conocer
el cono objetivo al que se dirige el piloto humano o sintético.

Componentes:
    1. Penalización exponencial por proximidad a obstáculos/paredes.
    2. Penalización por magnitud de la acción residual: -lambda * ||a_r||^2 (mínima intervención).
    3. Penalización severa por colisión o caída: evento terminal (-50.0).
    4. Bonificación opcional por paso sobrevivido en vuelo seguro.
"""

import numpy as np

from src.rewards.reward_goal_directed import obstacle_proximity_penalization


def general_safety_reward(
    distance_to_obstacle_m: float,
    residual_action: np.ndarray | list[float],
    collided: bool = False,
    lambda_intervention: float = 0.10,
    obstacle_weight: float = 1.0,
    obstacle_decay: float = 0.15,
    collision_penalty: float = 50.0,
    alive_bonus: float = 0.0,
) -> float:
    """
    Calcula la recompensa de seguridad continua y agnóstica a la meta para el copiloto residual.

    Args:
        distance_to_obstacle_m: Distancia mínima detectada a obstáculos/paredes en metros.
        residual_action: Vector de acción residual a_r en [-1.0, 1.0]^4.
        collided: True si el dron colisionó contra pared, cayó al suelo o salió de límites.
        lambda_intervention: Coeficiente de penalización por intervención (peso de ||a_r||^2).
        obstacle_weight: Amplitud de la penalización exponencial por cercanía a obstáculos.
        obstacle_decay: Distancia característica de decaimiento en metros (por defecto 0.15 m).
        collision_penalty: Penalización aplicada en el evento terminal de colisión (por defecto 50.0).
        alive_bonus: Bonificación por cada paso en vuelo seguro sin colisionar (por defecto 0.0).

    Returns:
        float: Recompensa escalar para el copiloto en el paso actual.
    """
    # 1. Penalización por cercanía a paredes u obstáculos (proximidad suave)
    r_obstacle = obstacle_proximity_penalization(
        distance_to_obstacle_m, amplitude=obstacle_weight, decay=obstacle_decay
    )

    # 2. Penalización por magnitud del residual ||a_r||^2 para promover mínima intervención
    a_r = np.asarray(residual_action, dtype=np.float32)
    intervention_magnitude = float(np.dot(a_r, a_r))
    r_intervention = -float(lambda_intervention) * intervention_magnitude

    # 3. Recompensa base
    reward = r_obstacle + r_intervention + float(alive_bonus)

    # 4. Evento terminal de colisión / caída
    if collided:
        reward -= float(collision_penalty)

    return float(reward)


def compute_general_safety_reward(
    distance_to_obstacle_m: float | None = None,
    residual_action: np.ndarray | list[float] | None = None,
    collided: bool = False,
    dist_sensors_mm: list[float] | tuple[float, ...] | np.ndarray | None = None,
    lambda_intervention: float = 0.10,
    obstacle_weight: float = 1.0,
    obstacle_decay: float = 0.15,
    collision_penalty: float = 50.0,
    alive_bonus: float = 0.0,
) -> float:
    """
    Función de utilidad para calcular la recompensa de seguridad a partir de sensores ToF o distancia directa.

    Args:
        distance_to_obstacle_m: Distancia en metros al obstáculo más cercano.
        residual_action: Vector de acción residual a_r.
        collided: True si ocurrió colisión o caída.
        dist_sensors_mm: Lecturas ToF en milímetros [front, back, right, left] (opcional).
        lambda_intervention: Coeficiente de intervención.
        obstacle_weight: Amplitud de penalización por obstáculos.
        obstacle_decay: Decaimiento espacial de obstáculos.
        collision_penalty: Penalización por colisión.
        alive_bonus: Bonificación de supervivencia.

    Returns:
        float: Recompensa escalar calculada.
    """
    if distance_to_obstacle_m is None:
        if dist_sensors_mm is not None and len(dist_sensors_mm) > 0:
            distance_to_obstacle_m = float(min(dist_sensors_mm)) / 1000.0
        else:
            distance_to_obstacle_m = 2.0  # Sin obstáculos cercanos por defecto

    act_r = residual_action if residual_action is not None else np.zeros(4, dtype=np.float32)

    return general_safety_reward(
        distance_to_obstacle_m=distance_to_obstacle_m,
        residual_action=act_r,
        collided=collided,
        lambda_intervention=lambda_intervention,
        obstacle_weight=obstacle_weight,
        obstacle_decay=obstacle_decay,
        collision_penalty=collision_penalty,
        alive_bonus=alive_bonus,
    )
