"""
Recompensa orientada a la meta, para entrenar al piloto (óptimo).

Port conceptual de get_reward() en SimpleCornerEnvRS10 (tesis 2025, sección
3.5.2, ecuaciones 3.1 y 3.2):

    walls_proximity_penalization(x) = 50x - 10       (x = distancia a obstáculo)
    penalization_distance_to_target(x) = 1 / x        (x = distancia al objetivo)

Más:
    +10 al alcanzar el objetivo
    -10 al caer al suelo / salir del escenario
    normalización final a [-1, 1]

TODO:
    - Confirmar si esta función vive acá o directamente dentro de
      CornerEnvContinuous.get_reward() (hay redundancia potencial a resolver:
      ver TODO en corner_env_continuous.py).
"""


def walls_proximity_penalization(distance_to_obstacle: float) -> float:
    """TODO: portar ecuación 3.1 de la tesis 2025."""
    raise NotImplementedError


def penalization_distance_to_target(distance_to_target: float) -> float:
    """TODO: portar ecuación 3.2 de la tesis 2025."""
    raise NotImplementedError
