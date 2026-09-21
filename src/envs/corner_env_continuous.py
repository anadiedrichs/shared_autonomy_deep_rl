"""
Port continuo de SimpleCornerEnvRS10 (tesis 2025): habitación 2x2m, objetivo en
una de las 4 esquinas.

Origen a portar:
    https://github.com/anadiedrichs/drone-deep-rl
    (v1_shared_autonomy, clase SimpleCornerEnvRS10)

Qué se mantiene igual:
    - Escenario Webots (habitación cerrada 2x2m, 4 esquinas como posibles objetivos).
    - Criterios de is_done(): drone en el suelo (truncado), esquina alcanzada
      (éxito), puntaje fuera de rango (truncado).

Qué cambia:
    - get_reward(): las ecuaciones 3.1 (walls_proximity_penalization) y 3.2
      (penalization_distance_to_target) de la tesis 2025 ya eran funciones de la
      distancia a obstáculos/objetivo, no de la acción discreta elegida -> se
      mantienen conceptualmente iguales, solo hay que confirmar que no dependan
      en ningún punto del índice de acción discreta.

TODO:
    - Portar _init_targets(), _initialization(), is_done(), get_reward() desde
      la tesis 2025.
    - Decidir si esta clase se usa tal cual como "reward orientada a la meta"
      para el piloto (ver src/rewards/reward_goal_directed.py) o si conviene
      separar reward y entorno en dos archivos distintos.
"""

from src.envs.drone_robot_supervisor import DroneRobotSupervisor


class CornerEnvContinuous(DroneRobotSupervisor):
    """
    Entorno de la habitación 2x2m con objetivo en una esquina, acciones continuas.
    """

    # TODO: portar constantes de la tesis 2025 (DISTANCE_THRESHOLD,
    # MIN_DIST_OBSTACLES, MAX_EPISODE_SCORE, MIN_EPISODE_SCORE)
    DISTANCE_THRESHOLD = None
    MIN_DIST_OBSTACLES = None
    MAX_EPISODE_SCORE = None
    MIN_EPISODE_SCORE = None

    def __init__(self, max_episode_steps: int = 10_000):
        super().__init__(max_episode_steps=max_episode_steps)
        # TODO: _init_targets(), _initialization()

    def is_done(self) -> bool:
        """TODO: portar desde la tesis 2025 (sección 3.5.2)."""
        raise NotImplementedError

    def get_reward(self) -> float:
        """
        TODO: portar ecuaciones 3.1 y 3.2 de la tesis 2025:
            walls_proximity_penalization(x) = 50x - 10
            penalization_distance_to_target(x) = 1 / x
        Normalizar a [-1, 1] como en el original.
        """
        raise NotImplementedError
