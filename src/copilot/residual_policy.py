"""
Política residual del copiloto: pi_r(s, a_h) -> a_r.

Reemplaza el mecanismo de la tesis 2025 (tabla de distancia Q(a*, ah) + umbral
alpha, que elegía entre acción del piloto o del copiloto) por una corrección
aditiva y continua, entrenada con RL contra una recompensa goal-agnostic
(ver src/rewards/reward_general_safety.py), siguiendo Schaff & Walter (2020).

No hay equivalente directo a portar desde drone-deep-rl: este módulo es nuevo.

TODO:
    - Definir la observación que recibe pi_r: ¿la misma observación del drone
      + la acción del piloto concatenada (obs_extendida = [obs, a_h]), como en
      la tesis 2025 (donde el copiloto tenía "una observación extra: la acción
      del piloto")?
    - Decidir el backend de entrenamiento (RLtools-SAC, por consistencia con el
      piloto óptimo, o SB3 -- ver pregunta abierta en docs/architecture.md).
    - Implementar carga de checkpoint entrenado, análoga a OptimalPilot.
"""

import numpy as np


class ResidualPolicy:
    """Interfaz de inferencia del copiloto: observación + acción del piloto -> a_r."""

    def __init__(self, checkpoint_path: str | None = None):
        self.checkpoint_path = checkpoint_path
        self._model = None  # TODO: cargar checkpoint si se provee

    def get_action(self, observation: np.ndarray, pilot_action: np.ndarray) -> np.ndarray:
        """
        Devuelve a_r en [-1, 1]^4 (o en un rango más acotado, a definir, ver
        pregunta abierta en docs/architecture.md sobre si el residual debe
        tener un rango más chico que el del piloto).
        """
        raise NotImplementedError
