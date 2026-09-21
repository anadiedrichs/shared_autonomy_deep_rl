"""
Piloto óptimo: carga una política ya entrenada (RLtools-SAC o SB3-PPO) y expone
una interfaz común de inferencia.

No hay aprendizaje por imitación en esta iteración (ver docs/architecture.md,
sección 2): el piloto óptimo sale 100% de RL.

TODO:
    - Implementar carga de checkpoint de rltools (state.export_policy() /
      load_checkpoint_from_path, ver README de rl-tools/python-interface).
    - Implementar carga de modelo SB3 (PPO.load(path)).
    - Unificar ambos bajo la misma interfaz get_action(observation).
"""

import numpy as np


class OptimalPilot:
    """Interfaz común para el piloto óptimo, sin importar el backend de entrenamiento."""

    def __init__(self, backend: str, checkpoint_path: str):
        """
        Args:
            backend: "rltools_sac" o "sb3_ppo".
            checkpoint_path: ruta al checkpoint entrenado.
        """
        self.backend = backend
        self.checkpoint_path = checkpoint_path
        self._model = None  # TODO: cargar según backend

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """Devuelve un vector de acción en [-1, 1]^4."""
        raise NotImplementedError
