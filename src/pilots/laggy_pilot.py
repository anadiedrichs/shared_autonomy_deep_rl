"""
Port continuo de LaggedPilot (tesis 2025).

Original (discreto): obliga al agente a repetir su acción anterior con cierta
probabilidad (parámetro alpha), simulando el retardo de reacción humano.

Versión continua: mismo mecanismo, pero repitiendo el último vector de acción
continuo en vez de la última acción discreta. Es un port directo, sin cambios
conceptuales.

TODO:
    - Portar el parámetro alpha de los experimentos de la tesis 2025 (0.2, 0.8).
"""

import numpy as np


class LaggyPilot:
    """Piloto óptimo que repite su última acción con probabilidad `p_repeat`."""

    def __init__(self, optimal_pilot, p_repeat: float = 0.2, seed: int | None = None):
        self.optimal_pilot = optimal_pilot
        self.p_repeat = p_repeat
        self._rng = np.random.default_rng(seed)
        self._last_action: np.ndarray | None = None

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """
        TODO:
            if self._last_action is not None and self._rng.random() < self.p_repeat:
                return self._last_action
            a = self.optimal_pilot.get_action(observation)
            self._last_action = a
            return a
        """
        raise NotImplementedError
