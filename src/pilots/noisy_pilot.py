"""
Port continuo de NoisyPilot (tesis 2025).

Original (discreto): el agente elige una acción al azar un porcentaje del
tiempo (parámetro alpha) en lugar de ejecutar la política óptima.

Versión continua: se le suma ruido gaussiano al vector de acción del piloto
óptimo, en vez de reemplazarlo por completo por una acción aleatoria. Esto es
una decisión de diseño razonable para pasar de discreto a continuo, pero
todavía no está validada contra el original -> TODO: confirmar con la autora
si el comportamiento resultante es comparable al NoisyPilot de la tesis 2025
antes de usarlo en los experimentos finales.

TODO:
    - Definir sigma (desviación estándar del ruido) como parámetro, análogo al
      alpha de la tesis 2025 (0.05, 0.1 en los experimentos originales).
    - Confirmar si el ruido se aplica a las 4 componentes por igual o con
      sigmas distintos por componente (p.ej. más ruido en yaw que en altura).
"""

import numpy as np


class NoisyPilot:
    """Piloto óptimo + ruido gaussiano sobre la acción continua."""

    def __init__(self, optimal_pilot, sigma: float = 0.1, seed: int | None = None):
        self.optimal_pilot = optimal_pilot
        self.sigma = sigma
        self._rng = np.random.default_rng(seed)

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """
        TODO:
            a_opt = self.optimal_pilot.get_action(observation)
            noise = self._rng.normal(0, self.sigma, size=a_opt.shape)
            return np.clip(a_opt + noise, -1, 1)
        """
        raise NotImplementedError
