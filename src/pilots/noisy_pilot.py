# -*- coding: utf-8 -*-
"""
Piloto sintético con inyección de ruido gaussiano sobre la acción continua.

Simula imprecisión de pilotaje humano o temblor en los comandos de control continuo,
sumando una perturbación gaussiana N(0, sigma^2) a la política del piloto óptimo.
"""

import numpy as np


class NoisyPilot:
    """
    Piloto sintético que perturba las acciones continuas del piloto óptimo con ruido gaussiano.
    """

    def __init__(
        self,
        optimal_pilot,
        sigma: float | np.ndarray = 0.1,
        seed: int | None = None,
    ):
        """
        Inicializa el NoisyPilot.

        Args:
            optimal_pilot: Instancia del piloto base u óptimo (debe proveer get_action o predict).
            sigma: Desviación estándar del ruido gaussiano (escalar o vector de 4 elementos).
            seed: Semilla para el generador de números aleatorios independiente.
        """
        self.optimal_pilot = optimal_pilot
        self.sigma = sigma
        self._rng = np.random.default_rng(seed)

    def _get_base_action(self, observation: np.ndarray) -> np.ndarray:
        """
        Obtiene la acción base propuesta por el piloto óptimo según su interfaz disponible.
        """
        if hasattr(self.optimal_pilot, "get_action"):
            act = self.optimal_pilot.get_action(observation)
        elif hasattr(self.optimal_pilot, "predict"):
            act, _ = self.optimal_pilot.predict(observation, deterministic=True)
        elif hasattr(self.optimal_pilot, "choose_action"):
            act_res = self.optimal_pilot.choose_action(observation)
            act = act_res[0] if isinstance(act_res, tuple) else act_res
        else:
            raise TypeError("El objeto optimal_pilot no expone get_action, predict ni choose_action.")

        return np.asarray(act, dtype=np.float32)

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """
        Calcula la acción perturbada sumando ruido gaussiano y recortando al rango [-1.0, 1.0].

        Args:
            observation: Vector de observaciones del entorno.

        Returns:
            np.ndarray: Vector de acción de 4 dimensiones en [-1.0, 1.0].
        """
        base_action = self._get_base_action(observation)
        noise = self._rng.normal(0.0, self.sigma, size=base_action.shape).astype(np.float32)
        noisy_action = np.clip(base_action + noise, -1.0, 1.0)
        return noisy_action

    def choose_action(self, observation: np.ndarray) -> tuple[np.ndarray, list]:
        """
        Alias compatible con la interfaz histórica de la tesis (Diedrichs, 2025).

        Returns:
            tuple: (acción seleccionada, lista de estados auxiliares vacía).
        """
        return self.get_action(observation), []
