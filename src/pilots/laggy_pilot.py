# -*- coding: utf-8 -*-
"""
Piloto sintético con retardo de reacción (LaggyPilot).

Simula el tiempo de respuesta o retardo humano repitiendo la acción ejecutada en el
paso anterior con una probabilidad 'p_repeat' (parámetro alpha de la tesis).
"""

import numpy as np


class LaggyPilot:
    """
    Piloto sintético que introduce dependencia temporal al repetir ocasionalmente la acción previa.
    """

    def __init__(
        self,
        optimal_pilot,
        p_repeat: float = 0.2,
        seed: int | None = None,
    ):
        """
        Inicializa el LaggyPilot.

        Args:
            optimal_pilot: Instancia del piloto base u óptimo.
            p_repeat: Probabilidad de repetir la última acción continua (entre 0.0 y 1.0).
            seed: Semilla para el generador de números aleatorios independiente.
        """
        if not (0.0 <= p_repeat <= 1.0):
            raise ValueError("p_repeat debe ser un valor flotante en el rango [0.0, 1.0].")

        self.optimal_pilot = optimal_pilot
        self.p_repeat = float(p_repeat)
        self._rng = np.random.default_rng(seed)
        self.last_action: np.ndarray | None = None

    def reset(self):
        """
        Reinicia la memoria del piloto (olvida la última acción registrada).
        """
        self.last_action = None

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
        Devuelve la acción a ejecutar. Si ocurre retardo, repite la acción anterior;
        de lo contrario, consulta al piloto óptimo y actualiza el historial.

        Args:
            observation: Vector de observaciones del entorno.

        Returns:
            np.ndarray: Vector continuo de acción en [-1.0, 1.0].
        """
        if self.last_action is not None and (self._rng.random() < self.p_repeat):
            return self.last_action.copy()

        action = self._get_base_action(observation)
        self.last_action = action.copy()
        return action

    def choose_action(self, observation: np.ndarray) -> tuple[np.ndarray, list]:
        """
        Alias compatible con la interfaz histórica de la tesis (Diedrichs, 2025).

        Returns:
            tuple: (acción seleccionada, lista de estados auxiliares vacía).
        """
        return self.get_action(observation), []
