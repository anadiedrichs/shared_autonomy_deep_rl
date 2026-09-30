# -*- coding: utf-8 -*-
"""
Clase base abstracta para todos los pilotos (aprendidos, sintéticos y heurísticos).

Define el contrato de interfaz unificada para el control continuo del cuadricóptero Crazyflie,
manteniendo retrocompatibilidad con la API histórica de la tesis (Diedrichs, 2025).
"""

from abc import ABC, abstractmethod
import numpy as np


class BasePilot(ABC):
    """
    Clase base abstracta que define la interfaz común de pilotaje continuo.
    """

    @abstractmethod
    def get_action(self, observation: np.ndarray, **kwargs) -> np.ndarray:
        """
        Calcula y devuelve el vector de acción continuo en [-1.0, 1.0]^4.

        Args:
            observation: Vector de observaciones del entorno.
            **kwargs: Parámetros opcionales adicionales (p.ej. referencia al entorno).

        Returns:
            np.ndarray: Vector continuo de acción de 4 dimensiones [vx, vy, yaw_rate, vz]
                acotado en [-1.0, 1.0].
        """
        raise NotImplementedError

    def choose_action(self, observation: np.ndarray, **kwargs) -> tuple[np.ndarray, list]:
        """
        Alias de compatibilidad con la interfaz histórica de la tesis (Diedrichs, 2025).

        Args:
            observation: Vector de observaciones del entorno.
            **kwargs: Parámetros opcionales.

        Returns:
            tuple: (acción continua seleccionada, lista vacía de estados auxiliares).
        """
        return self.get_action(observation, **kwargs), []

    def reset(self) -> None:
        """
        Reinicia el estado interno del piloto (para pilotos con memoria o retardos temporales).
        Por defecto no realiza ninguna acción.
        """
        pass

    def __call__(self, observation: np.ndarray, **kwargs) -> np.ndarray:
        """
        Permite invocar la instancia del piloto directamente como una función: pilot(obs).
        """
        return self.get_action(observation, **kwargs)


# Alias por compatibilidad con el nombre de clase de la tesis 2025 (drone-deep-rl)
Pilot = BasePilot
