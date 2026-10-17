# -*- coding: utf-8 -*-
"""
Abstract base class for all pilots (learned, synthetic, and heuristic).

Defines the unified interface contract for continuous control of the Crazyflie quadrotor,
maintaining backward compatibility with the historical thesis API (Diedrichs, 2025).
"""

from abc import ABC, abstractmethod
import numpy as np


class BasePilot(ABC):
    """
    Abstract base class defining the common continuous piloting interface.
    """

    @abstractmethod
    def get_action(self, observation: np.ndarray, **kwargs) -> np.ndarray:
        """
        Computes and returns the continuous action vector in [-1.0, 1.0]^4.

        Args:
            observation: Environment observation vector.
            **kwargs: Additional optional parameters (e.g., environment reference).

        Returns:
            np.ndarray: 4-dimensional continuous action vector [vx, vy, yaw_rate, vz]
                bounded in [-1.0, 1.0].
        """
        raise NotImplementedError

    def choose_action(self, observation: np.ndarray, **kwargs) -> tuple[np.ndarray, list]:
        """
        Compatibility alias for the historical thesis interface (Diedrichs, 2025).

        Args:
            observation: Environment observation vector.
            **kwargs: Optional parameters.

        Returns:
            tuple: (selected continuous action, empty list of auxiliary states).
        """
        return self.get_action(observation, **kwargs), []

    def reset(self) -> None:
        """
        Resets the internal state of the pilot (for pilots with memory or temporal delays).
        Performs no action by default.
        """
        pass

    def __call__(self, observation: np.ndarray, **kwargs) -> np.ndarray:
        """
        Allows invoking the pilot instance directly as a callable: pilot(obs).
        """
        return self.get_action(observation, **kwargs)


# Compatibility alias for the 2025 thesis class name (drone-deep-rl)
Pilot = BasePilot
