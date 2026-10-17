# -*- coding: utf-8 -*-
"""
Synthetic pilot with Gaussian noise injection over the continuous action.

Simulates human piloting imprecision or tremor in continuous control commands
by adding a Gaussian perturbation N(0, sigma^2) to the optimal pilot's policy.
"""

import numpy as np

from src.pilots.base_pilot import BasePilot


class NoisyPilot(BasePilot):
    """
    Synthetic pilot that perturbs the optimal pilot's continuous actions with Gaussian noise.
    """

    def __init__(
        self,
        optimal_pilot,
        sigma: float | np.ndarray = 0.1,
        seed: int | None = None,
    ):
        """
        Initializes the NoisyPilot.

        Args:
            optimal_pilot: Instance of the base or optimal pilot (must provide get_action or predict).
            sigma: Standard deviation of the Gaussian noise (scalar or 4-element vector).
            seed: Seed for the independent random number generator.
        """
        self.optimal_pilot = optimal_pilot
        self.sigma = sigma
        self._rng = np.random.default_rng(seed)

    def _get_base_action(self, observation: np.ndarray) -> np.ndarray:
        """
        Retrieves the base action proposed by the optimal pilot according to its available interface.
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
        Computes the perturbed action by adding Gaussian noise and clipping to [-1.0, 1.0].

        Args:
            observation: Environment observation vector.

        Returns:
            np.ndarray: 4-dimensional action vector in [-1.0, 1.0].
        """
        base_action = self._get_base_action(observation)
        noise = self._rng.normal(0.0, self.sigma, size=base_action.shape).astype(np.float32)
        noisy_action = np.clip(base_action + noise, -1.0, 1.0)
        return noisy_action

    def choose_action(self, observation: np.ndarray) -> tuple[np.ndarray, list]:
        """
        Compatibility alias for the historical thesis interface (Diedrichs, 2025).

        Returns:
            tuple: (selected action, empty list of auxiliary states).
        """
        return self.get_action(observation), []
