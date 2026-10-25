# -*- coding: utf-8 -*-
"""
Synthetic pilot with reaction delay (LaggyPilot).

Simulates human response time or latency by repeating the action executed in the
previous step with probability 'p_repeat' (alpha parameter from the thesis).
"""

import numpy as np

from src.pilots.base_pilot import BasePilot


class LaggyPilot(BasePilot):
    """
    Synthetic pilot that introduces temporal dependency by occasionally repeating the previous action.
    """

    def __init__(
        self,
        optimal_pilot,
        p_repeat: float = 0.2,
        seed: int | None = None,
    ):
        """
        Initializes the LaggyPilot.

        Args:
            optimal_pilot: Instance of the base or optimal pilot.
            p_repeat: Probability of repeating the last continuous action (between 0.0 and 1.0).
            seed: Seed for the independent random number generator.
        """
        if not (0.0 <= p_repeat <= 1.0):
            raise ValueError("p_repeat debe ser un valor flotante en el rango [0.0, 1.0].")

        self.optimal_pilot = optimal_pilot
        self.p_repeat = float(p_repeat)
        self._rng = np.random.default_rng(seed)
        self.last_action: np.ndarray | None = None

    def reset(self):
        """
        Resets the pilot's memory (forgets the last recorded action).
        """
        self.last_action = None

    def _get_base_action(self, observation: np.ndarray, **kwargs) -> np.ndarray:
        """
        Retrieves the base action proposed by the optimal pilot according to its available interface.
        """
        if hasattr(self.optimal_pilot, "get_action"):
            try:
                act = self.optimal_pilot.get_action(observation, **kwargs)
            except TypeError:
                act = self.optimal_pilot.get_action(observation)
        elif hasattr(self.optimal_pilot, "predict"):
            act, _ = self.optimal_pilot.predict(observation, deterministic=True)
        elif hasattr(self.optimal_pilot, "choose_action"):
            act_res = self.optimal_pilot.choose_action(observation)
            act = act_res[0] if isinstance(act_res, tuple) else act_res
        else:
            raise TypeError("El objeto optimal_pilot no expone get_action, predict ni choose_action.")

        return np.asarray(act, dtype=np.float32)

    def get_action(self, observation: np.ndarray, **kwargs) -> np.ndarray:
        """
        Returns the action to execute. If a delay occurs, repeats the previous action;
        otherwise, queries the optimal pilot and updates the history.

        Args:
            observation: Environment observation vector.

        Returns:
            np.ndarray: Continuous action vector in [-1.0, 1.0].
        """
        if self.last_action is not None and (self._rng.random() < self.p_repeat):
            return self.last_action.copy()

        action = self._get_base_action(observation, **kwargs)
        self.last_action = action.copy()
        return action

    def choose_action(self, observation: np.ndarray) -> tuple[np.ndarray, list]:
        """
        Compatibility alias for the historical thesis interface (Diedrichs, 2025).

        Returns:
            tuple: (selected action, empty list of auxiliary states).
        """
        return self.get_action(observation), []
