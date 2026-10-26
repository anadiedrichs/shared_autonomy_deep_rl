# -*- coding: utf-8 -*-
"""
Synthetic pilot with reaction delay (LaggyPilot).

Simulates human response time, latency, or teleoperation packet loss in continuous
control. With probability 'p_repeat' (alpha parameter from the thesis) the command
for the current step is dropped and a neutral action (zeros) is emitted instead,
so the drone receives no motion command during that cycle.

Note: in the 2025 discrete formulation the delay repeated the previous action. In
continuous control with a smoothness regularizer the optimal policy already satisfies
a_t ~= a_{t-1}, so repeating the previous action has no measurable effect. Dropping
the command makes the effective progress scale with (1 - p_repeat).
"""

import numpy as np

from src.pilots.base_pilot import BasePilot


class LaggyPilot(BasePilot):
    """
    Synthetic pilot that occasionally drops its command, emitting a neutral action.
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
            p_repeat: Probability of dropping the command at each step (between 0.0 and 1.0).
                The name is kept for backward compatibility with the CLI and thesis notation.
            seed: Seed for the independent random number generator.
        """
        if not (0.0 <= p_repeat <= 1.0):
            raise ValueError("p_repeat debe ser un valor flotante en el rango [0.0, 1.0].")

        self.optimal_pilot = optimal_pilot
        self.p_repeat = float(p_repeat)
        self._rng = np.random.default_rng(seed)
        self.last_action: np.ndarray = np.zeros(4, dtype=np.float32)

    def reset(self):
        """
        Resets the pilot's memory: the drone starts from a neutral (zero) command.
        """
        self.last_action = np.zeros(4, dtype=np.float32)

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
        Returns the action to execute. If a delay occurs, the command is dropped and a
        neutral action (zeros) is returned without querying the optimal pilot; otherwise,
        queries the optimal pilot.

        Args:
            observation: Environment observation vector.

        Returns:
            np.ndarray: Continuous action vector in [-1.0, 1.0].
        """
        if self._rng.random() < self.p_repeat:
            action = np.zeros(4, dtype=np.float32)
        else:
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
