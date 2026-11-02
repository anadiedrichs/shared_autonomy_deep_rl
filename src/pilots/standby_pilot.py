# -*- coding: utf-8 -*-
"""
Synthetic pilot with standby / failsafe hover mode (StandByPilot).

Simulates teleoperation packet loss or intermittent communication link failure.
With probability 'p_drop' (or 'p_repeat'), the command for the current step is dropped
and a neutral action (zeros) is emitted instead, commanding the low-level PID controller
to perform stable hovering (zero velocity request, fixed altitude hold).

Note: This pilot preserves the behavior evaluated in Section 4.4.4 of the TFM thesis
(Table 4.4 / Diedrichs, 2026), where command omissions produce kinematic starvation
without structural collisions.
"""

import numpy as np

from src.pilots.base_pilot import BasePilot


class StandByPilot(BasePilot):
    """
    Synthetic pilot that occasionally drops its command, emitting a neutral action (hovering).
    """

    def __init__(
        self,
        optimal_pilot,
        p_drop: float = 0.05,
        p_repeat: float | None = None,
        seed: int | None = None,
    ):
        """
        Initializes the StandByPilot.

        Args:
            optimal_pilot: Instance of the base or optimal pilot.
            p_drop: Probability of dropping the command at each step (between 0.0 and 1.0).
            p_repeat: Alias for p_drop for backwards compatibility.
            seed: Seed for the independent random number generator.
        """
        prob = p_repeat if p_repeat is not None else p_drop
        if not (0.0 <= prob <= 1.0):
            raise ValueError("p_drop/p_repeat debe ser un valor flotante en el rango [0.0, 1.0].")

        self.optimal_pilot = optimal_pilot
        self.p_drop = float(prob)
        self.p_repeat = self.p_drop  # Compatibility alias
        self._rng = np.random.default_rng(seed)
        self.last_action: np.ndarray = np.zeros(4, dtype=np.float32)

    def reset(self, seed: int | None = None):
        """
        Resets the pilot's memory and re-seeds the random generator if a seed is provided.
        """
        self.last_action = np.zeros(4, dtype=np.float32)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if hasattr(self.optimal_pilot, "reset"):
            try:
                self.optimal_pilot.reset(seed=seed)
            except TypeError:
                self.optimal_pilot.reset()

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
        Returns the action to execute. If a command drop occurs, a neutral action (zeros)
        is returned; otherwise, queries the optimal pilot.

        Args:
            observation: Environment observation vector.

        Returns:
            np.ndarray: Continuous action vector in [-1.0, 1.0].
        """
        if self._rng.random() < self.p_drop:
            action = np.zeros(4, dtype=np.float32)
        else:
            action = self._get_base_action(observation, **kwargs)

        self.last_action = action.copy()
        return action

    def choose_action(self, observation: np.ndarray) -> tuple[np.ndarray, list]:
        """
        Compatibility alias for the historical thesis interface.

        Returns:
            tuple: (selected action, empty list of auxiliary states).
        """
        return self.get_action(observation), []
