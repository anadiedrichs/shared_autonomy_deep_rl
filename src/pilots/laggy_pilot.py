# -*- coding: utf-8 -*-
"""
Synthetic pilot with reaction delay and zero-order hold (LaggyPilot).

Simulates human reaction time, transmission latency, and low-frequency neuromuscular
control following the methodology of Schaff & Walter (2020) [Residual Policy Learning
for Shared Autonomy, RSS 2020] and Diedrichs (2025):

At each simulation step:
- With probability 'p_repeat' (p_lag in Schaff & Walter, 2020), the operator fails to
  update the command, maintaining and repeating the previous continuous action (a_t = a_{t-1}).
  This simulates zero-order hold (ZOH) and human reaction latency, causing velocity
  and inertia accumulation when maneuvers or braking are urgently required.
- With probability (1 - p_repeat), the operator successfully observes the current state
  and computes an updated action from the base policy (a_t = pi*(s_t)).

Optionally, a deterministic FIFO pipeline delay can be configured via 'delay_steps' (k steps),
such that a_t = pi*(s_{t-k}), emulating pure transport latency.
"""

from collections import deque
import numpy as np

from src.pilots.base_pilot import BasePilot


class LaggyPilot(BasePilot):
    """
    Synthetic human pilot exhibiting reaction delay and command repetition (Schaff & Walter, 2020).
    """

    def __init__(
        self,
        optimal_pilot,
        p_repeat: float = 0.2,
        delay_steps: int = 0,
        seed: int | None = None,
    ):
        """
        Initializes the LaggyPilot.

        Args:
            optimal_pilot: Instance of the base or optimal pilot.
            p_repeat: Probability of repeating the previous continuous action at each step
                (between 0.0 and 1.0), as defined by Schaff & Walter (2020).
            delay_steps: Number of simulation steps for a deterministic FIFO pipeline delay.
                If 0 (default), stochastic repetition (ZOH) governed by p_repeat is used.
            seed: Seed for the independent random number generator.
        """
        if not (0.0 <= p_repeat <= 1.0):
            raise ValueError("p_repeat debe ser un valor flotante en el rango [0.0, 1.0].")
        if delay_steps < 0:
            raise ValueError("delay_steps debe ser un entero no negativo.")

        self.optimal_pilot = optimal_pilot
        self.p_repeat = float(p_repeat)
        self.delay_steps = int(delay_steps)
        self._rng = np.random.default_rng(seed)
        self.last_action: np.ndarray | None = None
        self._action_queue: deque[np.ndarray] = deque(maxlen=max(1, self.delay_steps))

    def reset(self, seed: int | None = None):
        """
        Resets the pilot's memory and clears action history.
        """
        self.last_action = None
        self._action_queue.clear()
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
        Returns the action to execute.
        
        If delay_steps > 0, enforces a fixed FIFO pipeline delay: commands are buffered
        and emitted k steps later.
        
        Otherwise, applies the Schaff & Walter (2020) stochastic action repetition model:
        repeats the previous continuous action with probability p_repeat; computes a new
        action with probability (1 - p_repeat).
        """
        # Deterministic FIFO delay mode
        if self.delay_steps > 0:
            fresh_action = self._get_base_action(observation, **kwargs)
            self._action_queue.append(fresh_action.copy())
            if len(self._action_queue) < self.delay_steps:
                # Buffer filling phase: emit initial action or zeros
                action = self._action_queue[0].copy()
            else:
                action = self._action_queue.popleft()
            self.last_action = action.copy()
            return action

        # Schaff & Walter (2020) stochastic action repetition (Zero-Order Hold)
        if self.last_action is not None and (self._rng.random() < self.p_repeat):
            action = self.last_action.copy()
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
