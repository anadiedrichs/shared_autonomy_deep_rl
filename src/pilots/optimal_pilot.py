# -*- coding: utf-8 -*-
"""
Optimal pilot with a unified continuous inference interface.

Allows loading and executing policies trained with different backends
(Stable-Baselines3 PPO or RLtools SAC) under the same common control interface.
"""

import os
import numpy as np

from src.pilots.base_pilot import BasePilot


class OptimalPilot(BasePilot):
    """
    Unified interface for the optimal pilot, abstracting away the training framework.
    """

    SUPPORTED_BACKENDS = ("sb3_ppo", "rltools_sac", "dummy", "heuristic")

    def __init__(
        self,
        backend: str = "sb3_ppo",
        checkpoint_path: str | None = None,
        model_instance: object | None = None,
        deterministic: bool = True,
        **kwargs,
    ):
        """
        Initializes the optimal pilot.

        Args:
            backend: Model framework ("sb3_ppo", "rltools_sac", "dummy", or "heuristic").
            checkpoint_path: Path to the checkpoint file on disk (.zip, .pt, etc.).
            model_instance: Optional pre-loaded model instance.
            deterministic: If True, executes the policy in deterministic mode without exploration.
            **kwargs: Additional arguments (e.g., for HeuristicPilot).
        """
        backend_lower = backend.lower()
        if backend_lower not in self.SUPPORTED_BACKENDS:
            raise ValueError(
                f"Backend '{backend}' no soportado. Opciones válidas: {self.SUPPORTED_BACKENDS}"
            )

        self.backend = backend_lower
        self.checkpoint_path = checkpoint_path
        self.deterministic = deterministic
        self._model = model_instance
        self._kwargs = kwargs

        if self.backend == "heuristic":
            from src.pilots.heuristic_pilot import HeuristicPilot
            self._model = HeuristicPilot(**kwargs)
        elif self._model is None and checkpoint_path is not None:
            self._load_checkpoint()

    def _load_checkpoint(self):
        """
        Loads the checkpoint from disk according to the selected backend.
        """
        # Smart resolution of relative paths if not found directly from CWD
        if self.checkpoint_path is not None and not os.path.exists(self.checkpoint_path):
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            candidate = os.path.normpath(os.path.join(project_root, self.checkpoint_path.lstrip("./")))
            if os.path.exists(candidate):
                self.checkpoint_path = candidate
            else:
                candidate_models = os.path.join(project_root, "models", os.path.basename(self.checkpoint_path))
                if os.path.exists(candidate_models):
                    self.checkpoint_path = candidate_models

        if self.checkpoint_path is None or not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(
                f"El archivo de checkpoint no fue encontrado en: {self.checkpoint_path}"
            )

        if self.backend == "sb3_ppo":
            try:
                from stable_baselines3 import PPO
                self._model = PPO.load(self.checkpoint_path)
            except ImportError:
                raise ImportError(
                    "stable-baselines3 no está instalado en el entorno actual. "
                    "Instálalo con 'pip install stable-baselines3'."
                )
            except Exception as e:
                raise RuntimeError(
                    f"Error al cargar el checkpoint SB3-PPO desde {self.checkpoint_path}: {e}"
                )

        elif self.backend == "rltools_sac":
            # Support for RLtools models and checkpoints (exported .h files)
            try:
                import rltools
                if (
                    hasattr(rltools, "load_checkpoint_from_path")
                    and self.checkpoint_path is not None
                    and self.checkpoint_path.endswith((".h", ".hpp", ".c"))
                ):
                    self._model = rltools.load_checkpoint_from_path(self.checkpoint_path)
                elif hasattr(rltools, "load"):
                    self._model = rltools.load(self.checkpoint_path)
                else:
                    self._model = self.checkpoint_path
            except ImportError:
                raise ImportError(
                    "rltools no está instalado en el entorno actual. "
                    "Instálalo con 'pip install rltools'."
                )

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """
        Computes the optimal pilot action for the current observation.

        Args:
            observation: 11-dimensional continuous observation vector in [-1.0, 1.0].

        Returns:
            np.ndarray: 4-component continuous action vector bounded in [-1.0, 1.0].
        """
        obs = np.asarray(observation, dtype=np.float32)

        if self.backend == "dummy":
            # If no model is provided, emit a neutral hover command
            if self._model is not None and callable(self._model):
                raw_act = self._model(obs)
            else:
                raw_act = np.zeros(4, dtype=np.float32)

        elif self.backend == "sb3_ppo":
            if self._model is None:
                raise RuntimeError("El modelo SB3-PPO no ha sido cargado.")
            # stable-baselines3 predict
            raw_act, _ = self._model.predict(obs, deterministic=self.deterministic)

        elif self.backend == "rltools_sac":
            if self._model is None:
                raise RuntimeError("El modelo RLtools-SAC no ha sido cargado.")
            if hasattr(self._model, "evaluate"):
                raw_act = self._model.evaluate(obs)
            elif hasattr(self._model, "action"):
                raw_act = self._model.action(obs)
            elif hasattr(self._model, "predict"):
                raw_act, _ = self._model.predict(obs, deterministic=self.deterministic)
            elif callable(self._model):
                raw_act = self._model(obs)
            else:
                raw_act = np.zeros(4, dtype=np.float32)

        elif self.backend == "heuristic":
            if self._model is None:
                from src.pilots.heuristic_pilot import HeuristicPilot
                self._model = HeuristicPilot(**self._kwargs)
            raw_act = self._model.get_action(obs)

        action = np.clip(np.asarray(raw_act, dtype=np.float32), -1.0, 1.0)
        return action

    def choose_action(self, observation: np.ndarray) -> tuple[np.ndarray, list]:
        """
        Compatibility alias for the thesis API (Diedrichs, 2025).

        Returns:
            tuple: (selected action, empty list of auxiliary states).
        """
        return self.get_action(observation), []
