# -*- coding: utf-8 -*-
"""
Copilot residual policy: pi_r(s, a_h) -> a_r and composite assisted pilot.

Implements the additive correction assistance model according to Schaff & Walter (2020):
    a = clip(a_h + a_r, -1.0, 1.0)
    a_h ~ pi_h(s)
    a_r ~ pi_r(s, a_h)
"""

import os
import numpy as np

from src.pilots.base_pilot import BasePilot


class ResidualPolicy:
    """
    Copilot inference interface: observation + pilot action -> a_r.
    Supports Stable-Baselines3 (.zip) checkpoints, RLtools C headers (.h), or in-memory / dummy models.
    """

    SUPPORTED_BACKENDS = ("auto", "sb3_ppo", "sb3_sac", "rltools_sac", "dummy")

    def __init__(
        self,
        checkpoint_path: str | None = None,
        model: object | None = None,
        backend: str = "auto",
        residual_scale: float = 1.0,
        deterministic: bool = True,
    ):
        """
        Initializes the residual policy.

        Args:
            checkpoint_path: Path to the trained model file (.zip for SB3, .h for RLtools).
            model: In-memory model object or callable function (optional).
            backend: Inference engine ("auto", "sb3_ppo", "sb3_sac", "rltools_sac", "dummy").
            residual_scale: Scale / attenuation factor in [0.0, 1.0] to limit copilot authority.
            deterministic: True for deterministic inference, False for stochastic.
        """
        self.checkpoint_path = checkpoint_path
        self._model = model
        self.backend = backend
        self.residual_scale = float(residual_scale)
        self.deterministic = bool(deterministic)

        if self.backend == "auto":
            self.backend = self._detect_backend(self.checkpoint_path)

        if self.checkpoint_path is not None and self._model is None:
            self.load(self.checkpoint_path, self.backend)

    @staticmethod
    def _detect_backend(checkpoint_path: str | None) -> str:
        if checkpoint_path is None:
            return "dummy"
        ext = os.path.splitext(checkpoint_path)[1].lower()
        if ext in (".h", ".hpp", ".c"):
            return "rltools_sac"
        if ext == ".zip":
            base = os.path.basename(checkpoint_path).lower()
            if "sac" in base:
                return "sb3_sac"
            return "sb3_ppo"
        return "dummy"

    def load(self, checkpoint_path: str, backend: str | None = None) -> None:
        """
        Loads the trained model from disk.

        Args:
            checkpoint_path: Path to the checkpoint file.
            backend: Specific inference engine (optional).
        """
        self.checkpoint_path = checkpoint_path
        if backend is not None and backend != "auto":
            self.backend = backend
        elif self.backend in ("auto", "dummy"):
            self.backend = self._detect_backend(checkpoint_path)

        if self.backend in ("sb3_ppo", "sb3_sac"):
            try:
                from stable_baselines3 import PPO, SAC

                target_cls = SAC if self.backend == "sb3_sac" else PPO
                try:
                    self._model = target_cls.load(self.checkpoint_path)
                except Exception:
                    # Fallback cross-attempt in case of non-standard naming
                    alt_cls = PPO if self.backend == "sb3_sac" else SAC
                    self._model = alt_cls.load(self.checkpoint_path)
                    self.backend = "sb3_ppo" if alt_cls is PPO else "sb3_sac"
            except ImportError:
                raise ImportError(
                    "stable-baselines3 no está disponible. Instálalo con 'pip install stable-baselines3'."
                )
            except Exception as e:
                raise RuntimeError(
                    f"Error al cargar checkpoint SB3 desde {self.checkpoint_path}: {e}"
                )

        elif self.backend == "rltools_sac":
            try:
                import rltools

                if hasattr(rltools, "load_checkpoint_from_path"):
                    self._model = rltools.load_checkpoint_from_path(self.checkpoint_path)
                elif hasattr(rltools, "load"):
                    self._model = rltools.load(self.checkpoint_path)
                else:
                    self._model = self.checkpoint_path
            except ImportError:
                raise ImportError("rltools no está disponible para cargar header C++.")
            except Exception as e:
                raise RuntimeError(
                    f"Error al cargar checkpoint RLtools desde {self.checkpoint_path}: {e}"
                )

    def get_action(
        self,
        observation: np.ndarray,
        pilot_action: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Computes the residual action a_r in [-1.0, 1.0]^4.

        Args:
            observation: Observation vector (11-dimensional base or 15-dimensional extended).
            pilot_action: Action vector proposed by the pilot (4 dimensions). If provided
                and the observation has 11 dimensions, they are concatenated into [obs, a_h] (15 dimensions).

        Returns:
            np.ndarray: Continuous residual action vector a_r in [-1.0, 1.0]^4.
        """
        obs = np.asarray(observation, dtype=np.float32).flatten()
        if pilot_action is not None:
            p_act = np.asarray(pilot_action, dtype=np.float32).flatten()
            if obs.shape[0] == 11:
                copilot_obs = np.concatenate([obs, p_act], dtype=np.float32)
            else:
                copilot_obs = obs
        else:
            copilot_obs = obs

        if self.backend == "dummy":
            if self._model is not None and callable(self._model):
                raw_act = self._model(copilot_obs)
            else:
                raw_act = np.zeros(4, dtype=np.float32)

        elif self.backend in ("sb3_ppo", "sb3_sac"):
            if self._model is None:
                raise RuntimeError("El modelo SB3 no ha sido cargado.")
            raw_act, _ = self._model.predict(copilot_obs, deterministic=self.deterministic)

        elif self.backend == "rltools_sac":
            if self._model is None:
                raise RuntimeError("El modelo RLtools no ha sido cargado.")
            if hasattr(self._model, "evaluate"):
                raw_act = self._model.evaluate(copilot_obs)
            elif hasattr(self._model, "action"):
                raw_act = self._model.action(copilot_obs)
            elif hasattr(self._model, "predict"):
                raw_act, _ = self._model.predict(copilot_obs, deterministic=self.deterministic)
            else:
                raw_act = np.zeros(4, dtype=np.float32)

        else:
            raw_act = np.zeros(4, dtype=np.float32)

        raw_act = np.asarray(raw_act, dtype=np.float32)
        scaled_act = raw_act * self.residual_scale
        return np.clip(scaled_act, -1.0, 1.0)

    def reset(self) -> None:
        """Resets the internal state of the copilot (for stateful models or filters)."""
        pass

    def __call__(
        self,
        observation: np.ndarray,
        pilot_action: np.ndarray | None = None,
    ) -> np.ndarray:
        return self.get_action(observation, pilot_action)


class AssistedPilot(BasePilot):
    """
    Composes a pilot (BasePilot) with a residual policy (ResidualPolicy).
    Complies with the BasePilot contract for transparent evaluation in the metrics suite.
    """

    def __init__(self, pilot: BasePilot, copilot: ResidualPolicy):
        """
        Args:
            pilot: BasePilot instance (HeuristicPilot, OptimalPilot, NoisyPilot, etc.).
            copilot: ResidualPolicy instance.
        """
        self.pilot = pilot
        self.copilot = copilot

    def get_action(self, observation: np.ndarray, **kwargs) -> np.ndarray:
        """
        Computes the combined action a = clip(a_h + a_r, -1.0, 1.0).
        """
        a_h = self.pilot.get_action(observation, **kwargs)
        a_r = self.copilot.get_action(observation, a_h)
        return np.clip(a_h + a_r, -1.0, 1.0)

    def reset(self) -> None:
        if hasattr(self.pilot, "reset"):
            self.pilot.reset()
        if hasattr(self.copilot, "reset"):
            self.copilot.reset()
