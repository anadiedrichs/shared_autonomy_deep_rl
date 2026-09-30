# -*- coding: utf-8 -*-
"""
Política residual del copiloto: pi_r(s, a_h) -> a_r y piloto asistido compuesto.

Implementa el modelo de asistencia por corrección aditiva según Schaff & Walter (2020):
    a = clip(a_h + a_r, -1.0, 1.0)
    a_h ~ pi_h(s)
    a_r ~ pi_r(s, a_h)
"""

import os
import numpy as np

from src.pilots.base_pilot import BasePilot


class ResidualPolicy:
    """
    Interfaz de inferencia del copiloto: observación + acción del piloto -> a_r.
    Soporta checkpoints de Stable-Baselines3 (.zip), RLtools C header (.h), o modelos en memoria / dummy.
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
        Inicializa la política residual.

        Args:
            checkpoint_path: Ruta al archivo del modelo entrenado (.zip para SB3, .h para RLtools).
            model: Objeto de modelo o función invocable en memoria (opcional).
            backend: Motor de inferencia ("auto", "sb3_ppo", "sb3_sac", "rltools_sac", "dummy").
            residual_scale: Factor de escala / atenuación en [0.0, 1.0] para limitar la autoridad del copiloto.
            deterministic: True para inferencia determinista, False para estocástica.
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
        Carga el modelo entrenado desde el disco.

        Args:
            checkpoint_path: Ruta al archivo del checkpoint.
            backend: Motor de inferencia específico (opcional).
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
                    # Intento cruzado en caso de nombre no estándar
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
        Calcula la acción residual a_r en [-1.0, 1.0]^4.

        Args:
            observation: Vector de observaciones (11 dimensiones base o 15 extendida).
            pilot_action: Vector de acción propuesto por el piloto (4 dimensiones). Si se provee
                y la observación tiene 11 dimensiones, se concatenan formando [obs, a_h] (15 dimensiones).

        Returns:
            np.ndarray: Vector continuo de acción residual a_r en [-1.0, 1.0]^4.
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
        """Reinicia el estado interno del copiloto (para modelos con estado o filtros)."""
        pass

    def __call__(
        self,
        observation: np.ndarray,
        pilot_action: np.ndarray | None = None,
    ) -> np.ndarray:
        return self.get_action(observation, pilot_action)


class AssistedPilot(BasePilot):
    """
    Compone un piloto (BasePilot) con una política residual (ResidualPolicy).
    Cumple con el contrato BasePilot para poder ser evaluado de forma transparente en la suite de métricas.
    """

    def __init__(self, pilot: BasePilot, copilot: ResidualPolicy):
        """
        Args:
            pilot: Instancia de BasePilot (HeuristicPilot, OptimalPilot, NoisyPilot, etc.).
            copilot: Instancia de ResidualPolicy.
        """
        self.pilot = pilot
        self.copilot = copilot

    def get_action(self, observation: np.ndarray, **kwargs) -> np.ndarray:
        """
        Calcula la acción combinada a = clip(a_h + a_r, -1.0, 1.0).
        """
        a_h = self.pilot.get_action(observation, **kwargs)
        a_r = self.copilot.get_action(observation, a_h)
        return np.clip(a_h + a_r, -1.0, 1.0)

    def reset(self) -> None:
        if hasattr(self.pilot, "reset"):
            self.pilot.reset()
        if hasattr(self.copilot, "reset"):
            self.copilot.reset()
