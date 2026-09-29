# -*- coding: utf-8 -*-
"""
Piloto óptimo con interfaz unificada de inferencia continua.

Permite cargar y ejecutar políticas entrenadas con diferentes backends
(Stable-Baselines3 PPO o RLtools SAC) bajo la misma interfaz común de control.
"""

import os
import numpy as np


class OptimalPilot:
    """
    Interfaz unificada para el piloto óptimo, abstrayendo el framework de entrenamiento.
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
        Inicializa el piloto óptimo.

        Args:
            backend: Framework del modelo ("sb3_ppo", "rltools_sac", "dummy" o "heuristic").
            checkpoint_path: Ruta al archivo de checkpoint en disco (.zip, .pt, etc.).
            model_instance: Instancia pre-cargada opcional del modelo.
            deterministic: Si es True, ejecuta la política en modo determinista sin exploración.
            **kwargs: Argumentos adicionales (por ejemplo para HeuristicPilot).
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
        Carga el checkpoint desde el disco según el backend seleccionado.
        """
        # Resolución inteligente de rutas relativas si no se encuentra directamente desde CWD
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
            # Soporte para modelos de RLtools
            try:
                import rltools
                # Si RLtools expone función de carga directa en su wrapper
                if hasattr(rltools, "load"):
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
        Calcula la acción del piloto óptimo para la observación actual.

        Args:
            observation: Vector de observaciones continuo de 11 dimensiones en [-1.0, 1.0].

        Returns:
            np.ndarray: Vector de acción continuo de 4 componentes acotado en [-1.0, 1.0].
        """
        obs = np.asarray(observation, dtype=np.float32)

        if self.backend == "dummy":
            # Si no hay modelo, emite un comando neutro de hover
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
            if hasattr(self._model, "predict"):
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
        Alias compatible con la API de la tesis (Diedrichs, 2025).

        Returns:
            tuple: (acción seleccionada, lista de estados auxiliares vacía).
        """
        return self.get_action(observation), []
