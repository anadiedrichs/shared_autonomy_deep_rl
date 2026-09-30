# -*- coding: utf-8 -*-
"""
Entorno de autonomía compartida para teleoperación asistida de cuadricópteros Crazyflie.

Compone un piloto (humano o sintético) y un copiloto residual siguiendo la formulación
de Schaff & Walter (2020):
    a = clip(a_h + a_r, -1.0, 1.0)
    a_h ~ pi_h(s)      # orientado a la meta
    a_r ~ pi_r(s, a_h) # corrección residual agnóstica a la meta

Funciona tanto en modo de entrenamiento de la política residual (recibiendo a_r en step())
como en modo de evaluación/inferencia autónoma (con un copiloto ResidualPolicy provisto).
"""

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    gym = None
    from src.envs.drone_robot_supervisor import spaces

from src.envs.corner_env_continuous import CornerEnvContinuous
from src.rewards.reward_general_safety import general_safety_reward


class SharedAutonomyEnv(CornerEnvContinuous):
    """
    Envuelve CornerEnvContinuous para implementar autonomía compartida continua.
    """

    def __init__(
        self,
        pilot,
        copilot=None,
        max_episode_steps: int = 1_000,
        corner_name: str = "cone_1",
        reward_mode: str = "safety",
        lambda_intervention: float = 0.10,
        residual_scale: float = 1.0,
        include_pilot_action_in_obs: bool = True,
        enable_trajectory_logging: bool = False,
        verbose: int = 1,
    ):
        """
        Inicializa el entorno de autonomía compartida.

        Args:
            pilot: Objeto piloto con método get_action(observation, **kwargs) -> np.ndarray(4,).
            copilot: Objeto copiloto opcional con método get_action(observation, pilot_action) -> np.ndarray(4,).
            max_episode_steps: Límite máximo de pasos por episodio.
            corner_name: Nombre del cono objetivo en Webots.
            reward_mode: Función de recompensa utilizada ("safety", "goal_directed", "combined").
            lambda_intervention: Coeficiente de penalización por intervención ||a_r||^2.
            residual_scale: Factor de atenuación para el residual (en (0, 1]).
            include_pilot_action_in_obs: Si True, el espacio de observación para el copiloto
                es de 15 dimensiones ([obs_dron(11), acción_piloto(4)]).
            enable_trajectory_logging: Guardar trayectoria física en disco.
            verbose: Nivel de detalle en consola.
        """
        self.pilot = pilot
        self.copilot = copilot
        self.reward_mode = str(reward_mode).lower()
        self.lambda_intervention = float(lambda_intervention)
        self.residual_scale = float(residual_scale)
        self.include_pilot_action_in_obs = bool(include_pilot_action_in_obs)

        # Variables internas de trazabilidad del paso actual
        self.current_drone_obs = np.zeros(11, dtype=np.float32)
        self.current_pilot_action = np.zeros(4, dtype=np.float32)
        self.current_residual_action = np.zeros(4, dtype=np.float32)
        self.current_composed_action = np.zeros(4, dtype=np.float32)
        self.last_safety_reward = 0.0
        self.last_goal_reward = 0.0

        super().__init__(
            max_episode_steps=max_episode_steps,
            corner_name=corner_name,
            enable_trajectory_logging=enable_trajectory_logging,
            verbose=verbose,
        )

        # Redefinir espacios formales de Gymnasium
        obs_dim = 15 if self.include_pilot_action_in_obs else 11
        if spaces is not None:
            self.observation_space = spaces.Box(
                low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
            )
            # El espacio de acción para el agente RL que aprende el copiloto es el residual a_r
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(4,), dtype=np.float32
            )

    def _get_pilot_action(self, drone_obs: np.ndarray) -> np.ndarray:
        """Obtiene la acción propuesta por el piloto para la observación física actual."""
        if self.pilot is None:
            return np.zeros(4, dtype=np.float32)
        try:
            act = self.pilot.get_action(drone_obs, env=self)
        except TypeError:
            act = self.pilot.get_action(drone_obs)
        return np.clip(np.asarray(act, dtype=np.float32), -1.0, 1.0)

    def _build_copilot_observation(
        self, drone_obs: np.ndarray, pilot_action: np.ndarray
    ) -> np.ndarray:
        """Construye la observación extendida para el copiloto [obs, a_h]."""
        if self.include_pilot_action_in_obs:
            return np.concatenate([drone_obs, pilot_action], dtype=np.float32)
        return np.asarray(drone_obs, dtype=np.float32)

    def reset(self, seed: int | None = None, options: dict | None = None):
        """
        Reinicia el entorno, el piloto y el copiloto, y devuelve la observación inicial para el copiloto.
        """
        drone_obs, info = super().reset(seed=seed, options=options)
        self.current_drone_obs = drone_obs

        # Reiniciar estados internos de pilotos si implementan memoria
        if hasattr(self.pilot, "reset"):
            self.pilot.reset()
        if self.copilot is not None and hasattr(self.copilot, "reset"):
            self.copilot.reset()

        # Computar acción inicial del piloto
        self.current_pilot_action = self._get_pilot_action(drone_obs)
        self.current_residual_action = np.zeros(4, dtype=np.float32)
        self.current_composed_action = np.clip(self.current_pilot_action, -1.0, 1.0)
        self.last_safety_reward = 0.0
        self.last_goal_reward = 0.0

        copilot_obs = self._build_copilot_observation(drone_obs, self.current_pilot_action)
        info.update(self._get_shared_autonomy_info())
        return copilot_obs, info

    def step(self, action: np.ndarray | None = None):
        """
        Ejecuta un paso en el entorno con autonomía compartida.

        Args:
            action: Si se proporciona, se interpreta como la acción residual a_r calculada por el
                algoritmo RL. Si es None, se consulta a self.copilot si está presente, o se asume 0.

        Returns:
            tuple: (copilot_obs, reward, terminated, truncated, info)
        """
        if action is None:
            if self.copilot is not None:
                a_r = self.copilot.get_action(self.current_drone_obs, self.current_pilot_action)
            else:
                a_r = np.zeros(4, dtype=np.float32)
        else:
            a_r = np.asarray(action, dtype=np.float32)

        # Aplicar factor de escala y acotamiento a la acción residual
        a_r = np.clip(a_r * self.residual_scale, -1.0, 1.0)
        self.current_residual_action = a_r

        # Composición aditiva: a = clip(a_h + a_r, -1, 1)
        a_composed = np.clip(self.current_pilot_action + a_r, -1.0, 1.0)
        self.current_composed_action = a_composed

        # Ejecución del paso físico subyacente en Webots
        drone_obs, goal_reward, terminated, truncated, info = super().step(a_composed)
        self.current_drone_obs = drone_obs
        self.last_goal_reward = float(goal_reward)

        # Cálculo de recompensa de seguridad general (goal-agnostic)
        d_min = (
            min(self.dist_front, self.dist_back, self.dist_right, self.dist_left) / 1000.0
        )
        collided = bool(
            self.alt < 0.10
            or self.is_out_of_bounds()
            or (self.episode_score <= self.MIN_EPISODE_SCORE)
            or (self.episode_score >= self.MAX_EPISODE_SCORE)
        )
        safety_reward = general_safety_reward(
            distance_to_obstacle_m=d_min,
            residual_action=a_r,
            collided=collided,
            lambda_intervention=self.lambda_intervention,
        )
        self.last_safety_reward = float(safety_reward)

        # Selección de la señal de recompensa según el modo configurado
        if self.reward_mode == "safety":
            step_reward = safety_reward
        elif self.reward_mode == "goal_directed":
            step_reward = goal_reward
        elif self.reward_mode == "combined":
            step_reward = goal_reward + safety_reward
        else:
            step_reward = safety_reward

        # Próxima acción del piloto para el siguiente paso y construcción de la nueva observación
        next_pilot_action = self._get_pilot_action(drone_obs)
        self.current_pilot_action = next_pilot_action
        next_copilot_obs = self._build_copilot_observation(drone_obs, next_pilot_action)

        info.update(self._get_shared_autonomy_info())
        return next_copilot_obs, step_reward, terminated, truncated, info

    def _get_shared_autonomy_info(self) -> dict:
        """Devuelve el diccionario con métricas del sistema de autonomía compartida."""
        return {
            "pilot_action": self.current_pilot_action.copy(),
            "residual_action": self.current_residual_action.copy(),
            "composed_action": self.current_composed_action.copy(),
            "residual_norm": float(np.linalg.norm(self.current_residual_action)),
            "safety_reward": float(self.last_safety_reward),
            "goal_reward": float(self.last_goal_reward),
        }
