# -*- coding: utf-8 -*-
"""
Entorno continuo del Crazyflie en habitación 2x2m con objetivo en una esquina (CornerEnv).

Port continuo de la clase SimpleCornerEnvRS10 de la tesis (Diedrichs, 2025, drone-deep-rl).
"""

from math import cos, sin, sqrt
import numpy as np

from src.envs.drone_robot_supervisor import DroneRobotSupervisor, spaces
from src.rewards.reward_goal_directed import compute_goal_directed_reward
from src.utils.utilities import normalize_to_range


class CornerEnvContinuous(DroneRobotSupervisor):
    """
    Entorno continuo de navegación hacia un cono objetivo en una esquina en una habitación de 2x2m.
    """

    # Identificador del objetivo y umbrales del escenario
    CORNER_NAME = "cone_1"
    DISTANCE_THRESHOLD = 0.15  # Distancia euclídea mínima al objetivo (15 cm)
    MIN_DIST_OBSTACLES = 100  # Distancia mínima a paredes en mm (100 mm = 0.10 m)
    MIN_EPISODE_SCORE = -1_000.0  # Puntaje mínimo acumulado antes de truncar
    MAX_EPISODE_SCORE = 1_000.0  # Puntaje máximo acumulado antes de truncar

    # Límites espaciales de la habitación (2x2 metros centrada en [0, 0])
    X_MIN, X_MAX = -1.0, 1.0
    Y_MIN, Y_MAX = -1.0, 1.0

    def __init__(
        self,
        max_episode_steps: int = 10_000,
        corner_name: str = "cone_1",
        enable_trajectory_logging: bool = False,
        verbose: int = 1,
    ):
        """
        Inicializa el entorno continuo de la habitación con esquinas.

        Args:
            max_episode_steps: Número máximo de pasos por episodio.
            corner_name: Identificador DEF del nodo objetivo en el mundo Webots (por defecto "cone_1").
            enable_trajectory_logging: Si es True, habilita el guardado en disco de la trayectoria CSV.
            verbose: Nivel de detalle en consola (0: silencioso para RL masivo, 1: transiciones clave y despegue, 2: depuración extendida).
        """
        self.corner_name = corner_name
        self.target_node = None
        self.dist_min_target = 0.0
        self.prev_dist_min_target = 0.0
        self.corner = None
        self.x_init = 0.0
        self.y_init = 0.0

        # Posición por defecto del objetivo para pruebas unitarias sin Webots
        self._target_pos_fallback = np.array([0.8, 0.8, 0.0], dtype=np.float32)

        super().__init__(
            max_episode_steps=max_episode_steps,
            enable_trajectory_logging=enable_trajectory_logging,
            verbose=verbose,
        )

        # Espacio de observaciones extendido (14 dimensiones):
        # 11 sensores base de DroneRobotSupervisor + 3 variables relativas a la meta
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(14,), dtype=np.float32
        )

        self._init_targets()
        self.put_drone_in_the_center()

    def _init_targets(self):
        """
        Obtiene la referencia al nodo del cono objetivo en Webots mediante su identificador DEF.
        """
        if hasattr(self, "getFromDef"):
            self.target_node = self.getFromDef(self.corner_name)

    def put_drone_in_the_center(self):
        """
        Posiciona el dron en el centro del escenario (coordenadas [0, 0, z]).
        """
        self.x_init = 0.0
        self.y_init = 0.0

        if self.robot is not None and hasattr(self.robot, "getField"):
            pos_field = self.robot.getField("translation")
            if pos_field is not None and hasattr(pos_field, "getSFVec3f"):
                current_z = pos_field.getSFVec3f()[2]
                pos_field.setSFVec3f([0.0, 0.0, current_z])

    def is_out_of_bounds(self) -> bool:
        """
        Verifica si el dron ha salido de los límites geométricos de la habitación (2x2 metros).

        Returns:
            bool: True si el dron cruzó alguna pared, False si permanece dentro del recinto.
        """
        return (
            self.x_global < self.X_MIN
            or self.x_global > self.X_MAX
            or self.y_global < self.Y_MIN
            or self.y_global > self.Y_MAX
        )

    def _get_target_pos(self) -> tuple[float, float]:
        """
        Obtiene las coordenadas (x, y) del cono objetivo en Webots.
        Usa la posición de respaldo para pruebas sin simulador activo.
        """
        if self.target_node is not None and hasattr(self.target_node, "getField"):
            target_field = self.target_node.getField("translation")
            if target_field is not None and hasattr(target_field, "getSFVec3f"):
                pos = target_field.getSFVec3f()
                return float(pos[0]), float(pos[1])
        return float(self._target_pos_fallback[0]), float(self._target_pos_fallback[1])

    def get_distance_to_target(self) -> float:
        """
        Calcula la distancia euclídea en el plano horizontal (XY) entre el dron y el objetivo.

        Returns:
            float: Distancia en metros al objetivo.
        """
        tx, ty = self._get_target_pos()
        dx = self.x_global - tx
        dy = self.y_global - ty
        return float(sqrt(dx * dx + dy * dy))

    def get_observations(self) -> np.ndarray:
        """
        Genera el vector de observaciones continuo de 14 dimensiones para CornerEnvContinuous.

        Combina las 11 lecturas base del dron con 3 observaciones relativas a la meta:
            12. Distancia euclídea normalizada al cono objetivo en [-1.0, 1.0].
            13. Desplazamiento longitudinal hacia la meta en el marco del cuerpo del dron (body X / forward) en [-1.0, 1.0].
            14. Desplazamiento lateral hacia la meta en el marco del cuerpo del dron (body Y / sideways) en [-1.0, 1.0].

        Returns:
            np.ndarray: Vector de 14 valores continuos en [-1.0, 1.0].
        """
        base_obs = super().get_observations()

        self.dist_min_target = self.get_distance_to_target()
        dist_norm = normalize_to_range(self.dist_min_target, 0.0, 3.0, -1.0, 1.0, clip=True)

        tx, ty = self._get_target_pos()
        dx_global = tx - self.x_global
        dy_global = ty - self.y_global

        cos_yaw = cos(self.yaw)
        sin_yaw = sin(self.yaw)

        # Proyección en ejes del cuerpo del dron (body-fixed frame)
        dx_body = dx_global * cos_yaw + dy_global * sin_yaw
        dy_body = -dx_global * sin_yaw + dy_global * cos_yaw

        forward_norm = normalize_to_range(dx_body, -2.5, 2.5, -1.0, 1.0, clip=True)
        sideways_norm = normalize_to_range(dy_body, -2.5, 2.5, -1.0, 1.0, clip=True)

        target_obs = np.array([dist_norm, forward_norm, sideways_norm], dtype=np.float32)
        return np.concatenate([base_obs, target_obs])

    def achieve_goal(self) -> bool:
        """
        Determina si el dron alcanzó con éxito el cono objetivo manteniéndose en vuelo.
        Requiere estar a menos de DISTANCE_THRESHOLD (15 cm) y a una altitud mayor a 0.5 metros.

        Returns:
            bool: True si la meta fue alcanzada válidamente en vuelo.
        """
        self.dist_min_target = self.get_distance_to_target()
        return bool(self.dist_min_target <= self.DISTANCE_THRESHOLD and self.alt > 0.5)

    def is_done(self) -> bool:
        """
        Evalúa las condiciones de terminación física del episodio:
        1. Caída al suelo (altitud < 0.10 m)
        2. Salida de los límites del escenario
        3. Éxito al alcanzar el cono objetivo en vuelo seguro
        4. Truncado por límites de puntaje acumulado

        Nota: El límite de pasos por tiempo (max_episode_steps) se maneja
        como truncamiento en step() según el estándar Gymnasium, sin penalizar
        al agente como si fuera una caída.

        Returns:
            bool: True si el episodio terminó por una condición terminal física.
        """
        # 1. Caída del cuadricóptero
        if self.alt < 0.10:
            self.terminated = True
            self.truncated = False
            self.is_success = False
            self.corner = "fall"
            if self.verbose >= 1:
                print(f"[CornerEnv] Caída detectada (altitud: {self.alt:.2f}m). Episodio terminado.")
            return True

        # 2. Salida de límites de la habitación
        if self.is_out_of_bounds():
            self.terminated = True
            self.truncated = False
            self.is_success = False
            self.corner = "out_of_bounds"
            if self.verbose >= 1:
                print(
                    f"[CornerEnv] Fuera de límites detectado ([x={self.x_global:.2f}, y={self.y_global:.2f}]). "
                    "Episodio terminado."
                )
            return True

        # 3. Éxito al llegar al objetivo en altitud adecuada
        if self.achieve_goal():
            self.terminated = True
            self.truncated = False
            self.is_success = True
            self.corner = self.corner_name
            if self.verbose >= 1:
                print(
                    f"[CornerEnv] ¡Objetivo alcanzado con éxito en '{self.corner_name}'! "
                    f"(Distancia: {self.dist_min_target:.2f}m, Altitud: {self.alt:.2f}m)"
                )
            return True

        # 4. Truncado por puntuación acumulada extrema
        if (
            self.episode_score <= self.MIN_EPISODE_SCORE
            or self.episode_score >= self.MAX_EPISODE_SCORE
        ):
            self.terminated = True
            self.truncated = False
            self.is_success = False
            self.corner = "score_limit"
            if self.verbose >= 1:
                print(
                    f"[CornerEnv] Límite de puntaje acumulado alcanzado ({self.episode_score:.2f}). "
                    "Episodio terminado."
                )
            return True

        return False

    def get_reward(self) -> float:
        """
        Calcula la recompensa del paso actual mediante compute_goal_directed_reward.

        Returns:
            float: Recompensa escalar (+100.0 al alcanzar la meta, shaping diferencial y coste temporal).
        """
        self.dist_min_target = self.get_distance_to_target()

        # Distancia mínima a obstáculos en metros
        min_obstacle_m = min(
            self.dist_front, self.dist_back, self.dist_right, self.dist_left
        ) / 1000.0

        # Evaluación de fallos terminales (caída, fuera de límites o score extremo)
        goal_reached = self.achieve_goal()
        is_failure = (
            self.alt < 0.10
            or self.is_out_of_bounds()
            or (self.episode_score <= self.MIN_EPISODE_SCORE)
            or (self.episode_score >= self.MAX_EPISODE_SCORE)
        )

        min_threshold_m = self.MIN_DIST_OBSTACLES / 1000.0  # 100 mm = 0.10 m

        reward = compute_goal_directed_reward(
            goal_reached=goal_reached,
            is_terminal_failure=is_failure,
            min_obstacle_dist_m=min_obstacle_m,
            dist_to_target_m=self.dist_min_target,
            prev_dist_to_target_m=self.prev_dist_min_target,
            min_dist_threshold_m=min_threshold_m,
            goal_reward=100.0,
            failure_reward=-10.0,
            progress_weight=30.0,
            time_penalty=0.01,
        )

        self.prev_dist_min_target = self.dist_min_target
        return reward

    def reset(self, seed: int | None = None, options: dict | None = None):
        """
        Reinicia el entorno, reposiciona el dron en el centro y reanuda el despegue.

        Returns:
            tuple: (observation, info)
        """
        obs, info = super().reset(seed=seed, options=options)
        self.put_drone_in_the_center()
        self.dist_min_target = self.get_distance_to_target()
        self.prev_dist_min_target = self.dist_min_target
        self.corner = None
        obs = self.get_observations()
        info = self.get_info()
        return obs, info

    def get_info(self) -> dict:
        """
        Devuelve el diccionario con información y métricas detalladas del paso actual.
        """
        info = super().get_info()
        info.update(
            {
                "x_init": self.x_init,
                "y_init": self.y_init,
                "is_success": self.is_success,
                "corner": self.corner,
                "dist_min_target": self.dist_min_target,
            }
        )
        return info
