# -*- coding: utf-8 -*-
"""
Entorno continuo del Crazyflie en habitación 2x2m con objetivo en una esquina (CornerEnv).

Port continuo de la clase SimpleCornerEnvRS10 de la tesis (Diedrichs, 2025, drone-deep-rl).
"""

from math import sqrt
import numpy as np

from src.envs.drone_robot_supervisor import DroneRobotSupervisor
from src.rewards.reward_goal_directed import compute_goal_directed_reward


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

    def get_distance_to_target(self) -> float:
        """
        Calcula la distancia euclídea en el plano horizontal (XY) entre el dron y el objetivo.

        Returns:
            float: Distancia en metros al objetivo.
        """
        if self.target_node is not None and hasattr(self.target_node, "getField"):
            target_field = self.target_node.getField("translation")
            if target_field is not None and hasattr(target_field, "getSFVec3f"):
                target_pos = target_field.getSFVec3f()
                dx = self.x_global - target_pos[0]
                dy = self.y_global - target_pos[1]
                return float(sqrt(dx * dx + dy * dy))

        # Cálculo de respaldo para pruebas unitarias / CI sin Webots
        dx = self.x_global - self._target_pos_fallback[0]
        dy = self.y_global - self._target_pos_fallback[1]
        return float(sqrt(dx * dx + dy * dy))

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
        Evalúa las condiciones de finalización del episodio:
        1. Caída al suelo (altitud < 0.10 m)
        2. Salida de los límites del escenario
        3. Éxito al alcanzar el cono objetivo en vuelo seguro
        4. Truncado por límites de puntaje acumulado
        5. Límite de pasos del episodio excedido

        Returns:
            bool: True si el episodio ha terminado o debe truncarse.
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
            float: Recompensa continua no normalizada.
        """
        self.dist_min_target = self.get_distance_to_target()

        # Distancia mínima a obstáculos en metros
        d_min = min(
            self.dist_front, self.dist_back, self.dist_right, self.dist_left
        ) / 1000.0

        # Evaluación de eventos terminales
        reached = self.achieve_goal()
        collided = bool(
            self.alt < 0.10
            or self.is_out_of_bounds()
            or (self.episode_score <= self.MIN_EPISODE_SCORE)
            or (self.episode_score >= self.MAX_EPISODE_SCORE)
        )

        reward = compute_goal_directed_reward(
            d_prev=self.prev_dist_min_target,
            d_now=self.dist_min_target,
            action=self.current_action,
            prev_action=self.prev_action,
            ang_vel=self.angular_velocity,
            d_min=d_min,
            collided=collided,
            reached=reached,
            progress_weight=30.0,
            obstacle_weight=0.5,
            obstacle_decay=0.10,
            action_smoothness_weight=0.05,
            angular_velocity_weight=0.01,
            time_penalty=0.01,
            reached_bonus=50.0,
            collision_penalty=50.0,
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
