# -*- coding: utf-8 -*-
"""
Piloto determinista basado en heurísticas para el entorno CornerEnvContinuous.

Permite operar en dos enfoques principales según el parámetro 'mode':
  1. 'open_loop' (o 'enfoque1', 'fixed'):
     Emite una acción fija constante en lazo abierto hacia la esquina objetivo.
     Ideal para validación cinemática directa y experimentos en línea recta.
  2. 'proportional' (o 'enfoque2', 'closed_loop'):
     Controlador proporcional en lazo cerrado tipo P-Controller sobre la posición 3D.
     Calcula el vector de error cartesiano hacia el cono, lo proyecta al sistema del
     cuerpo del dron (body frame), y modula la velocidad de aproximación con frenado
     automático dentro del radio de éxito para mantener hover estable.
"""

from math import cos, sin, sqrt
import numpy as np


class HeuristicPilot:
    """
    Piloto heurístico determinista con múltiples modalidades de control.
    """

    SUPPORTED_MODES = (
        "open_loop",
        "proportional",
        "enfoque1",
        "enfoque2",
        "fixed",
        "closed_loop",
    )

    # Coordenadas de los 4 conos en la habitación de 2x2m (altitud de referencia: 0.55m)
    CORNER_COORDINATES = {
        "cone_1": np.array([0.9, 0.9, 0.55], dtype=np.float32),
        "cone_2": np.array([-0.9, 0.9, 0.55], dtype=np.float32),
        "cone_3": np.array([-0.9, -0.9, 0.55], dtype=np.float32),
        "cone_4": np.array([0.9, -0.9, 0.55], dtype=np.float32),
    }

    def __init__(
        self,
        mode: str = "proportional",
        corner_name: str = "cone_1",
        target_position: np.ndarray | tuple | list | None = None,
        env: object | None = None,
        fixed_action: np.ndarray | tuple | list | None = None,
        kp_xy: float = 2.5,
        kp_z: float = 3.0,
        target_altitude: float = 0.55,
        target_radius: float = 0.15,
        target_speed_ratio: float = 0.90,
    ):
        """
        Inicializa el piloto heurístico determinista.

        Args:
            mode: Modo de funcionamiento ('open_loop'/'enfoque1' o 'proportional'/'enfoque2').
            corner_name: Identificador del cono objetivo ("cone_1", "cone_2", "cone_3", "cone_4").
            target_position: Posición cartesiana objetivo opcional [x, y, z] (anula corner_name si se especifica).
            env: Referencia opcional al entorno CornerEnvContinuous para lectura directa de odometría/GPS.
            fixed_action: Vector de acción continuo [vx, vy, yaw_rate, vz] para modo open_loop.
            kp_xy: Ganancia proporcional para velocidad horizontal en lazo cerrado.
            kp_z: Ganancia proporcional para variación de altitud en lazo cerrado.
            target_altitude: Altitud objetivo en metros (por defecto 0.55 m para superar el umbral de 0.50 m).
            target_radius: Radio en metros dentro del cual se modula la desaceleración (por defecto 0.15 m).
            target_speed_ratio: Fracción de la velocidad máxima en lazo abierto (en [0.0, 1.0]).
        """
        mode_lower = mode.lower()
        if mode_lower not in self.SUPPORTED_MODES:
            raise ValueError(
                f"Modo '{mode}' no soportado. Opciones válidas: {self.SUPPORTED_MODES}"
            )

        self.mode = mode_lower
        self.is_open_loop = self.mode in ("open_loop", "enfoque1", "fixed")
        self.corner_name = corner_name
        self.env = env
        self.kp_xy = float(kp_xy)
        self.kp_z = float(kp_z)
        self.target_altitude = float(target_altitude)
        self.target_radius = float(target_radius)
        self.target_speed_ratio = float(np.clip(target_speed_ratio, 0.1, 1.0))

        # Configurar posición objetivo cartesiana [x, y, z]
        if target_position is not None:
            self.target_position = np.asarray(target_position, dtype=np.float32)
        else:
            self.target_position = self.CORNER_COORDINATES.get(
                corner_name, np.array([0.9, 0.9, self.target_altitude], dtype=np.float32)
            ).copy()
        # Asegurar altitud objetivo deseada
        self.target_position[2] = self.target_altitude

        # Configurar acción fija para Enfoque 1 (open_loop)
        if fixed_action is not None:
            self.fixed_action = np.clip(np.asarray(fixed_action, dtype=np.float32), -1.0, 1.0)
        else:
            self.fixed_action = self._calculate_default_fixed_action()

    def _calculate_default_fixed_action(self) -> np.ndarray:
        """
        Calcula la acción constante por defecto hacia la esquina objetivo.
        Para un objetivo en el cuadrante (+x, +y), emite [speed, speed, 0.0, vz_climb].
        """
        dx = float(self.target_position[0])
        dy = float(self.target_position[1])

        vx_sign = 1.0 if dx >= 0.0 else -1.0
        vy_sign = 1.0 if dy >= 0.0 else -1.0

        # vz = 0.60 para generar ascenso constante de 0.30m a 0.55m durante el vuelo
        return np.array(
            [
                vx_sign * self.target_speed_ratio,
                vy_sign * self.target_speed_ratio,
                0.0,
                0.60,
            ],
            dtype=np.float32,
        )

    def _get_open_loop_action(self, observation: np.ndarray) -> np.ndarray:
        """
        Enfoque 1: Acción fija constante en lazo abierto.
        """
        return np.copy(self.fixed_action)

    def _get_proportional_action(
        self, observation: np.ndarray, env: object | None = None
    ) -> np.ndarray:
        """
        Enfoque 2: Controlador proporcional en lazo cerrado con frenado y mantenimiento de altitud.
        """
        active_env = env if env is not None else self.env

        # 1. Obtención de odometría global del dron
        if active_env is not None and hasattr(active_env, "x_global"):
            curr_x = float(active_env.x_global)
            curr_y = float(active_env.y_global)
            curr_z = float(getattr(active_env, "alt", getattr(active_env, "z_global", 0.30)))
            curr_yaw = float(getattr(active_env, "yaw", 0.0))
        else:
            # Estimación desde la observación si no se provee el entorno
            # obs[6] = altitud normalizada en [0.1, 5.0]
            obs = np.asarray(observation, dtype=np.float32)
            alt_norm = float(obs[6]) if len(obs) > 6 else 0.0
            curr_z = 0.1 + (alt_norm + 1.0) / 2.0 * (5.0 - 0.1)
            curr_x = 0.0
            curr_y = 0.0
            curr_yaw = 0.0

        # 2. Errores cartesianos hacia el objetivo
        err_x = float(self.target_position[0] - curr_x)
        err_y = float(self.target_position[1] - curr_y)
        err_z = float(self.target_position[2] - curr_z)
        dist_xy = float(sqrt(err_x * err_x + err_y * err_y))

        # 3. Transformación del error horizontal al marco de referencia del cuerpo del dron (body frame)
        cos_yaw = cos(curr_yaw)
        sin_yaw = sin(curr_yaw)
        err_x_body = float(err_x * cos_yaw + err_y * sin_yaw)
        err_y_body = float(-err_x * sin_yaw + err_y * cos_yaw)

        # 4. Modulación proporcional de comandos horizontales
        if dist_xy > self.target_radius:
            # En vuelo libre hacia el cono: comando proporcional acotado a [-1.0, 1.0]
            cmd_vx = float(np.clip(self.kp_xy * err_x_body, -1.0, 1.0))
            cmd_vy = float(np.clip(self.kp_xy * err_y_body, -1.0, 1.0))
        else:
            # Dentro del radio del objetivo: zona de frenado suave para hover estable
            brake_scale = max(dist_xy / self.target_radius, 0.1)
            cmd_vx = float(np.clip(self.kp_xy * err_x_body * brake_scale, -0.25, 0.25))
            cmd_vy = float(np.clip(self.kp_xy * err_y_body * brake_scale, -0.25, 0.25))

        # 5. Comando vertical proporcional hacia la altitud objetivo (0.55m)
        cmd_vz = float(np.clip(self.kp_z * err_z, -1.0, 1.0))

        # 6. Yaw neutral (mantiene orientación frontal estable sin giros bruscos)
        cmd_yaw = 0.0

        return np.array([cmd_vx, cmd_vy, cmd_yaw, cmd_vz], dtype=np.float32)

    def get_action(
        self, observation: np.ndarray, env: object | None = None
    ) -> np.ndarray:
        """
        Calcula la acción del piloto heurístico para la observación actual.

        Args:
            observation: Vector continuo de observaciones de 11 dimensiones.
            env: Referencia opcional al entorno para lectura de posición y yaw en lazo cerrado.

        Returns:
            np.ndarray: Vector de acción continuo [vx, vy, yaw_rate, vz] en [-1.0, 1.0]^4.
        """
        if self.is_open_loop:
            return self._get_open_loop_action(observation)
        return self._get_proportional_action(observation, env=env)

    def choose_action(
        self, observation: np.ndarray, env: object | None = None
    ) -> tuple[np.ndarray, list]:
        """
        Alias compatible con la API de pilotos sintéticos de la tesis (Diedrichs, 2025).

        Returns:
            tuple: (acción seleccionada, lista vacía de estados auxiliares).
        """
        return self.get_action(observation, env=env), []
