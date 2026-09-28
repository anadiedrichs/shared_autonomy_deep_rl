# -*- coding: utf-8 -*-
"""
Recompensa orientada a la meta para el Crazyflie en control continuo.

Implementa la formulación de reward shaping multicomponente calibrada:
    - Progreso diferencial: w_prog * (d_prev - d_now)
    - Proximidad a obstáculos: -w_obs * exp(-d_min / decay_obs)
    - Suavidad de acciones: -w_act * ||action - prev_action||^2
    - Estabilidad angular: -w_omega * ||ang_vel||^2
    - Costo por paso (anti-hovering): -w_time

Eventos terminales:
    - +50.0 al alcanzar el objetivo en altitud segura
    - -50.0 al colisionar o caer al suelo
"""

import numpy as np


def obstacle_proximity_penalization(
    distance_to_obstacle_m: float,
    amplitude: float = 0.5,
    decay: float = 0.10,
) -> float:
    """
    Penalización exponencial por proximidad a paredes u obstáculos.

    Args:
        distance_to_obstacle_m: Distancia mínima detectada a obstáculos en metros.
        amplitude: Magnitud de penalización en contacto (por defecto 0.5).
        decay: Distancia característica de decaimiento en metros (por defecto 0.10 m = 10 cm).

    Returns:
        float: Valor de penalización no positivo en [-amplitude, 0.0].
    """
    d = max(float(distance_to_obstacle_m), 0.0)
    return -float(amplitude) * float(np.exp(-d / float(decay)))


def walls_proximity_penalization(distance_to_obstacle_m: float) -> float:
    """
    Ecuación 3.1 de la tesis (conservada por compatibilidad): penalización lineal por proximidad.
    """
    return 50.0 * float(distance_to_obstacle_m) - 10.0


def penalization_distance_to_target(distance_to_target_m: float) -> float:
    """
    Ecuación 3.2 de la tesis (conservada por compatibilidad): 1 / x.
    """
    dist = max(float(distance_to_target_m), 0.05)
    return 1.0 / dist


def compute_goal_directed_reward(
    d_prev: float | None = None,
    d_now: float | None = None,
    action: np.ndarray | list[float] | None = None,
    prev_action: np.ndarray | list[float] | None = None,
    ang_vel: np.ndarray | list[float] | None = None,
    d_min: float = 1.0,
    collided: bool = False,
    reached: bool = False,
    state_pos: np.ndarray | list[float] | None = None,
    prev_state_pos: np.ndarray | list[float] | None = None,
    goal_pos: np.ndarray | list[float] | None = None,
    progress_weight: float = 30.0,
    obstacle_weight: float = 0.5,
    obstacle_decay: float = 0.10,
    action_smoothness_weight: float = 0.05,
    angular_velocity_weight: float = 0.01,
    time_penalty: float = 0.01,
    reached_bonus: float = 50.0,
    collision_penalty: float = 50.0,
) -> float:
    """
    Calcula la recompensa continua no normalizada para el Crazyflie.

    Args:
        d_prev: Distancia euclídea anterior a la meta (en metros).
        d_now: Distancia euclídea actual a la meta (en metros).
        action: Vector de acción continuo actual de 4 componentes en [-1.0, 1.0].
        prev_action: Vector de acción continuo del paso anterior.
        ang_vel: Velocidad angular en rad/s (vector de 3 componentes del giróscopo).
        d_min: Distancia mínima detectada a obstáculos en metros.
        collided: True si el dron colisionó contra pared, cayó al suelo o salió de límites.
        reached: True si el dron alcanzó el cono meta en altitud segura.
        state_pos: Posición actual del dron (opcional si se provee d_now).
        prev_state_pos: Posición previa del dron (opcional si se provee d_prev).
        goal_pos: Posición del cono objetivo (opcional si se proveen d_prev y d_now).
        progress_weight: Factor multiplicativo para el progreso diferencial (por defecto 30.0).
        obstacle_weight: Factor de amplitud de penalización a obstáculos (por defecto 0.5).
        obstacle_decay: Radio de decaimiento en metros para obstáculos (por defecto 0.10 m).
        action_smoothness_weight: Peso para penalizar cambios bruscos de acción (por defecto 0.05).
        angular_velocity_weight: Peso para penalizar velocidades angulares altas (por defecto 0.01).
        time_penalty: Costo fijo por cada paso para evitar hovering inactivo (por defecto 0.01).
        reached_bonus: Bonificación por alcanzar el objetivo (por defecto +50.0).
        collision_penalty: Penalización por colisión / caída (por defecto -50.0).

    Returns:
        float: Recompensa escalar del paso.
    """
    # Si se pasan posiciones vectoriales en lugar de distancias escalares
    if d_prev is None and prev_state_pos is not None and goal_pos is not None:
        p_prev = np.asarray(prev_state_pos, dtype=np.float32)
        g = np.asarray(goal_pos, dtype=np.float32)
        d_prev = float(np.linalg.norm(p_prev - g))

    if d_now is None and state_pos is not None and goal_pos is not None:
        p_now = np.asarray(state_pos, dtype=np.float32)
        g = np.asarray(goal_pos, dtype=np.float32)
        d_now = float(np.linalg.norm(p_now - g))

    d_prev_val = float(d_prev) if d_prev is not None else 0.0
    d_now_val = float(d_now) if d_now is not None else 0.0

    act = np.asarray(action, dtype=np.float32) if action is not None else np.zeros(4, dtype=np.float32)
    prev_act = (
        np.asarray(prev_action, dtype=np.float32)
        if prev_action is not None
        else np.zeros(4, dtype=np.float32)
    )
    omega = np.asarray(ang_vel, dtype=np.float32) if ang_vel is not None else np.zeros(3, dtype=np.float32)

    # 1. Progreso diferencial hacia la meta
    r = float(progress_weight) * (d_prev_val - d_now_val)

    # 2. Cercanía a obstáculos (exponencial suave)
    r += obstacle_proximity_penalization(d_min, amplitude=obstacle_weight, decay=obstacle_decay)

    # 3. Suavidad de acciones (penaliza cambios bruscos entre pasos consecutivos)
    act_diff = act - prev_act
    r -= float(action_smoothness_weight) * float(np.dot(act_diff, act_diff))

    # 4. Velocidad angular (penaliza giros bruscos y fomenta estabilidad)
    r -= float(angular_velocity_weight) * float(np.dot(omega, omega))

    # 5. Costo temporal por paso (elimina reward farming por inacción)
    r -= float(time_penalty)

    # 6. Eventos terminales
    if reached:
        r += float(reached_bonus)
    if collided:
        r -= float(collision_penalty)

    return float(r)
