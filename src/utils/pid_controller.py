# -*- coding: utf-8 -*-
"""
Controlador PID para Crazyflie con control de velocidad y altitud fija.

Adaptado del controlador en C de Bitcraze AB para Crazyflie en Webots.
"""

import numpy as np


class pid_velocity_fixed_height_controller:
    """
    Controlador PID de velocidad horizontal y mantenimiento de altitud fija para Crazyflie.
    """

    def __init__(self):
        # Errores en pasos de tiempo previos
        self.past_vx_error = 0.0
        self.past_vy_error = 0.0
        self.past_alt_error = 0.0
        self.past_pitch_error = 0.0
        self.past_roll_error = 0.0
        # Integrador de error de altitud
        self.altitude_integrator = 0.0
        self.last_time = 0.0

    def pid(
        self,
        dt: float,
        desired_vx: float,
        desired_vy: float,
        desired_yaw_rate: float,
        desired_altitude: float,
        actual_roll: float,
        actual_pitch: float,
        actual_yaw_rate: float,
        actual_altitude: float,
        actual_vx: float,
        actual_vy: float,
    ) -> list[float]:
        """
        Calcula las potencias individuales de los 4 motores para alcanzar las velocidades y altitud deseadas.

        Args:
            dt: Paso de tiempo de simulación transcurrido (en segundos).
            desired_vx: Velocidad lineal deseada en eje X (adelante/atrás en m/s).
            desired_vy: Velocidad lineal deseada en eje Y (lateral en m/s).
            desired_yaw_rate: Tasa de giro deseada en yaw (rad/s).
            desired_altitude: Altitud objetivo deseada (metros).
            actual_roll: Ángulo actual de roll (radianes).
            actual_pitch: Ángulo actual de pitch (radianes).
            actual_yaw_rate: Velocidad angular actual en yaw (rad/s).
            actual_altitude: Altitud actual medida (metros).
            actual_vx: Velocidad lineal actual medida en eje X (m/s).
            actual_vy: Velocidad lineal actual medida en eje Y (m/s).

        Returns:
            list[float]: Lista de 4 valores de comando de potencia [m1, m2, m3, m4] en el rango [0, 600].
        """
        # Ganancias del controlador PID (idénticas a la configuración validada en C de Webots / tesis 2025)
        gains = {
            "kp_att_y": 1.0,
            "kd_att_y": 0.5,
            "kp_att_rp": 0.5,
            "kd_att_rp": 0.1,
            "kp_vel_xy": 2.0,
            "kd_vel_xy": 0.5,
            "kp_z": 10.0,
            "ki_z": 5.0,
            "kd_z": 5.0,
        }

        # 1. Control de velocidad horizontal (genera ángulos de inclinación deseados: pitch y roll)
        vx_error = desired_vx - actual_vx
        vx_deriv = (vx_error - self.past_vx_error) / dt
        vy_error = desired_vy - actual_vy
        vy_deriv = (vy_error - self.past_vy_error) / dt

        desired_pitch = gains["kp_vel_xy"] * np.clip(vx_error, -1.0, 1.0) + gains["kd_vel_xy"] * vx_deriv
        desired_roll = -gains["kp_vel_xy"] * np.clip(vy_error, -1.0, 1.0) - gains["kd_vel_xy"] * vy_deriv
        self.past_vx_error = vx_error
        self.past_vy_error = vy_error

        # 2. Control de altitud (PID con término integral acotado y offset base de empuje de 48)
        alt_error = desired_altitude - actual_altitude
        alt_deriv = (alt_error - self.past_alt_error) / dt
        self.altitude_integrator += alt_error * dt
        alt_command = (
            gains["kp_z"] * alt_error
            + gains["kd_z"] * alt_deriv
            + gains["ki_z"] * np.clip(self.altitude_integrator, -2.0, 2.0)
            + 48.0
        )
        self.past_alt_error = alt_error

        # 3. Control de actitud (orientación roll, pitch y velocidad de guiñada yaw)
        pitch_error = desired_pitch - actual_pitch
        pitch_deriv = (pitch_error - self.past_pitch_error) / dt
        roll_error = desired_roll - actual_roll
        roll_deriv = (roll_error - self.past_roll_error) / dt
        yaw_rate_error = desired_yaw_rate - actual_yaw_rate

        roll_command = gains["kp_att_rp"] * np.clip(roll_error, -1.0, 1.0) + gains["kd_att_rp"] * roll_deriv
        pitch_command = -gains["kp_att_rp"] * np.clip(pitch_error, -1.0, 1.0) - gains["kd_att_rp"] * pitch_deriv
        yaw_command = gains["kp_att_y"] * np.clip(yaw_rate_error, -1.0, 1.0)

        self.past_pitch_error = pitch_error
        self.past_roll_error = roll_error

        # 4. Mezclado de motores (motor mixing)
        m1 = alt_command - roll_command + pitch_command + yaw_command
        m2 = alt_command - roll_command - pitch_command - yaw_command
        m3 = alt_command + roll_command - pitch_command + yaw_command
        m4 = alt_command + roll_command + pitch_command - yaw_command

        # Limitar la salida de comando a los actuadores [0, 600]
        m1 = float(np.clip(m1, 0.0, 600.0))
        m2 = float(np.clip(m2, 0.0, 600.0))
        m3 = float(np.clip(m3, 0.0, 600.0))
        m4 = float(np.clip(m4, 0.0, 600.0))

        return [m1, m2, m3, m4]
