# -*- coding: utf-8 -*-
"""
PID controller for the Crazyflie with velocity and fixed altitude control.

Adapted from the Bitcraze AB C controller for the Crazyflie in Webots.
"""

import numpy as np


class pid_velocity_fixed_height_controller:
    """
    Horizontal velocity and fixed altitude hold PID controller for the Crazyflie.
    """

    def __init__(self):
        # Errors from previous time steps
        self.past_vx_error = 0.0
        self.past_vy_error = 0.0
        self.past_alt_error = 0.0
        self.past_pitch_error = 0.0
        self.past_roll_error = 0.0
        # Altitude error integrator
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
        Computes individual motor power commands for all 4 motors to achieve the desired velocities and altitude.

        Args:
            dt: Elapsed simulation time step (in seconds).
            desired_vx: Desired linear velocity along the X axis (forward/backward in m/s).
            desired_vy: Desired linear velocity along the Y axis (lateral in m/s).
            desired_yaw_rate: Desired yaw rate (rad/s).
            desired_altitude: Desired target altitude (meters).
            actual_roll: Current roll angle (radians).
            actual_pitch: Current pitch angle (radians).
            actual_yaw_rate: Current angular velocity in yaw (rad/s).
            actual_altitude: Current measured altitude (meters).
            actual_vx: Current measured linear velocity along the X axis (m/s).
            actual_vy: Current measured linear velocity along the Y axis (m/s).

        Returns:
            list[float]: List of 4 motor power command values [m1, m2, m3, m4] in the range [0, 600].
        """
        # PID controller gains (identical to the validated Webots C configuration / 2025 thesis)
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

        # 1. Horizontal velocity control (generates desired tilt angles: pitch and roll)
        vx_error = desired_vx - actual_vx
        vx_deriv = np.clip((vx_error - self.past_vx_error) / dt, -2.0, 2.0)
        vy_error = desired_vy - actual_vy
        vy_deriv = np.clip((vy_error - self.past_vy_error) / dt, -2.0, 2.0)

        desired_pitch = gains["kp_vel_xy"] * np.clip(vx_error, -1.0, 1.0) + gains["kd_vel_xy"] * vx_deriv
        desired_roll = -gains["kp_vel_xy"] * np.clip(vy_error, -1.0, 1.0) - gains["kd_vel_xy"] * vy_deriv
        self.past_vx_error = vx_error
        self.past_vy_error = vy_error

        # 2. Altitude control (PID with bounded integral term and base thrust offset of 48)
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

        # 3. Attitude control (roll, pitch orientation and yaw angular velocity)
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

        # 4. Motor mixing
        m1 = alt_command - roll_command + pitch_command + yaw_command
        m2 = alt_command - roll_command - pitch_command - yaw_command
        m3 = alt_command + roll_command - pitch_command + yaw_command
        m4 = alt_command + roll_command + pitch_command - yaw_command

        # Clamp actuator command output to [0, 600]
        m1 = float(np.clip(m1, 0.0, 600.0))
        m2 = float(np.clip(m2, 0.0, 600.0))
        m3 = float(np.clip(m3, 0.0, 600.0))
        m4 = float(np.clip(m4, 0.0, 600.0))

        return [m1, m2, m3, m4]
