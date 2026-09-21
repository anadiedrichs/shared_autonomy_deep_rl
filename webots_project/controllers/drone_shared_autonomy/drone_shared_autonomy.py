# -*- coding: utf-8 -*-
"""
Punto de entrada (Entry point) del controlador de Webots.

Webots ejecuta este script directamente al asociarlo al nodo 'crazyflie'
en el archivo del mundo (.wbt).
"""

import os
import sys
import numpy as np

# Asegurar que la raíz del proyecto esté en el PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.envs.corner_env_continuous import CornerEnvContinuous


def run_simulation(max_episodes: int | None = None):
    """
    Bucle principal de simulación y control dentro de Webots.

    Args:
        max_episodes: Número máximo de episodios a ejecutar (None para ejecución infinita).
    """
    print("[drone_shared_autonomy] Inicializando entorno CornerEnvContinuous...")
    env = CornerEnvContinuous(corner_name="cone_1")

    obs, info = env.reset()
    episode_count = 0

    print("[drone_shared_autonomy] Iniciando bucle de control...")

    while True:
        # Acción base continua por defecto: [vx=0, vy=0, yaw_rate=0, height_cmd=0] (hover estable)
        # En etapas siguientes se inyectará aquí la política del piloto / copiloto residual
        action = np.zeros(4, dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            episode_count += 1
            print(
                f"[drone_shared_autonomy] Episodio {episode_count} finalizado | "
                f"Éxito: {info.get('is_success')} | Causa: {info.get('corner')} | "
                f"Distancia meta: {info.get('dist_min_target', 0.0):.2f}m"
            )

            if max_episodes is not None and episode_count >= max_episodes:
                print(f"[drone_shared_autonomy] Límite de {max_episodes} episodios alcanzado.")
                break

            obs, info = env.reset()


if __name__ == "__main__":
    run_simulation()
