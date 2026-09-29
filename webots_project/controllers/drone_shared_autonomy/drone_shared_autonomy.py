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
from src.pilots.heuristic_pilot import HeuristicPilot
from src.pilots.optimal_pilot import OptimalPilot
from src.pilots.noisy_pilot import NoisyPilot
from src.pilots.laggy_pilot import LaggyPilot

# Definimos una heurística simple hacia el cono (apunta hacia adelante y a la derecha [0.8, 0.8])
# Observación obs[7]: distancia al objetivo normalizada, obs[8]: ángulo relativo al objetivo
def heuristic_pilot_fn(obs):
    # vx_cmd moderado hacia adelante (0.3), vy lateral leve (0.1), yaw alineado
    return np.array([0.3, 0.1, 0.0, 0.0], dtype=np.float32)


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

    # Opción A0: Piloto heurístico determinista (enfoque 1: open_loop, enfoque 2: proportional)
    # pilot = HeuristicPilot(mode="proportional", corner_name="cone_1", env=env)
    # pilot = HeuristicPilot(mode="open_loop", corner_name="cone_1")

    # Opción A1: Piloto heurístico dummy
    # pilot = OptimalPilot(backend="dummy", model_instance=heuristic_pilot_fn)

    # Opción A2: Carga la red neuronal PPO entrenada (Stable-Baselines3)
    checkpoint_ppo = os.path.join(project_root, "models", "test_ppo_pilot.zip")
    pilot = OptimalPilot(backend="sb3_ppo", checkpoint_path=checkpoint_ppo)

    # Opción A3: Carga el checkpoint SAC entrenado con RLtools (.h)
    # checkpoint_sac = os.path.join(project_root, "models", "test_sac_checkpoint.h")
    # pilot = OptimalPilot(backend="rltools_sac", checkpoint_path=checkpoint_sac)

    # Opción B: Piloto con Ruido Gaussiano (simula imprecisión/temblor)
    # pilot = NoisyPilot(optimal_pilot=pilot, sigma=0.25)

    # Opción C: Piloto con Lag / Retardo (repite acciones pasadas con probabilidad 0.4)
    # pilot = LaggyPilot(optimal_pilot=pilot, p_repeat=0.4)
    
    print("[drone_shared_autonomy] Iniciando bucle de control...")

    while True:
        ###########################################################################################
        # Acción base continua por defecto: [vx=0, vy=0, yaw_rate=0, height_cmd=0] (hover estable)
        # action = np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float32) # avanzar hacia adelante. 
        # action = np.zeros(4, dtype=np.float32) # hover o mantenerse en el mismo lugar.
        ###############################################################
        # Se inyectará aquí la política del piloto / copiloto residual
        action = pilot.get_action(obs)
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
