# -*- coding: utf-8 -*-
"""
Entry point of the Webots controller.

Webots executes this script directly when associated with the 'crazyflie' node
in the world file (.wbt).
"""

import os
import sys
import numpy as np

# Ensure project root is in PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from src.envs.corner_env_continuous import CornerEnvContinuous
from src.pilots.heuristic_pilot import HeuristicPilot
from src.pilots.optimal_pilot import OptimalPilot
from src.pilots.noisy_pilot import NoisyPilot
from src.pilots.laggy_pilot import LaggyPilot

# Define a simple heuristic toward the cone (points forward and to the right [0.8, 0.8])
# Observation obs[7]: normalized distance to target, obs[8]: relative angle to target
def heuristic_pilot_fn(obs):
    # Moderate forward vx_cmd (0.3), slight lateral vy (0.1), aligned yaw
    return np.array([0.3, 0.1, 0.0, 0.0], dtype=np.float32)


def run_simulation(max_episodes: int | None = None):
    """
    Main simulation and control loop inside Webots.

    Args:
        max_episodes: Maximum number of episodes to run (None for infinite execution).
    """
    print("[drone_shared_autonomy] Inicializando entorno CornerEnvContinuous...")
    env = CornerEnvContinuous(corner_name="cone_1")

    obs, info = env.reset()
    episode_count = 0

    # Option A0: Deterministic heuristic pilot (approach 1: open_loop, approach 2: proportional)
    # pilot = HeuristicPilot(mode="proportional", corner_name="cone_1", env=env)
    # pilot = HeuristicPilot(mode="open_loop", corner_name="cone_1")

    # Option A1: Dummy heuristic pilot
    # pilot = OptimalPilot(backend="dummy", model_instance=heuristic_pilot_fn)

    # Option A2: Load trained PPO neural network (Stable-Baselines3)
    checkpoint_ppo = os.path.join(project_root, "models", "test_ppo_pilot.zip")
    pilot = OptimalPilot(backend="sb3_ppo", checkpoint_path=checkpoint_ppo)

    # Option A3: Load SAC checkpoint trained with RLtools (.h)
    # checkpoint_sac = os.path.join(project_root, "models", "test_sac_checkpoint.h")
    # pilot = OptimalPilot(backend="rltools_sac", checkpoint_path=checkpoint_sac)

    # Option B: Pilot with Gaussian Noise (simulates imprecision/tremor)
    # pilot = NoisyPilot(optimal_pilot=pilot, sigma=0.25)

    # Option C: Pilot with Lag / Delay (repeats past actions with probability 0.4)
    # pilot = LaggyPilot(optimal_pilot=pilot, p_repeat=0.4)
    
    print("[drone_shared_autonomy] Iniciando bucle de control...")

    while True:
        ###########################################################################################
        # Default continuous base action: [vx=0, vy=0, yaw_rate=0, height_cmd=0] (stable hover)
        # action = np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float32) # move forward. 
        # action = np.zeros(4, dtype=np.float32) # hover or hold position in place.
        ###############################################################
        # The pilot / residual copilot policy will be injected here
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
