# -*- coding: utf-8 -*-
"""
Tests unitarios para el script de entrenamiento de la política residual (Fase 5).
"""

import os
import sys
import numpy as np

# Asegurar que la raíz del proyecto esté en el PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

os.environ["WEBOTS_MOCK"] = "1"

from src.pilots.heuristic_pilot import HeuristicPilot
from src.pilots.laggy_pilot import LaggyPilot
from src.pilots.noisy_pilot import NoisyPilot
from src.training.train_residual_copilot import create_base_pilot_for_training


def test_create_base_pilot_for_training_heuristic():
    """Valida la creación de HeuristicPilot como piloto congelado."""
    pilot = create_base_pilot_for_training(pilot_type="heuristic", corner_name="cone_1")
    assert isinstance(pilot, HeuristicPilot)
    act = pilot.get_action(np.zeros(11, dtype=np.float32))
    assert act.shape == (4,)


def test_create_base_pilot_for_training_noisy():
    """Valida la creación de NoisyPilot envolviendo el piloto base."""
    pilot = create_base_pilot_for_training(
        pilot_type="noisy",
        corner_name="cone_1",
        noise_std=0.25,
        seed=123,
    )
    assert isinstance(pilot, NoisyPilot)
    assert pilot.sigma == 0.25


def test_create_base_pilot_for_training_laggy():
    """Valida la creación de LaggyPilot envolviendo el piloto base."""
    pilot = create_base_pilot_for_training(
        pilot_type="laggy",
        corner_name="cone_1",
        lag_prob=0.35,
        seed=456,
    )
    assert isinstance(pilot, LaggyPilot)
    assert pilot.p_repeat == 0.35


def test_smoke_training_step_ppo():
    """Smoke test: valida que PPO pueda realizar pasos de optimización en SharedAutonomyEnv."""
    from stable_baselines3 import PPO
    from src.envs.shared_autonomy_env import SharedAutonomyEnv

    pilot = create_base_pilot_for_training(pilot_type="heuristic")
    env = SharedAutonomyEnv(pilot=pilot, verbose=0)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        n_steps=32,
        batch_size=16,
        n_epochs=1,
        verbose=0,
    )

    # Entrenar por 32 timesteps para verificar ciclo de gradientes y forward pass
    model.learn(total_timesteps=32)
    assert model.num_timesteps >= 32
    env.close()


def test_smoke_training_step_sac():
    """Smoke test: valida que SAC pueda realizar pasos de optimización en SharedAutonomyEnv."""
    from stable_baselines3 import SAC
    from src.envs.shared_autonomy_env import SharedAutonomyEnv

    pilot = create_base_pilot_for_training(pilot_type="heuristic")
    env = SharedAutonomyEnv(pilot=pilot, verbose=0)

    model = SAC(
        policy="MlpPolicy",
        env=env,
        buffer_size=1000,
        batch_size=16,
        learning_starts=16,
        verbose=0,
    )

    # Entrenar por 20 timesteps para verificar replay buffer y target update
    model.learn(total_timesteps=20)
    assert model.num_timesteps >= 20
    env.close()


if __name__ == "__main__":
    test_create_base_pilot_for_training_heuristic()
    test_create_base_pilot_for_training_noisy()
    test_create_base_pilot_for_training_laggy()
    test_smoke_training_step_ppo()
    test_smoke_training_step_sac()
    print("Todos los tests de entrenamiento de copiloto residual pasaron exitosamente.")
