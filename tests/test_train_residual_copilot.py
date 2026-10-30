# -*- coding: utf-8 -*-
"""
Unit tests for the residual policy training script (Phase 5).
"""

import os
import sys
import numpy as np

# Ensure the project root is in PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

os.environ["WEBOTS_MOCK"] = "1"

from src.pilots.heuristic_pilot import HeuristicPilot
from src.pilots.laggy_pilot import LaggyPilot
from src.pilots.noisy_pilot import NoisyPilot
from src.training.train_residual_copilot import create_base_pilot_for_training


def test_create_base_pilot_for_training_heuristic():
    """Validate creation of HeuristicPilot as a frozen pilot."""
    pilot = create_base_pilot_for_training(pilot_type="heuristic", corner_name="cone_1")
    assert isinstance(pilot, HeuristicPilot)
    act = pilot.get_action(np.zeros(11, dtype=np.float32))
    assert act.shape == (4,)


def test_create_base_pilot_for_training_noisy():
    """Validate creation of NoisyPilot wrapping the base pilot."""
    pilot = create_base_pilot_for_training(
        pilot_type="noisy",
        corner_name="cone_1",
        noise_std=0.25,
        seed=123,
    )
    assert isinstance(pilot, NoisyPilot)
    assert pilot.sigma == 0.25


def test_create_base_pilot_for_training_laggy():
    """Validate creation of LaggyPilot wrapping the base pilot."""
    pilot = create_base_pilot_for_training(
        pilot_type="laggy",
        corner_name="cone_1",
        lag_prob=0.35,
        seed=456,
    )
    assert isinstance(pilot, LaggyPilot)
    assert pilot.p_repeat == 0.35


def test_smoke_training_step_ppo():
    """Smoke test: validate that PPO can perform optimization steps in SharedAutonomyEnv."""
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

    # Train for 32 timesteps to verify gradient cycle and forward pass
    model.learn(total_timesteps=32)
    assert model.num_timesteps >= 32
    env.close()


def test_smoke_training_step_sac():
    """Smoke test: validate that SAC can perform optimization steps in SharedAutonomyEnv."""
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

    # Train for 20 timesteps to verify replay buffer and target update
    model.learn(total_timesteps=20)
    assert model.num_timesteps >= 20
    env.close()


def test_create_base_pilot_with_optimal_checkpoint_path():
    """Validate that passing a relative path to an optimal model properly resolves and creates OptimalPilot."""
    optimal_path = "models/ppo_optimal_pilot.zip"
    if os.path.exists(os.path.join(project_root, optimal_path)):
        pilot = create_base_pilot_for_training(
            pilot_type="noisy",
            optimal_model_path=optimal_path,
            noise_std=0.30,
        )
        assert isinstance(pilot, NoisyPilot)
        assert hasattr(pilot, "optimal_pilot")
        from src.pilots.optimal_pilot import OptimalPilot

        assert isinstance(pilot.optimal_pilot, OptimalPilot)


if __name__ == "__main__":
    test_create_base_pilot_for_training_heuristic()
    test_create_base_pilot_for_training_noisy()
    test_create_base_pilot_for_training_laggy()
    test_create_base_pilot_with_optimal_checkpoint_path()
    test_smoke_training_step_ppo()
    test_smoke_training_step_sac()
    print("Todos los tests de entrenamiento de copiloto residual pasaron exitosamente.")
