# -*- coding: utf-8 -*-
"""
Unit tests for evaluation functions and metrics in src/evaluation/metrics.py.
"""

import os
import sys
import tempfile
import numpy as np

# Ensure the project root is in PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

os.environ["WEBOTS_MOCK"] = "1"

from src.envs.corner_env_continuous import CornerEnvContinuous
from src.pilots.base_pilot import BasePilot
from src.evaluation.metrics import evaluate_pilot, create_pilot_instance


class MockPilot(BasePilot):
    """Simulated pilot for unit tests."""

    def __init__(self, action: np.ndarray | None = None):
        self.action = action if action is not None else np.array([0.5, -0.5, 0.0, 0.0], dtype=np.float32)
        self.reset_called = False

    def reset(self):
        self.reset_called = True

    def get_action(self, observation: np.ndarray, **kwargs) -> np.ndarray:
        return self.action


def test_create_pilot_instance():
    """Validate instantiation of different pilot types via the factory."""
    # 1. HeuristicPilot
    h_pilot = create_pilot_instance(pilot_type="heuristic", heuristic_mode="open_loop")
    assert isinstance(h_pilot, BasePilot)

    # 2. OptimalPilot with dummy backend
    opt_pilot = create_pilot_instance(pilot_type="optimal", backend="dummy")
    assert isinstance(opt_pilot, BasePilot)

    # 3. Invalid type
    try:
        create_pilot_instance(pilot_type="desconocido")
        assert False, "Debería haber lanzado ValueError"
    except ValueError:
        pass


def test_evaluate_pilot_mock_environment():
    """Validate execution and statistical computation of evaluate_pilot."""
    env = CornerEnvContinuous(max_episode_steps=10, verbose=0)
    pilot = MockPilot()

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_file = os.path.join(tmpdir, "test_eval_metrics.csv")

        results = evaluate_pilot(
            env=env,
            pilot=pilot,
            n_episodes=5,
            seed=42,
            verbose=0,
            save_csv_path=csv_file,
        )

        # 1. Validate required keys in the results dictionary
        expected_keys = [
            "n_episodes",
            "success_rate",
            "crash_rate",
            "timeout_rate",
            "reward_mean",
            "reward_std",
            "length_mean",
            "length_std",
            "final_distance_mean",
            "rewards",
            "lengths",
            "successes",
            "terminations",
            "final_distances",
        ]
        for key in expected_keys:
            assert key in results, f"Falta clave '{key}' en results"

        assert results["n_episodes"] == 5
        assert len(results["rewards"]) == 5
        assert len(results["lengths"]) == 5
        assert 0.0 <= results["success_rate"] <= 1.0
        assert 0.0 <= results["crash_rate"] <= 1.0
        assert 0.0 <= results["timeout_rate"] <= 1.0

        # 2. Validate that the pilot was reset
        assert pilot.reset_called is True

        # 3. Validate creation of the CSV file
        assert os.path.exists(csv_file), "El archivo CSV de resultados no fue creado"
        with open(csv_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 6  # 1 header row + 5 episodes


if __name__ == "__main__":
    test_create_pilot_instance()
    test_evaluate_pilot_mock_environment()
    print("Todos los tests de métricas pasaron exitosamente.")
