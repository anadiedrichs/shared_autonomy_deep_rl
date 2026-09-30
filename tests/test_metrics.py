# -*- coding: utf-8 -*-
"""
Tests unitarios para las funciones de evaluación y métricas en src/evaluation/metrics.py.
"""

import os
import sys
import tempfile
import numpy as np

# Asegurar que la raíz del proyecto esté en el PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

os.environ["WEBOTS_MOCK"] = "1"

from src.envs.corner_env_continuous import CornerEnvContinuous
from src.pilots.base_pilot import BasePilot
from src.evaluation.metrics import evaluate_pilot, create_pilot_instance


class MockPilot(BasePilot):
    """Piloto simulado para tests unitarios."""

    def __init__(self, action: np.ndarray | None = None):
        self.action = action if action is not None else np.array([0.5, -0.5, 0.0, 0.0], dtype=np.float32)
        self.reset_called = False

    def reset(self):
        self.reset_called = True

    def get_action(self, observation: np.ndarray, **kwargs) -> np.ndarray:
        return self.action


def test_create_pilot_instance():
    """Valida la instanciación de diferentes tipos de pilotos mediante la fábrica."""
    # 1. HeuristicPilot
    h_pilot = create_pilot_instance(pilot_type="heuristic", heuristic_mode="open_loop")
    assert isinstance(h_pilot, BasePilot)

    # 2. OptimalPilot con dummy backend
    opt_pilot = create_pilot_instance(pilot_type="optimal", backend="dummy")
    assert isinstance(opt_pilot, BasePilot)

    # 3. Tipo no válido
    try:
        create_pilot_instance(pilot_type="desconocido")
        assert False, "Debería haber lanzado ValueError"
    except ValueError:
        pass


def test_evaluate_pilot_mock_environment():
    """Valida la ejecución y cálculo estadístico de evaluate_pilot."""
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

        # 1. Validar claves requeridas en el diccionario de resultados
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

        # 2. Validar que el piloto fue reseteado
        assert pilot.reset_called is True

        # 3. Validar creación del archivo CSV
        assert os.path.exists(csv_file), "El archivo CSV de resultados no fue creado"
        with open(csv_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 6  # 1 fila encabezado + 5 episodios


if __name__ == "__main__":
    test_create_pilot_instance()
    test_evaluate_pilot_mock_environment()
    print("Todos los tests de métricas pasaron exitosamente.")
