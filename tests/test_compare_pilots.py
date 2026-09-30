# -*- coding: utf-8 -*-
"""
Tests unitarios para compare_pilots.py (Fase 6: Análisis Comparativo).
"""

import os
import sys
import tempfile
import numpy as np
import pandas as pd

# Asegurar que la raíz del proyecto esté en el PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.evaluation.compare_pilots import (
    calculate_pilot_metrics,
    compare_pilots_from_csvs,
    generate_comparison_plots,
    load_evaluation_csv,
)


def _create_mock_eval_csv(path: str, successes: list[bool], terminations: list[str]):
    n = len(successes)
    df = pd.DataFrame(
        {
            "episode": list(range(1, n + 1)),
            "reward": [10.0 * (i + 1) for i in range(n)],
            "length": [100 + 10 * i for i in range(n)],
            "success": successes,
            "termination": terminations,
            "final_distance": [0.10 if s else 0.85 for s in successes],
        }
    )
    df.to_csv(path, index=False)


def test_load_and_calculate_metrics():
    """Valida la carga de CSV y cálculo de tasas de éxito, colisión y timeout."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_file = os.path.join(tmpdir, "eval_test.csv")
        # 10 episodios: 7 éxitos, 2 caídas (colisión), 1 timeout
        successes = [True] * 7 + [False] * 3
        terminations = ["goal"] * 7 + ["fall", "out_of_bounds", "timeout"]
        _create_mock_eval_csv(csv_file, successes, terminations)

        df = load_evaluation_csv(csv_file)
        assert len(df) == 10

        metrics = calculate_pilot_metrics(df, label="TestPilot")
        assert metrics["pilot"] == "TestPilot"
        assert metrics["episodes"] == 10
        assert np.isclose(metrics["success_rate"], 70.0)
        assert np.isclose(metrics["collision_rate"], 20.0)
        assert np.isclose(metrics["timeout_rate"], 10.0)
        assert metrics["mean_final_distance"] > 0.0


def test_compare_multiple_pilots_from_csvs():
    """Valida la agregación tabular de múltiples pilotos."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_p1 = os.path.join(tmpdir, "eval_p1.csv")
        csv_p2 = os.path.join(tmpdir, "eval_p2.csv")

        _create_mock_eval_csv(csv_p1, [True, True, True, False], ["goal", "goal", "goal", "fall"])
        _create_mock_eval_csv(csv_p2, [True, False, False, False], ["goal", "fall", "fall", "timeout"])

        df_summary = compare_pilots_from_csvs([csv_p1, csv_p2], labels=["PilotOptimal", "PilotNoisy"])

        assert len(df_summary) == 2
        assert list(df_summary["pilot"]) == ["PilotOptimal", "PilotNoisy"]
        assert df_summary.loc[0, "success_rate"] == 75.0
        assert df_summary.loc[1, "success_rate"] == 25.0
        assert df_summary.loc[1, "collision_rate"] == 50.0


def test_generate_comparison_plots():
    """Valida la generación física de imágenes de reporte PNG."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_p1 = os.path.join(tmpdir, "eval_p1.csv")
        _create_mock_eval_csv(csv_p1, [True, False], ["goal", "fall"])

        df_summary = compare_pilots_from_csvs([csv_p1], labels=["P1"])
        plots = generate_comparison_plots(df_summary, output_dir=tmpdir)

        # Si matplotlib está instalado, debe generar 2 imágenes
        for p in plots:
            assert os.path.exists(p)
            assert p.endswith(".png")
            assert os.path.getsize(p) > 0


if __name__ == "__main__":
    test_load_and_calculate_metrics()
    test_compare_multiple_pilots_from_csvs()
    test_generate_comparison_plots()
    print("Todos los tests de compare_pilots pasaron exitosamente.")
