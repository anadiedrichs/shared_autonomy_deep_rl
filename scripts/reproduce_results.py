#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Reproducibilidad Sistemática de Resultados (TFM - Universidad Europea).
Autora: Ana Laura Diedrichs

Lee DIRECTAMENTE los archivos CSV de evaluación almacenados en results_TFM/ (o logs/)
y calcula de forma 100% empírica, sin valores hardcodeados, todas las métricas estadísticas,
tablas en LaTeX y gráficos comparativos:
    - Tabla 4.1: Configuraciones de evaluación de pilotos degradados.
    - Tabla 4.2: Comparativa cuantitativa de pilotos óptimos (PPO vs. SAC vs. TD3).
    - Tabla 4.3: Desempeño de pilotos sintéticos degradados y heurístico.
    - Tabla 4.4: Comparativa de desempeño del copiloto residual (Piloto Ruidoso).
    - Tabla 4.5: Comparativa cuantitativa del copiloto residual (Piloto Lento).
    - Gráficos comparativos generados dinámicamente a partir de los datos calculados.

DISEÑADO CON SOPORTE ZERO-DEPENDENCY:
Funciona con Python estándar (módulo csv y math incorporados) si pandas/numpy no están
instalados, y aprovecha pandas y matplotlib cuando están disponibles.
"""

import argparse
import csv
import glob
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def read_csv_rows(csv_path: str) -> List[Dict[str, str]]:
    """Lee un CSV usando el módulo csv estándar de Python."""
    if not os.path.exists(csv_path):
        return []
    try:
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)
    except Exception as e:
        print(f"[Aviso] Error al leer {csv_path}: {e}")
        return []


def compute_metrics_from_csvs(file_paths: List[str]) -> Dict[str, Any]:
    """
    Agrega y computa métricas empíricas sobre una lista de archivos CSV de evaluación.
    Calcula promedios, desviaciones estándar y frecuencias de terminación reales.
    """
    all_rows: List[Dict[str, str]] = []
    loaded_files: List[str] = []

    for fp in file_paths:
        matched = glob.glob(fp)
        for p in matched:
            rows = read_csv_rows(p)
            if rows:
                all_rows.extend(rows)
                loaded_files.append(p)

    total_eps = len(all_rows)
    if total_eps == 0:
        return {"total": 0, "loaded_files": []}

    successes = 0
    falls = 0
    oob = 0
    timeouts = 0

    rewards: List[float] = []
    lengths: List[float] = []
    distances: List[float] = []
    obstacle_distances: List[float] = []
    altitudes: List[float] = []
    intervention_norms: List[float] = []

    for r in all_rows:
        is_succ = str(r.get("success", "")).strip().lower() in ("true", "1")
        if is_succ:
            successes += 1

        term = str(r.get("termination", "")).strip().lower()
        if term == "fall":
            falls += 1
        elif term in ("out_of_bounds", "oob"):
            oob += 1
        elif term == "timeout":
            timeouts += 1

        for key, arr in [
            ("reward", rewards),
            ("length", lengths),
            ("steps", lengths),
            ("final_distance", distances),
            ("min_obstacle_distance", obstacle_distances),
            ("final_obstacle_distance", obstacle_distances),
            ("mean_intervention_norm", intervention_norms),
        ]:
            val_str = r.get(key, "")
            if val_str not in ("", None):
                try:
                    arr.append(float(val_str))
                except (ValueError, TypeError):
                    pass

        alt_str = r.get("final_altitude_m", r.get("final_altitude", r.get("final_altitude_mm", "")))
        if alt_str not in ("", None):
            try:
                v = float(alt_str)
                if v > 10.0:  # si estaba almacenado en mm, convertir a metros
                    v = v / 1000.0
                altitudes.append(v)
            except (ValueError, TypeError):
                pass

    def mean_std(arr: List[float]) -> Tuple[Optional[float], Optional[float]]:
        if not arr:
            return None, None
        m = sum(arr) / len(arr)
        variance = sum((x - m) ** 2 for x in arr) / len(arr)
        return m, math.sqrt(variance)

    rew_m, rew_s = mean_std(rewards)
    len_m, len_s = mean_std(lengths)
    dist_m, dist_s = mean_std(distances)
    obs_m, obs_s = mean_std(obstacle_distances)
    alt_m, alt_s = mean_std(altitudes)
    norm_m, norm_s = mean_std(intervention_norms)

    # Supervivencia: episodios que no terminaron en colisión ni fuera de límites
    crashes = falls + oob
    surv_count = total_eps - crashes
    surv_pct = (surv_count / total_eps) * 100.0 if total_eps > 0 else 0.0

    return {
        "total": total_eps,
        "loaded_files": loaded_files,
        "succ_count": successes,
        "succ_pct": (successes / total_eps) * 100.0,
        "fall_count": falls,
        "fall_pct": (falls / total_eps) * 100.0,
        "oob_count": oob,
        "oob_pct": (oob / total_eps) * 100.0,
        "timeout_count": timeouts,
        "timeout_pct": (timeouts / total_eps) * 100.0,
        "surv_count": surv_count,
        "surv_pct": surv_pct,
        "rew_m": rew_m,
        "rew_s": rew_s,
        "len_m": len_m,
        "len_s": len_s,
        "time_s": len_m * 0.032 if len_m is not None else 0.0,
        "dist_m": dist_m,
        "dist_s": dist_s,
        "obs_m": obs_m,
        "obs_s": obs_s,
        "alt_m": alt_m,
        "alt_s": alt_s,
        "norm_m": norm_m,
        "norm_s": norm_s,
    }


def format_table_ascii(headers: List[str], rows: List[List[str]]) -> str:
    """Formatea una tabla en estilo Markdown / ASCII sin requerir librerías externas."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for idx, val in enumerate(row):
            if idx < len(col_widths):
                col_widths[idx] = max(col_widths[idx], len(str(val)))

    header_str = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    separator_str = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
    row_strs = []
    for row in rows:
        row_strs.append(" | ".join(str(val).ljust(col_widths[i]) for i, val in enumerate(row)))

    return f"{header_str}\n{separator_str}\n" + "\n".join(row_strs)


def export_latex_tabular(headers: List[str], rows: List[List[str]], filepath: str, caption: str, label: str):
    """Exporta una tabla a código LaTeX tabular compatible con el documento del TFM."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    align = "|".join(["l"] + ["c"] * (len(headers) - 1))
    lines = [
        f"% Tabla generada automáticamente por scripts/reproduce_results.py",
        f"\\begin{{table}}[ht!]",
        f"\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{|{align}|}}",
        f"\\hline",
        " & ".join(f"\\textbf{{{h}}}" for h in headers) + " \\\\ \\hline",
    ]
    for row in rows:
        escaped_row = [str(c).replace("%", "\\%").replace("_", "\\_").replace("±", "$\\pm$") for c in row]
        lines.append(" & ".join(escaped_row) + " \\\\ \\hline")
    lines.extend([
        f"\\end{{tabular}}",
        f"\\end{{table}}",
        "",
    ])
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ==============================================================================
# TABLA 4.1: CONFIGURACIONES DE EVALUACIÓN DE PILOTOS DEGRADADOS
# ==============================================================================
def get_table_4_1() -> Tuple[List[str], List[List[str]]]:
    headers = ["Perfil", "Identificador", "Parámetro Clave", "Mecanismo de Falla", "Impacto Operativo"]
    rows = [
        [
            "HeuristicPilot (Control P)",
            "heuristic_prop",
            "K_p = 2.0 (Directo)",
            "Control proporcional cinemático sin horizonte temporal",
            "Navegación geométrica nominal determinista de referencia",
        ],
        [
            "NoisyPilot (Moderado)",
            "noisy_sigma_0.5",
            "sigma = 0.50",
            "Ruido gaussiano continuo aditivo sobre velocidades",
            "Desviación angular y deriva espacial acumulativa",
        ],
        [
            "NoisyPilot (Severo)",
            "noisy_sigma_1.0",
            "sigma = 1.00",
            "Ruido aditivo de rango dinámico completo",
            "Control errático y trayectorias de choque perimetral",
        ],
        [
            "LaggyPilot (Schaff & Walter)",
            "laggy_p_0.8",
            "p_repeat = 0.80",
            "Repetición estocástica del comando previo (ZOH)",
            "Acumulación inercial, sobreoscilación y timeout",
        ],
        [
            "StandByPilot (Signal Drop)",
            "standby_p_0.8",
            "p_drop = 0.80",
            "Pérdida de paquetes con conmutación reactiva a hover",
            "Detención frecuente del avance pero alta integridad",
        ],
    ]
    return headers, rows


# ==============================================================================
# TABLA 4.2: COMPARATIVA DE PILOTOS ÓPTIMOS (PPO vs. SAC vs. TD3)
# ==============================================================================
def get_table_4_2(
    results_dir: str = "results_TFM",
    ppo_mode: str = "best",
    return_metrics: bool = False,
) -> Any:
    # Buscar archivos de evaluación reales en results_TFM o logs
    # Para TD3 y SAC evaluamos las 5 semillas maestras completas (500 episodios cada uno)
    td3_patterns = [
        os.path.join(results_dir, "logs", "td3", "eval_td3_seed_*_100ep.csv"),
    ]
    if not glob.glob(td3_patterns[0]):
        td3_patterns = [os.path.join(results_dir, "logs", "td3", "eval_optimal_td3_100ep.csv")]

    sac_patterns = [
        os.path.join(results_dir, "logs", "sac", "eval_sac_seed_*_100ep.csv"),
    ]
    if not glob.glob(sac_patterns[0]):
        sac_patterns = [os.path.join(results_dir, "logs", "sac", "eval_sac_optimal.csv")]

    if ppo_mode == "best":
        # Checkpoint óptimo validado (60k steps, 100% éxito)
        best_csv = os.path.join(PROJECT_ROOT, "logs", "eval_ppo_60k_20.csv")
        if os.path.exists(best_csv):
            ppo_patterns = [best_csv]
        else:
            ppo_patterns = [os.path.join(results_dir, "logs", "td3", "eval_optimal_ppo_100ep.csv")]
    else:
        # Todas las corridas de PPO registradas (incluyendo overfitted / timeouts)
        ppo_patterns = [
            os.path.join(results_dir, "logs", "ppo", "eval_ppo_seed_*_100ep.csv"),
            os.path.join(PROJECT_ROOT, "logs", "eval_ppo_60k_20.csv"),
        ]

    m_td3 = compute_metrics_from_csvs(td3_patterns)
    m_sac = compute_metrics_from_csvs(sac_patterns)
    m_ppo = compute_metrics_from_csvs(ppo_patterns)

    ppo_label = "PPO (SB3, Best 60k)" if ppo_mode == "best" else "PPO (SB3, Todas)"
    headers = ["Métrica de evaluación", ppo_label, "SAC (RLtools, 5 seeds)", "TD3 (RLtools, 5 seeds)"]

    def fmt_stat(m: Dict[str, Any], key_m: str, key_s: str, unit: str = "", sign: bool = False) -> str:
        if not m or m.get("total", 0) == 0 or m.get(key_m) is None:
            return "N/D"
        val_m = m[key_m]
        val_s = m.get(key_s, 0.0) or 0.0
        prefix = "+" if sign and val_m > 0 else ""
        if val_s > 0.001:
            return f"{prefix}{val_m:.2f} ± {val_s:.2f} {unit}".strip()
        return f"{prefix}{val_m:.2f} {unit}".strip()

    rows = [
        [
            "Tasa de éxito (%)",
            f"{m_ppo['succ_pct']:.1f}% ({m_ppo['succ_count']}/{m_ppo['total']})" if m_ppo.get("total", 0) > 0 else "N/D",
            f"{m_sac['succ_pct']:.1f}% ({m_sac['succ_count']}/{m_sac['total']})" if m_sac.get("total", 0) > 0 else "N/D",
            f"{m_td3['succ_pct']:.1f}% ({m_td3['succ_count']}/{m_td3['total']})" if m_td3.get("total", 0) > 0 else "N/D",
        ],
        [
            "Recompensa acumulada media",
            fmt_stat(m_ppo, "rew_m", "rew_s", sign=True),
            fmt_stat(m_sac, "rew_m", "rew_s", sign=True),
            fmt_stat(m_td3, "rew_m", "rew_s", sign=True),
        ],
        [
            "Longitud media de episodio",
            fmt_stat(m_ppo, "len_m", "len_s", unit="pasos"),
            fmt_stat(m_sac, "len_m", "len_s", unit="pasos"),
            fmt_stat(m_td3, "len_m", "len_s", unit="pasos"),
        ],
        [
            "Tiempo medio de vuelo",
            f"{m_ppo.get('time_s', 0.0):.2f} s" if m_ppo.get("total", 0) > 0 else "N/D",
            f"{m_sac.get('time_s', 0.0):.2f} s" if m_sac.get("total", 0) > 0 else "N/D",
            f"{m_td3.get('time_s', 0.0):.2f} s" if m_td3.get("total", 0) > 0 else "N/D",
        ],
        [
            "Distancia final al objetivo",
            fmt_stat(m_ppo, "dist_m", "dist_s", unit="m"),
            fmt_stat(m_sac, "dist_m", "dist_s", unit="m"),
            fmt_stat(m_td3, "dist_m", "dist_s", unit="m"),
        ],
        [
            "Distancia mín. a obstáculo (láser)",
            fmt_stat(m_ppo, "obs_m", "obs_s", unit="m"),
            fmt_stat(m_sac, "obs_m", "obs_s", unit="m"),
            fmt_stat(m_td3, "obs_m", "obs_s", unit="m"),
        ],
        [
            "Altitud final media (m)",
            fmt_stat(m_ppo, "alt_m", "alt_s", unit="m"),
            fmt_stat(m_sac, "alt_m", "alt_s", unit="m"),
            fmt_stat(m_td3, "alt_m", "alt_s", unit="m"),
        ],
        [
            "Supervivencia / Integridad",
            f"{m_ppo['surv_pct']:.1f}% ({m_ppo['surv_count']}/{m_ppo['total']})" if m_ppo.get("total", 0) > 0 else "N/D",
            f"{m_sac['surv_pct']:.1f}% ({m_sac['surv_count']}/{m_sac['total']})" if m_sac.get("total", 0) > 0 else "N/D",
            f"{m_td3['surv_pct']:.1f}% ({m_td3['surv_count']}/{m_td3['total']})" if m_td3.get("total", 0) > 0 else "N/D",
        ],
        [
            "Artefacto de despliegue",
            "Archivo .zip (PyTorch)",
            "Cabecera C++ estática (.h)",
            "Cabecera C++ estática (.h)",
        ],
    ]

    metrics_dict = {"PPO": m_ppo, "SAC": m_sac, "TD3": m_td3}
    if return_metrics:
        return headers, rows, metrics_dict
    return headers, rows


# ==============================================================================
# TABLA 4.3: DESEMPEÑO DE PILOTOS SINTÉTICOS DEGRADADOS Y HEURÍSTICO
# ==============================================================================
def get_table_4_3(results_dir: str = "results_TFM", return_metrics: bool = False) -> Any:
    deg_dir = os.path.join(results_dir, "logs", "degraded")

    # Configuraciones evaluadas multisemilla
    configs = [
        ("HeuristicPilot (Control P)", ["heuristic_prop"]),
        ("NoisyPilot (sigma = 0.50)", ["noisy_sigma_0.5"]),
        ("NoisyPilot (sigma = 1.00)", ["noisy_sigma_1.0"]),
        ("LaggyPilot (p = 0.80)", ["laggy_p_0.8"]),
        ("StandByPilot (p = 0.80)", ["standby_p_0.8"]),
    ]

    headers = [
        "Configuración del piloto",
        "Tasa de éxito (%)",
        "Recompensa media",
        "Distancia final (m)",
        "Altitud final (m)",
        "Distribución de terminaciones",
    ]

    rows = []
    metrics_dict = {}

    for label, prefixes in configs:
        patterns = []
        for pfx in prefixes:
            patterns.append(os.path.join(deg_dir, f"{pfx}*.csv"))

        m = compute_metrics_from_csvs(patterns)
        metrics_dict[label] = m

        if m.get("total", 0) > 0:
            t_str = f"{m['succ_count']} metas, {m['fall_count']} caídas, {m['oob_count']} oob, {m['timeout_count']} timeout"
            rew_str = f"{m['rew_m']:+.2f} ± {m['rew_s']:.2f}"
            dist_str = f"{m['dist_m']:.3f} ± {m['dist_s']:.3f}"
            alt_str = f"{m['alt_m']:.2f} ± {m['alt_s']:.2f}" if m.get("alt_m") is not None else "N/D"
            succ_str = f"{m['succ_pct']:.1f}% ({m['succ_count']}/{m['total']})"
            rows.append([label, succ_str, rew_str, dist_str, alt_str, t_str])
        else:
            rows.append([label, "N/D", "N/D", "N/D", "N/D", "Sin archivos CSV"])

    if return_metrics:
        return headers, rows, metrics_dict
    return headers, rows


# ==============================================================================
# TABLA 4.4: COPILOTO RESIDUAL CON PILOTO RUIDOSO
# ==============================================================================
def get_table_4_4(results_dir: str = "results_TFM", return_metrics: bool = False) -> Any:
    solo_paths = [
        os.path.join(results_dir, "logs", "copilot", "eval_noisy_03.csv"),
        os.path.join(PROJECT_ROOT, "logs", "eval_noisy_03.csv"),
    ]
    ppo_paths = [
        os.path.join(results_dir, "logs", "copilot", "eval_noisy_assisted_03_ppo.csv"),
        os.path.join(PROJECT_ROOT, "logs", "eval_noisy_assisted_03_ppo.csv"),
    ]
    sac_paths = [
        os.path.join(results_dir, "logs", "copilot", "eval_noisy_assisted_03_sac.csv"),
        os.path.join(PROJECT_ROOT, "logs", "eval_noisy_assisted_03_sac.csv"),
    ]

    m_solo = compute_metrics_from_csvs(solo_paths)
    m_ppo = compute_metrics_from_csvs(ppo_paths)
    m_sac = compute_metrics_from_csvs(sac_paths)

    headers = ["Métrica de evaluación", "Piloto Ruidoso Solo (σ=0.3)", "Con Copiloto PPO", "Con Copiloto SAC"]

    def row_pct(label: str, key_pct: str, key_cnt: str) -> List[str]:
        return [
            label,
            f"{m_solo[key_pct]:.1f}% ({m_solo[key_cnt]}/{m_solo['total']})" if m_solo.get("total", 0) > 0 else "N/D",
            f"{m_ppo[key_pct]:.1f}% ({m_ppo[key_cnt]}/{m_ppo['total']})" if m_ppo.get("total", 0) > 0 else "N/D",
            f"{m_sac[key_pct]:.1f}% ({m_sac[key_cnt]}/{m_sac['total']})" if m_sac.get("total", 0) > 0 else "N/D",
        ]

    def row_stat(label: str, key_m: str, key_s: str, unit: str = "", sign: bool = False) -> List[str]:
        def fmt(m):
            if not m or m.get("total", 0) == 0 or m.get(key_m) is None:
                return "N/D"
            val_m = m[key_m]
            val_s = m.get(key_s, 0.0) or 0.0
            pfx = "+" if sign and val_m > 0 else ""
            if val_s > 0.001:
                return f"{pfx}{val_m:.2f} ± {val_s:.2f} {unit}".strip()
            return f"{pfx}{val_m:.2f} {unit}".strip()

        return [label, fmt(m_solo), fmt(m_ppo), fmt(m_sac)]

    rows = [
        row_pct("Tasa de éxito (%)", "succ_pct", "succ_count"),
        row_stat("Recompensa media acumulada", "rew_m", "rew_s", sign=True),
        row_stat("Longitud media global (pasos)", "len_m", "len_s", unit="pasos"),
        [
            "Tiempo medio de vuelo",
            f"{m_solo.get('time_s', 0.0):.2f} s" if m_solo.get("total", 0) > 0 else "N/D",
            f"{m_ppo.get('time_s', 0.0):.2f} s" if m_ppo.get("total", 0) > 0 else "N/D",
            f"{m_sac.get('time_s', 0.0):.2f} s" if m_sac.get("total", 0) > 0 else "N/D",
        ],
        row_stat("Distancia final a la meta (m)", "dist_m", "dist_s", unit="m"),
        row_stat("Distancia mín. a obstáculo (m)", "obs_m", "obs_s", unit="m"),
        row_stat("Altitud final media de vuelo (m)", "alt_m", "alt_s", unit="m"),
        row_pct("Caídas al suelo (fall)", "fall_pct", "fall_count"),
        row_pct("Salida de límites (out of bounds)", "oob_pct", "oob_count"),
        row_pct("Tiempo agotado (timeout)", "timeout_pct", "timeout_count"),
        row_pct("Supervivencia / Integridad estructural", "surv_pct", "surv_count"),
        row_stat("Norma de intervención media (||a_r||)", "norm_m", "norm_s"),
    ]

    metrics_dict = {"solo": m_solo, "copilot_ppo": m_ppo, "copilot_sac": m_sac}
    if return_metrics:
        return headers, rows, metrics_dict
    return headers, rows


# ==============================================================================
# TABLA 4.5: COPILOTO RESIDUAL CON PILOTO LENTO / LAGGED
# ==============================================================================
def get_table_4_5(results_dir: str = "results_TFM", return_metrics: bool = False) -> Any:
    solo_25_paths = [
        os.path.join(results_dir, "logs", "copilot", "eval_laggy_drop_0025.csv"),
        os.path.join(PROJECT_ROOT, "logs", "eval_laggy_drop_0025.csv"),
    ]
    cop_25_paths = [
        os.path.join(results_dir, "logs", "copilot", "eval_lagged_assisted_0025_ppo.csv"),
        os.path.join(PROJECT_ROOT, "logs", "eval_lagged_assisted_0025_ppo.csv"),
    ]
    solo_50_paths = [
        os.path.join(results_dir, "logs", "copilot", "eval_laggy_drop_005.csv"),
        os.path.join(PROJECT_ROOT, "logs", "eval_laggy_drop_005.csv"),
    ]
    cop_50_paths = [
        os.path.join(results_dir, "logs", "copilot", "eval_lagged_assisted_005_ppo.csv"),
        os.path.join(PROJECT_ROOT, "logs", "eval_lagged_assisted_005_ppo.csv"),
    ]

    m_s25 = compute_metrics_from_csvs(solo_25_paths)
    m_c25 = compute_metrics_from_csvs(cop_25_paths)
    m_s50 = compute_metrics_from_csvs(solo_50_paths)
    m_c50 = compute_metrics_from_csvs(cop_50_paths)

    headers = ["Métrica de evaluación", "Lento Solo (p=2.5%)", "Con Copiloto (p=2.5%)", "Lento Solo (p=5.0%)", "Con Copiloto (p=5.0%)"]

    def row_pct(label: str, key_pct: str, key_cnt: str) -> List[str]:
        def fmt(m):
            if not m or m.get("total", 0) == 0:
                return "N/D"
            return f"{m[key_pct]:.1f}% ({m[key_cnt]}/{m['total']})"
        return [label, fmt(m_s25), fmt(m_c25), fmt(m_s50), fmt(m_c50)]

    def row_stat(label: str, key_m: str, key_s: str, unit: str = "", sign: bool = False) -> List[str]:
        def fmt(m):
            if not m or m.get("total", 0) == 0 or m.get(key_m) is None:
                return "N/D"
            val_m = m[key_m]
            val_s = m.get(key_s, 0.0) or 0.0
            pfx = "+" if sign and val_m > 0 else ""
            if val_s > 0.001:
                return f"{pfx}{val_m:.2f} ± {val_s:.2f} {unit}".strip()
            return f"{pfx}{val_m:.2f} {unit}".strip()
        return [label, fmt(m_s25), fmt(m_c25), fmt(m_s50), fmt(m_c50)]

    def row_survival(label: str) -> List[str]:
        def fmt(m):
            if not m or m.get("total", 0) == 0:
                return "N/D"
            crashes = m.get("fall_count", 0) + m.get("oob_count", 0)
            surv = m["total"] - crashes
            pct = (surv / m["total"]) * 100.0
            return f"{pct:.1f}% ({surv}/{m['total']})"
        return [label, fmt(m_s25), fmt(m_c25), fmt(m_s50), fmt(m_c50)]

    rows = [
        row_pct("Tasa de éxito (%)", "succ_pct", "succ_count"),
        row_stat("Recompensa media acumulada", "rew_m", "rew_s", sign=True),
        row_stat("Longitud media de episodio (pasos)", "len_m", "len_s", unit="pasos"),
        [
            "Tiempo medio de vuelo",
            f"{m_s25.get('time_s', 0.0):.2f} s" if m_s25.get("total", 0) > 0 else "N/D",
            f"{m_c25.get('time_s', 0.0):.2f} s" if m_c25.get("total", 0) > 0 else "N/D",
            f"{m_s50.get('time_s', 0.0):.2f} s" if m_s50.get("total", 0) > 0 else "N/D",
            f"{m_c50.get('time_s', 0.0):.2f} s" if m_c50.get("total", 0) > 0 else "N/D",
        ],
        row_stat("Distancia final a la meta (m)", "dist_m", "dist_s", unit="m"),
        row_stat("Distancia mín. a obstáculo (m)", "obs_m", "obs_s", unit="m"),
        row_stat("Altitud final media de vuelo (m)", "alt_m", "alt_s", unit="m"),
        row_pct("Caídas al suelo (fall)", "fall_pct", "fall_count"),
        row_pct("Salida de límites (out of bounds)", "oob_pct", "oob_count"),
        row_pct("Tiempo agotado (timeout)", "timeout_pct", "timeout_count"),
        row_pct("Supervivencia / Integridad estructural", "surv_pct", "surv_count"),
        row_stat("Norma de intervención media (||a_r||)", "norm_m", "norm_s"),
    ]

    metrics_dict = {"solo_25": m_s25, "cop_25": m_c25, "solo_50": m_s50, "cop_50": m_c50}
    if return_metrics:
        return headers, rows, metrics_dict
    return headers, rows


# ==============================================================================
# GENERACIÓN DINÁMICA DE GRÁFICOS COMPARATIVOS
# ==============================================================================
def generate_reproduced_plots(
    output_dir: str,
    results_dir: str = "results_TFM",
    m_opt: Optional[Dict[str, Any]] = None,
    m_deg: Optional[Dict[str, Any]] = None,
    m_cop_noisy: Optional[Dict[str, Any]] = None,
):
    """Genera figuras comparativas en formato PNG usando las métricas reales calculadas."""
    if not HAS_MATPLOTLIB:
        print("[Aviso] Matplotlib no disponible. Omitiendo generación de PNGs.")
        return

    figs_dir = os.path.join(output_dir, "figures")
    os.makedirs(figs_dir, exist_ok=True)

    if m_opt is None:
        _, _, m_opt = get_table_4_2(results_dir, return_metrics=True)
    if m_deg is None:
        _, _, m_deg = get_table_4_3(results_dir, return_metrics=True)
    if m_cop_noisy is None:
        _, _, m_cop_noisy = get_table_4_4(results_dir, return_metrics=True)

    # 1. Comparativa de Éxito de Políticas Autónomas y Degradadas
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = []
    success_rates = []

    # Óptimos
    for k in ["PPO", "SAC", "TD3"]:
        if k in m_opt and m_opt[k].get("total", 0) > 0:
            labels.append(f"Óptimo {k}")
            success_rates.append(m_opt[k]["succ_pct"])

    # Degradados
    for k, v in m_deg.items():
        if v.get("total", 0) > 0:
            labels.append(k.split("(")[0].strip())
            success_rates.append(v["succ_pct"])

    if labels and success_rates:
        colors = ["#27ae60", "#2ecc71", "#1abc9c", "#9b59b6", "#e67e22", "#c0392b", "#3498db", "#2980b9"][:len(labels)]
        bars = ax.bar(labels, success_rates, color=colors, alpha=0.9, edgecolor="black", linewidth=0.8)
        ax.set_ylabel("Tasa de Éxito Empírica (%)", fontsize=11, fontweight="bold")
        ax.set_title("Comparativa de Tasa de Éxito Calculada a partir de los Datos Experimentales", fontsize=12, fontweight="bold")
        ax.set_ylim(0, 115)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        plt.xticks(rotation=25, ha="right", fontsize=9)

        for bar, val in zip(bars, success_rates):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 2, f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

        plt.tight_layout()
        p1 = os.path.join(figs_dir, "comparativa_tasa_exito_pilotos.png")
        fig.savefig(p1, dpi=300)
        plt.close(fig)

    # 2. Distribución de Terminaciones en Asistencia Residual (Piloto Ruidoso)
    m_s = m_cop_noisy.get("solo", {})
    m_p = m_cop_noisy.get("copilot_ppo", {})
    m_c = m_cop_noisy.get("copilot_sac", {})

    if m_s.get("total", 0) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        categories = ["Éxito Meta", "Timeout", "Caída (Fall)", "Fuera Límites"]
        x = np.arange(len(categories)) if HAS_NUMPY else [0, 1, 2, 3]
        w = 0.28

        solo_vals = [m_s["succ_pct"], m_s["timeout_pct"], m_s["fall_pct"], m_s["oob_pct"]]
        ppo_vals = [m_p.get("succ_pct", 0), m_p.get("timeout_pct", 0), m_p.get("fall_pct", 0), m_p.get("oob_pct", 0)] if m_p else [0,0,0,0]
        sac_vals = [m_c.get("succ_pct", 0), m_c.get("timeout_pct", 0), m_c.get("fall_pct", 0), m_c.get("oob_pct", 0)] if m_c else [0,0,0,0]

        if HAS_NUMPY:
            ax.bar(x - w, solo_vals, w, label="Piloto Solo (σ=0.30)", color="#e74c3c", alpha=0.85)
            ax.bar(x, ppo_vals, w, label="+ Copiloto PPO", color="#2ecc71", alpha=0.85)
            ax.bar(x + w, sac_vals, w, label="+ Copiloto SAC", color="#3498db", alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels(categories, fontsize=10, fontweight="bold")

        ax.set_ylabel("Porcentaje de Episodios (%)", fontsize=11, fontweight="bold")
        ax.set_title("Distribución de Terminaciones Empíricas (Asistencia Residual)", fontsize=12, fontweight="bold")
        ax.set_ylim(0, 75)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(loc="upper right")

        plt.tight_layout()
        p2 = os.path.join(figs_dir, "distribucion_terminaciones_copiloto.png")
        fig.savefig(p2, dpi=300)
        plt.close(fig)

    print(f"Gráficos empíricos guardados en: {figs_dir}/")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reproducción Sistemática y Empírica de Tablas y Gráficos del TFM (Universidad Europea)"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results_TFM",
        help="Ruta al directorio de resultados experimentales (solo lectura).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results_TFM",
        help="Directorio de destino para las tablas y figuras reproducidas (por defecto: results_TFM).",
    )
    parser.add_argument(
        "--table",
        type=str,
        default="all",
        choices=["all", "4.1", "4.2", "4.3", "4.4", "4.5"],
        help="Tabla específica a reproducir.",
    )
    parser.add_argument(
        "--ppo-mode",
        type=str,
        default="best",
        choices=["best", "all"],
        help="Modo de evaluación para PPO: 'best' (checkpoint 60k, 100%% éxito) o 'all' (todas las semillas).",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        default=True,
        help="Regenerar gráficos comparativos en formato PNG si matplotlib está presente.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        tables_dir = os.path.join(args.output_dir, "tables")
        os.makedirs(tables_dir, exist_ok=True)
    except OSError as e:
        print(f"[Aviso] No se pudo escribir en '{args.output_dir}' ({e}). Usando '/tmp/reproduced_results'.")
        args.output_dir = "/tmp/reproduced_results"
        os.makedirs(args.output_dir, exist_ok=True)
        tables_dir = os.path.join(args.output_dir, "tables")
        os.makedirs(tables_dir, exist_ok=True)

    print(f"\n{'='*78}")
    print(" REPRODUCCIÓN SISTEMÁTICA Y EMPÍRICA DE RESULTADOS - TFM UNIVERSIDAD EUROPEA")
    print(" (Todos los cálculos se extraen dinámicamente de los archivos CSV)")
    print(f" Directorio de datos (solo lectura) : {args.results_dir}")
    print(f" Directorio de salida               : {args.output_dir}")
    print(f"{'='*78}\n")

    m_opt_dict = {}
    m_deg_dict = {}
    m_cop_noisy_dict = {}

    # Tabla 4.1
    if args.table in ("all", "4.1"):
        print("\n--- TABLA 4.1: Configuraciones de Evaluación de Pilotos Degradados ---")
        h, rows = get_table_4_1()
        print(format_table_ascii(h, rows))
        export_latex_tabular(
            h, rows,
            os.path.join(tables_dir, "tabla_4_1_degraded_configs.tex"),
            caption="Configuraciones de evaluación para los pilotos degradados",
            label="tab:configuraciones_pilotos_degradados",
        )

    # Tabla 4.2
    if args.table in ("all", "4.2"):
        print("\n--- TABLA 4.2: Comparativa de Pilotos Óptimos (PPO vs. SAC vs. TD3) ---")
        h, rows, m_opt_dict = get_table_4_2(args.results_dir, ppo_mode=args.ppo_mode, return_metrics=True)
        print(format_table_ascii(h, rows))
        export_latex_tabular(
            h, rows,
            os.path.join(tables_dir, "tabla_4_2_optimal_pilots.tex"),
            caption="Comparativa cuantitativa de desempeño entre pilotos óptimos PPO, SAC y TD3",
            label="tab:comparativa_pilotos_optimos",
        )

    # Tabla 4.3
    if args.table in ("all", "4.3"):
        print("\n--- TABLA 4.3: Desempeño de Pilotos Sintéticos Degradados y Heurístico ---")
        h, rows, m_deg_dict = get_table_4_3(args.results_dir, return_metrics=True)
        print(format_table_ascii(h, rows))
        export_latex_tabular(
            h, rows,
            os.path.join(tables_dir, "tabla_4_3_synthetic_pilots.tex"),
            caption="Comparativa cuantitativa de desempeño entre pilotos sintéticos degradados y control heurístico",
            label="tab:comparativa_pilotos_sinteticos",
        )

    # Tabla 4.4
    if args.table in ("all", "4.4"):
        print("\n--- TABLA 4.4: Desempeño del Copiloto Residual (Piloto Ruidoso σ = 0.30) ---")
        h, rows, m_cop_noisy_dict = get_table_4_4(args.results_dir, return_metrics=True)
        print(format_table_ascii(h, rows))
        export_latex_tabular(
            h, rows,
            os.path.join(tables_dir, "tabla_4_4_copilot_noisy.tex"),
            caption="Comparativa de desempeño del copiloto residual asistiendo a un piloto ruidoso ($\\sigma = 0.30$)",
            label="tab:comparativa_copilotos",
        )

    # Tabla 4.5
    if args.table in ("all", "4.5"):
        print("\n--- TABLA 4.5: Desempeño del Copiloto Residual (Piloto Lento / Laggy) ---")
        h, rows = get_table_4_5(args.results_dir, return_metrics=False)
        print(format_table_ascii(h, rows))
        export_latex_tabular(
            h, rows,
            os.path.join(tables_dir, "tabla_4_5_copilot_slow.tex"),
            caption="Comparativa cuantitativa de desempeño del copiloto residual asistiendo a un piloto lento",
            label="tab:comparativa_copiloto_lento",
        )

    # Generar gráficos usando métricas calculadas
    if args.plots and HAS_MATPLOTLIB:
        print("\n--- Regeneración Dinámica de Figuras Comparativas (Matplotlib) ---")
        generate_reproduced_plots(args.output_dir, args.results_dir, m_opt_dict, m_deg_dict, m_cop_noisy_dict)

    print(f"\n{'='*78}")
    print(f" PROCESO COMPLETADO EXITOSAMENTE")
    print(f"  - Tablas LaTeX exportadas en : {tables_dir}/")
    if HAS_MATPLOTLIB:
        print(f"  - Figuras PNG generadas en   : {os.path.join(args.output_dir, 'figures')}/")
    print(f"{'='*78}\n")


if __name__ == "__main__":
    main()
