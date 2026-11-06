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
import re
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


def compute_wilson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calcula el intervalo de confianza de Wilson para una proporción binomial (k éxitos en n ensayos).
    Proporciona cotas asimétricas rigurosas en [0.0, 100.0]%, evitando el colapso de la aproximación
    normal Wald cuando p = 0 o p = 1.
    """
    if n <= 0:
        return 0.0, 0.0
    z = 1.95996  # 95% bilateral
    p = float(k) / float(n)
    denom = 1.0 + (z**2) / float(n)
    center = (p + (z**2) / (2.0 * float(n))) / denom
    spread = (z * math.sqrt(max(0.0, p * (1.0 - p) / float(n) + (z**2) / (4.0 * (float(n)**2))))) / denom
    lower = max(0.0, center - spread) * 100.0
    upper = min(1.0, center + spread) * 100.0
    return round(lower, 2), round(upper, 2)


def compute_metrics_from_rows(all_rows: List[Dict[str, str]], loaded_files: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Agrega y computa métricas empíricas sobre una lista de filas de evaluación (diccionarios).
    Calcula promedios, desviaciones estándar, frecuencias de terminación e intervalos de Wilson al 95%.
    """
    if loaded_files is None:
        loaded_files = []

    total_eps = len(all_rows)
    if total_eps == 0:
        return {"total": 0, "loaded_files": loaded_files}

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
    jerks: List[float] = []

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
            ("mean_jerk", jerks),
            ("jerk", jerks),
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
    jerk_m, jerk_s = mean_std(jerks)

    # Supervivencia: episodios que no terminaron en colisión ni fuera de límites
    crashes = falls + oob
    surv_count = total_eps - crashes
    surv_pct = (surv_count / total_eps) * 100.0 if total_eps > 0 else 0.0

    succ_w_low, succ_w_high = compute_wilson_ci(successes, total_eps)
    fall_w_low, fall_w_high = compute_wilson_ci(falls, total_eps)
    oob_w_low, oob_w_high = compute_wilson_ci(oob, total_eps)
    timeout_w_low, timeout_w_high = compute_wilson_ci(timeouts, total_eps)
    surv_w_low, surv_w_high = compute_wilson_ci(surv_count, total_eps)

    return {
        "total": total_eps,
        "loaded_files": loaded_files,
        "succ_count": successes,
        "succ_pct": (successes / total_eps) * 100.0,
        "succ_w_low": succ_w_low,
        "succ_w_high": succ_w_high,
        "fall_count": falls,
        "fall_pct": (falls / total_eps) * 100.0,
        "fall_w_low": fall_w_low,
        "fall_w_high": fall_w_high,
        "oob_count": oob,
        "oob_pct": (oob / total_eps) * 100.0,
        "oob_w_low": oob_w_low,
        "oob_w_high": oob_w_high,
        "timeout_count": timeouts,
        "timeout_pct": (timeouts / total_eps) * 100.0,
        "timeout_w_low": timeout_w_low,
        "timeout_w_high": timeout_w_high,
        "surv_count": surv_count,
        "surv_pct": surv_pct,
        "surv_w_low": surv_w_low,
        "surv_w_high": surv_w_high,
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
        "jerk_m": jerk_m,
        "jerk_s": jerk_s,
    }


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

    return compute_metrics_from_rows(all_rows, loaded_files)


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


def export_spanish_csv(headers: List[str], rows: List[List[str]], filepath: str):
    """
    Exporta una tabla a formato CSV estructurado en español:
    - Campos delimitados por punto y coma (;)
    - Separador decimal con coma (,)
    - Limpieza de comandos LaTeX residuales
    - Sincronización automática de los archivos .csv hacia UEM_TFM/tables/
    - NUNCA sobreescribe los archivos .tex para proteger las configuraciones de formato del usuario.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    def clean_cell(cell: Any) -> str:
        s = str(cell)
        # Convertir punto decimal en números a coma decimal (ej. 10.87 -> 10,87)
        s = re.sub(r'(\d+)\.(\d+)', r'\1,\2', s)
        # Limpieza de comandos y notaciones LaTeX
        s = s.replace('$', '')
        s = s.replace(r'\pm', '±')
        s = s.replace(r'\textbf', '')
        s = s.replace(r'\mathrm', '')
        s = s.replace(r'\mathbf', '')
        s = s.replace(r'\|', '||')
        s = s.replace(r'\_', '_')
        s = s.replace(r'\&', '&')
        s = s.replace(r'\%', '%')
        s = s.replace(r'\sigma', 'σ')
        s = s.replace('{', '').replace('}', '')
        # Evitar colisión de separadores si algún campo contiene punto y coma
        s = s.replace(';', ',')
        return s.strip()

    csv_rows = []
    # Encabezados
    csv_rows.append(";".join(clean_cell(h) for h in headers))
    # Filas
    for row in rows:
        csv_rows.append(";".join(clean_cell(c) for c in row))

    csv_content = "\n".join(csv_rows) + "\n"

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(csv_content)

    # Sincronizar automáticamente el archivo .csv con UEM_TFM/tables si existe
    uem_tables = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "UEM_TFM", "tables"))
    if os.path.exists(uem_tables):
        uem_fp = os.path.join(uem_tables, os.path.basename(filepath))
        try:
            with open(uem_fp, "w", encoding="utf-8") as f:
                f.write(csv_content)
        except OSError:
            pass

    # Sincronizar también con results_TFM/tables en la raíz si existe
    root_res_tables = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "results_TFM", "tables"))
    if os.path.exists(root_res_tables):
        root_fp = os.path.join(root_res_tables, os.path.basename(filepath))
        try:
            with open(root_fp, "w", encoding="utf-8") as f:
                f.write(csv_content)
        except OSError:
            pass


def export_latex_tabular(headers: List[str], rows: List[List[str]], filepath: str, caption: str, label: str):
    """(Compatibilidad) Exporta tabla a LaTeX. En el nuevo flujo se utiliza export_spanish_csv."""
    pass


# ==============================================================================
# TABLA 4.1: CONFIGURACIONES DE EVALUACIÓN DE PILOTOS DEGRADADOS
# ==============================================================================
def get_table_4_1() -> Tuple[List[str], List[List[str]]]:
    headers = ["Perfil", "Identificador", "Parámetro clave", "Mecanismo de falla", "Impacto operativo"]
    rows = [
        [
            "HeuristicPilot (Control P)",
            "heuristic_prop",
            "K_p = 2.5 (Directo)",
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
    seeds_mode: str = "100ep",
    return_metrics: bool = False,
) -> Any:
    # Buscar archivos de evaluación reales en results_TFM o logs
    if seeds_mode == "100ep":
        # Evaluación simétrica de 100 episodios por modelo con semillas de evaluación distintas
        td3_patterns = [os.path.join(results_dir, "logs", "td3", "eval_td3_seed_42_100ep.csv")]
        if not glob.glob(td3_patterns[0]):
            td3_patterns = [os.path.join(results_dir, "logs", "td3", "eval_optimal_td3_100ep.csv")]

        sac_patterns = [os.path.join(results_dir, "logs", "sac", "eval_sac_seed_42_100ep.csv")]
        if not glob.glob(sac_patterns[0]):
            sac_patterns = [os.path.join(results_dir, "logs", "sac", "eval_sac_optimal.csv")]

        ppo_patterns = [os.path.join(results_dir, "logs", "ppo", "eval_ppo_seed_42_100ep.csv")]
        if not glob.glob(ppo_patterns[0]):
            ppo_patterns = [os.path.join(results_dir, "logs", "ppo", "eval_ppo_60k_100ep.csv")]

        headers = ["Métrica de evaluación", "PPO (SB3, 100 ep)", "SAC (RLtools, 100 ep)", "TD3 (RLtools, 100 ep)"]
    else:
        # Modo multi-semilla (5 semillas x 100 episodios = 500 episodios cada uno)
        td3_patterns = [os.path.join(results_dir, "logs", "td3", "eval_td3_seed_*_100ep.csv")]
        if not glob.glob(td3_patterns[0]):
            td3_patterns = [os.path.join(results_dir, "logs", "td3", "eval_optimal_td3_100ep.csv")]

        sac_patterns = [os.path.join(results_dir, "logs", "sac", "eval_sac_seed_*_100ep.csv")]
        if not glob.glob(sac_patterns[0]):
            sac_patterns = [os.path.join(results_dir, "logs", "sac", "eval_sac_optimal.csv")]

        ppo_patterns = [os.path.join(results_dir, "logs", "ppo", "eval_ppo_seed_*_100ep.csv")]
        if not glob.glob(ppo_patterns[0]):
            ppo_patterns = [os.path.join(results_dir, "logs", "ppo", "eval_ppo_seed_42_100ep.csv")]

        headers = ["Métrica de evaluación", "PPO (SB3, 5 seeds)", "SAC (RLtools, 5 seeds)", "TD3 (RLtools, 5 seeds)"]

    m_td3 = compute_metrics_from_csvs(td3_patterns)
    m_sac = compute_metrics_from_csvs(sac_patterns)
    m_ppo = compute_metrics_from_csvs(ppo_patterns)

    def fmt_pct_wilson(m: Dict[str, Any], key_pct: str, key_cnt: str, key_w_low: str, key_w_high: str) -> str:
        if not m or m.get("total", 0) == 0:
            return "N/D"
        pct = m[key_pct]
        cnt = m[key_cnt]
        tot = m["total"]
        low = m.get(key_w_low, 0.0)
        high = m.get(key_w_high, 100.0)
        return f"{pct:.1f}% [{low:.1f}, {high:.1f}]% ({cnt}/{tot})"

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
            fmt_pct_wilson(m_ppo, "succ_pct", "succ_count", "succ_w_low", "succ_w_high"),
            fmt_pct_wilson(m_sac, "succ_pct", "succ_count", "succ_w_low", "succ_w_high"),
            fmt_pct_wilson(m_td3, "succ_pct", "succ_count", "succ_w_low", "succ_w_high"),
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
            fmt_pct_wilson(m_ppo, "surv_pct", "surv_count", "surv_w_low", "surv_w_high"),
            fmt_pct_wilson(m_sac, "surv_pct", "surv_count", "surv_w_low", "surv_w_high"),
            fmt_pct_wilson(m_td3, "surv_pct", "surv_count", "surv_w_low", "surv_w_high"),
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
def get_table_4_3(
    results_dir: str = "results_TFM",
    seeds_mode: str = "100ep",
    return_metrics: bool = False,
) -> Any:
    deg_dir = os.path.join(results_dir, "logs", "degraded")

    # Configuraciones evaluadas multisemilla
    configs = [
        ("HeuristicPilot (Control P)", ["heuristic_prop"]),
        ("NoisyPilot (sigma = 0.50)", ["noisy_sigma_0.5"]),
        ("NoisyPilot (sigma = 1.00)", ["noisy_sigma_1.0"]),
        ("LaggyPilot (p_repeat = 0.80)", ["laggy_p_0.8"]),
        ("StandByPilot (p_drop = 0.80)", ["standby_p_0.8"]),
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
            if seeds_mode == "100ep":
                patterns.append(os.path.join(deg_dir, f"{pfx}_seed_42.csv"))
            else:
                patterns.append(os.path.join(deg_dir, f"{pfx}_seed_*.csv"))

        m = compute_metrics_from_csvs(patterns)
        metrics_dict[label] = m

        if m.get("total", 0) > 0:
            t_str = f"{m['succ_count']} metas, {m['fall_count']} caídas, {m['oob_count']} oob, {m['timeout_count']} timeout"
            rew_str = f"{m['rew_m']:+.2f} ± {m['rew_s']:.2f}"
            dist_str = f"{m['dist_m']:.3f} ± {m['dist_s']:.3f}"
            alt_str = f"{m['alt_m']:.2f} ± {m['alt_s']:.2f}" if m.get("alt_m") is not None else "N/D"
            succ_str = f"{m['succ_pct']:.1f}% [{m['succ_w_low']:.1f}, {m['succ_w_high']:.1f}]% ({m['succ_count']}/{m['total']})"
            rows.append([label, succ_str, rew_str, dist_str, alt_str, t_str])
        else:
            rows.append([label, "N/D", "N/D", "N/D", "N/D", "Sin archivos CSV"])

    if return_metrics:
        return headers, rows, metrics_dict
    return headers, rows


# ==============================================================================
# TABLA 4.4: COPILOTO CONSTREÑIDO RESIDUAL CON PILOTO RUIDOSO
# ==============================================================================
def get_table_4_4(
    results_dir: str = "results_TFM",
    seeds_mode: str = "100ep",
    return_metrics: bool = False,
) -> Any:
    cop_episodes_csv = os.path.join(results_dir, "logs", "copilot_eval", "eval_copilot_episodes.csv")
    rows = read_csv_rows(cop_episodes_csv) if os.path.exists(cop_episodes_csv) else []

    rows_n05_solo = [r for r in rows if r.get("pilot_key") == "noisy_0.5" and r.get("mode") == "placebo"]
    rows_n05_asst = [r for r in rows if r.get("pilot_key") == "noisy_0.5" and r.get("mode") == "assisted"]
    rows_n10_solo = [r for r in rows if r.get("pilot_key") == "noisy_1.0" and r.get("mode") == "placebo"]
    rows_n10_asst = [r for r in rows if r.get("pilot_key") == "noisy_1.0" and r.get("mode") == "assisted"]

    deg_dir = os.path.join(results_dir, "logs", "degraded")
    if (len(rows_n05_solo) < len(rows_n05_asst) or len(rows_n05_solo) == 0) and len(rows_n05_asst) > 0:
        patt = os.path.join(deg_dir, "noisy_sigma_0.5_seed_42.csv" if seeds_mode == "100ep" else "noisy_sigma_0.5_seed_*.csv")
        s_rows = []
        for p in sorted(glob.glob(patt)):
            s_rows.extend(read_csv_rows(p))
        if s_rows:
            for r in s_rows:
                r["mean_intervention_norm"] = "0.0"
                r["mean_jerk"] = "0.0"
            rows_n05_solo = s_rows

    if (len(rows_n10_solo) < len(rows_n10_asst) or len(rows_n10_solo) == 0) and len(rows_n10_asst) > 0:
        patt = os.path.join(deg_dir, "noisy_sigma_1.0_seed_42.csv" if seeds_mode == "100ep" else "noisy_sigma_1.0_seed_*.csv")
        s_rows = []
        for p in sorted(glob.glob(patt)):
            s_rows.extend(read_csv_rows(p))
        if s_rows:
            for r in s_rows:
                r["mean_intervention_norm"] = "0.0"
                r["mean_jerk"] = "0.0"
            rows_n10_solo = s_rows

    m_n05_solo = compute_metrics_from_rows(rows_n05_solo, [cop_episodes_csv])
    m_n05_asst = compute_metrics_from_rows(rows_n05_asst, [cop_episodes_csv])
    m_n10_solo = compute_metrics_from_rows(rows_n10_solo, [cop_episodes_csv])
    m_n10_asst = compute_metrics_from_rows(rows_n10_asst, [cop_episodes_csv])

    headers = [
        "Métrica de evaluación",
        "Noisy solo (σ = 0.50)",
        "Con copiloto (σ = 0.50)",
        "Noisy solo (σ = 1.00)",
        "Con copiloto (σ = 1.00)",
    ]

    def row_pct(label: str, key_pct: str, key_cnt: str, key_w_low: str, key_w_high: str) -> List[str]:
        def fmt(m):
            if not m or m.get("total", 0) == 0:
                return "N/D"
            pct = m[key_pct]
            cnt = m[key_cnt]
            tot = m["total"]
            low = m.get(key_w_low, 0.0)
            high = m.get(key_w_high, 100.0)
            return f"{pct:.1f}% [{low:.1f}, {high:.1f}]% ({cnt}/{tot})"

        return [label, fmt(m_n05_solo), fmt(m_n05_asst), fmt(m_n10_solo), fmt(m_n10_asst)]

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

        return [label, fmt(m_n05_solo), fmt(m_n05_asst), fmt(m_n10_solo), fmt(m_n10_asst)]

    rows = [
        row_pct("Tasa de éxito (%)", "succ_pct", "succ_count", "succ_w_low", "succ_w_high"),
        row_stat("Recompensa media acumulada", "rew_m", "rew_s", sign=True),
        row_stat("Longitud media global (pasos)", "len_m", "len_s", unit="pasos"),
        [
            "Tiempo medio de vuelo",
            f"{m_n05_solo.get('time_s', 0.0):.2f} s" if m_n05_solo.get("total", 0) > 0 else "N/D",
            f"{m_n05_asst.get('time_s', 0.0):.2f} s" if m_n05_asst.get("total", 0) > 0 else "N/D",
            f"{m_n10_solo.get('time_s', 0.0):.2f} s" if m_n10_solo.get("total", 0) > 0 else "N/D",
            f"{m_n10_asst.get('time_s', 0.0):.2f} s" if m_n10_asst.get("total", 0) > 0 else "N/D",
        ],
        row_stat("Distancia final a la meta (m)", "dist_m", "dist_s", unit="m"),
        row_pct("Caídas al suelo (fall)", "fall_pct", "fall_count", "fall_w_low", "fall_w_high"),
        row_pct("Salida de límites (out of bounds)", "oob_pct", "oob_count", "oob_w_low", "oob_w_high"),
        row_pct("Tiempo agotado (timeout)", "timeout_pct", "timeout_count", "timeout_w_low", "timeout_w_high"),
        row_pct("Supervivencia / Integridad estructural", "surv_pct", "surv_count", "surv_w_low", "surv_w_high"),
        row_stat("Norma de intervención media (||a_r||)", "norm_m", "norm_s"),
        row_stat("Suavidad de acción (Jerk medio)", "jerk_m", "jerk_s"),
    ]

    metrics_dict = {
        "noisy_05_solo": m_n05_solo,
        "noisy_05_asst": m_n05_asst,
        "noisy_10_solo": m_n10_solo,
        "noisy_10_asst": m_n10_asst,
    }
    if return_metrics:
        return headers, rows, metrics_dict
    return headers, rows


# ==============================================================================
# TABLA 4.5: COPILOTO CONSTREÑIDO RESIDUAL CON PILOTOS DE LATENCIA Y PAQUETES
# ==============================================================================
def get_table_4_5(
    results_dir: str = "results_TFM",
    seeds_mode: str = "100ep",
    return_metrics: bool = False,
) -> Any:
    cop_episodes_csv = os.path.join(results_dir, "logs", "copilot_eval", "eval_copilot_episodes.csv")
    rows = read_csv_rows(cop_episodes_csv) if os.path.exists(cop_episodes_csv) else []

    rows_lag_solo = [r for r in rows if r.get("pilot_key") == "laggy_0.8" and r.get("mode") == "placebo"]
    rows_lag_asst = [r for r in rows if r.get("pilot_key") == "laggy_0.8" and r.get("mode") == "assisted"]
    rows_std_solo = [r for r in rows if r.get("pilot_key") == "standby_0.8" and r.get("mode") == "placebo"]
    rows_std_asst = [r for r in rows if r.get("pilot_key") == "standby_0.8" and r.get("mode") == "assisted"]

    deg_dir = os.path.join(results_dir, "logs", "degraded")
    if (len(rows_lag_solo) < len(rows_lag_asst) or len(rows_lag_solo) == 0) and len(rows_lag_asst) > 0:
        patt = os.path.join(deg_dir, "laggy_p_0.8_seed_42.csv" if seeds_mode == "100ep" else "laggy_p_0.8_seed_*.csv")
        s_rows = []
        for p in sorted(glob.glob(patt)):
            s_rows.extend(read_csv_rows(p))
        if s_rows:
            for r in s_rows:
                r["mean_intervention_norm"] = "0.0"
                r["mean_jerk"] = "0.0"
            rows_lag_solo = s_rows

    if (len(rows_std_solo) < len(rows_std_asst) or len(rows_std_solo) == 0) and len(rows_std_asst) > 0:
        patt = os.path.join(deg_dir, "standby_p_0.8_seed_42.csv" if seeds_mode == "100ep" else "standby_p_0.8_seed_*.csv")
        s_rows = []
        for p in sorted(glob.glob(patt)):
            s_rows.extend(read_csv_rows(p))
        if s_rows:
            for r in s_rows:
                r["mean_intervention_norm"] = "0.0"
                r["mean_jerk"] = "0.0"
            rows_std_solo = s_rows

    m_lag_solo = compute_metrics_from_rows(rows_lag_solo, [cop_episodes_csv])
    m_lag_asst = compute_metrics_from_rows(rows_lag_asst, [cop_episodes_csv])
    m_std_solo = compute_metrics_from_rows(rows_std_solo, [cop_episodes_csv])
    m_std_asst = compute_metrics_from_rows(rows_std_asst, [cop_episodes_csv])

    headers = [
        "Métrica de evaluación",
        "Laggy solo (p_repeat = 0.80)",
        "Con copiloto (p_repeat = 0.80)",
        "StandBy solo (p_drop = 0.80)",
        "Con copiloto (p_drop = 0.80)",
    ]

    def row_pct(label: str, key_pct: str, key_cnt: str, key_w_low: str, key_w_high: str) -> List[str]:
        def fmt(m):
            if not m or m.get("total", 0) == 0:
                return "N/D"
            pct = m[key_pct]
            cnt = m[key_cnt]
            tot = m["total"]
            low = m.get(key_w_low, 0.0)
            high = m.get(key_w_high, 100.0)
            return f"{pct:.1f}% [{low:.1f}, {high:.1f}]% ({cnt}/{tot})"

        return [label, fmt(m_lag_solo), fmt(m_lag_asst), fmt(m_std_solo), fmt(m_std_asst)]

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

        return [label, fmt(m_lag_solo), fmt(m_lag_asst), fmt(m_std_solo), fmt(m_std_asst)]

    rows = [
        row_pct("Tasa de éxito (%)", "succ_pct", "succ_count", "succ_w_low", "succ_w_high"),
        row_stat("Recompensa media acumulada", "rew_m", "rew_s", sign=True),
        row_stat("Longitud media de episodio (pasos)", "len_m", "len_s", unit="pasos"),
        [
            "Tiempo medio de vuelo",
            f"{m_lag_solo.get('time_s', 0.0):.2f} s" if m_lag_solo.get("total", 0) > 0 else "N/D",
            f"{m_lag_asst.get('time_s', 0.0):.2f} s" if m_lag_asst.get("total", 0) > 0 else "N/D",
            f"{m_std_solo.get('time_s', 0.0):.2f} s" if m_std_solo.get("total", 0) > 0 else "N/D",
            f"{m_std_asst.get('time_s', 0.0):.2f} s" if m_std_asst.get("total", 0) > 0 else "N/D",
        ],
        row_stat("Distancia final a la meta (m)", "dist_m", "dist_s", unit="m"),
        row_pct("Caídas al suelo (fall)", "fall_pct", "fall_count", "fall_w_low", "fall_w_high"),
        row_pct("Salida de límites (out of bounds)", "oob_pct", "oob_count", "oob_w_low", "oob_w_high"),
        row_pct("Tiempo agotado (timeout)", "timeout_pct", "timeout_count", "timeout_w_low", "timeout_w_high"),
        row_pct("Supervivencia / Integridad estructural", "surv_pct", "surv_count", "surv_w_low", "surv_w_high"),
        row_stat("Norma de intervención media (||a_r||)", "norm_m", "norm_s"),
        row_stat("Suavidad de acción (Jerk medio)", "jerk_m", "jerk_s"),
    ]

    metrics_dict = {
        "laggy_solo": m_lag_solo,
        "laggy_asst": m_lag_asst,
        "standby_solo": m_std_solo,
        "standby_asst": m_std_asst,
    }
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
    m_cop_slow: Optional[Dict[str, Any]] = None,
):
    """Genera figuras comparativas en formato PNG usando las métricas reales calculadas."""
    if not HAS_MATPLOTLIB:
        print("[Aviso] Matplotlib no disponible. Omitiendo generación de PNGs.")
        return

    figs_dir = os.path.join(output_dir, "figures")
    os.makedirs(figs_dir, exist_ok=True)
    uem_imgs = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "UEM_TFM", "imagenes"))

    if m_opt is None:
        _, _, m_opt = get_table_4_2(results_dir, return_metrics=True)
    if m_deg is None:
        _, _, m_deg = get_table_4_3(results_dir, return_metrics=True)
    if m_cop_noisy is None:
        _, _, m_cop_noisy = get_table_4_4(results_dir, return_metrics=True)
    if m_cop_slow is None:
        _, _, m_cop_slow = get_table_4_5(results_dir, return_metrics=True)

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
    m_n05_s = m_cop_noisy.get("noisy_05_solo", {})
    m_n05_a = m_cop_noisy.get("noisy_05_asst", {})
    m_n10_s = m_cop_noisy.get("noisy_10_solo", {})
    m_n10_a = m_cop_noisy.get("noisy_10_asst", {})

    if m_n05_s.get("total", 0) > 0:
        fig, ax = plt.subplots(figsize=(9, 5))
        categories = ["Éxito Meta", "Timeout", "Caída (Fall)", "Supervivencia"]
        x = np.arange(len(categories)) if HAS_NUMPY else [0, 1, 2, 3]
        w = 0.20

        v1 = [m_n05_s.get("succ_pct", 0), m_n05_s.get("timeout_pct", 0), m_n05_s.get("fall_pct", 0), m_n05_s.get("surv_pct", 0)]
        v2 = [m_n05_a.get("succ_pct", 0), m_n05_a.get("timeout_pct", 0), m_n05_a.get("fall_pct", 0), m_n05_a.get("surv_pct", 0)]
        v3 = [m_n10_s.get("succ_pct", 0), m_n10_s.get("timeout_pct", 0), m_n10_s.get("fall_pct", 0), m_n10_s.get("surv_pct", 0)]
        v4 = [m_n10_a.get("succ_pct", 0), m_n10_a.get("timeout_pct", 0), m_n10_a.get("fall_pct", 0), m_n10_a.get("surv_pct", 0)]

        if HAS_NUMPY:
            ax.bar(x - 1.5 * w, v1, w, label="Noisy Solo (σ=0.50)", color="#e74c3c", alpha=0.85)
            ax.bar(x - 0.5 * w, v2, w, label="+ Copiloto SAC (σ=0.50)", color="#2ecc71", alpha=0.85)
            ax.bar(x + 0.5 * w, v3, w, label="Noisy Solo (σ=1.00)", color="#962d22", alpha=0.85)
            ax.bar(x + 1.5 * w, v4, w, label="+ Copiloto SAC (σ=1.00)", color="#3498db", alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels(categories, fontsize=10, fontweight="bold")

        ax.set_ylabel("Porcentaje de Episodios (%)", fontsize=11, fontweight="bold")
        ax.set_title("Distribución de Terminaciones Empíricas (Piloto Ruidoso)", fontsize=12, fontweight="bold")
        ax.set_ylim(0, 115)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(loc="upper right")

        plt.tight_layout()
        p2 = os.path.join(figs_dir, "distribucion_terminaciones_copiloto.png")
        fig.savefig(p2, dpi=300)
        plt.close(fig)

    # 3. Distribución de Terminaciones en Asistencia Residual (Laggy y StandBy)
    m_lag_s = m_cop_slow.get("laggy_solo", {})
    m_lag_a = m_cop_slow.get("laggy_asst", {})
    m_std_s = m_cop_slow.get("standby_solo", {})
    m_std_a = m_cop_slow.get("standby_asst", {})

    if m_lag_s.get("total", 0) > 0 and HAS_NUMPY:
        fig, ax = plt.subplots(figsize=(9, 5))
        categories = ["Éxito Meta", "Timeout", "Caída (Fall)", "Supervivencia"]
        x = np.arange(len(categories))
        w = 0.20

        v1 = [m_lag_s.get("succ_pct", 0), m_lag_s.get("timeout_pct", 0), m_lag_s.get("fall_pct", 0), m_lag_s.get("surv_pct", 0)]
        v2 = [m_lag_a.get("succ_pct", 0), m_lag_a.get("timeout_pct", 0), m_lag_a.get("fall_pct", 0), m_lag_a.get("surv_pct", 0)]
        v3 = [m_std_s.get("succ_pct", 0), m_std_s.get("timeout_pct", 0), m_std_s.get("fall_pct", 0), m_std_s.get("surv_pct", 0)]
        v4 = [m_std_a.get("succ_pct", 0), m_std_a.get("timeout_pct", 0), m_std_a.get("fall_pct", 0), m_std_a.get("surv_pct", 0)]

        ax.bar(x - 1.5 * w, v1, w, label="Laggy Solo (p=0.80)", color="#f39c12", alpha=0.85)
        ax.bar(x - 0.5 * w, v2, w, label="+ Copiloto SAC (Laggy)", color="#27ae60", alpha=0.85)
        ax.bar(x + 0.5 * w, v3, w, label="StandBy Solo (p=0.80)", color="#8e44ad", alpha=0.85)
        ax.bar(x + 1.5 * w, v4, w, label="+ Copiloto SAC (StandBy)", color="#2980b9", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=10, fontweight="bold")

        ax.set_ylabel("Porcentaje de Episodios (%)", fontsize=11, fontweight="bold")
        ax.set_title("Distribución de Terminaciones Empíricas (Laggy y StandBy)", fontsize=12, fontweight="bold")
        ax.set_ylim(0, 115)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(loc="upper right")

        plt.tight_layout()
        p3 = os.path.join(figs_dir, "distribucion_terminaciones_copiloto_lento.png")
        fig.savefig(p3, dpi=300)
        plt.close(fig)

    # 4. Dashboard Global Comparativo (4 paneles)
    profiles = ["Noisy (σ=0.50)", "Noisy (σ=1.00)", "Laggy (p=0.80)", "StandBy (p=0.80)"]
    succ_solo = [m_n05_s.get("succ_pct", 0), m_n10_s.get("succ_pct", 0), m_lag_s.get("succ_pct", 0), m_std_s.get("succ_pct", 0)]
    succ_asst = [m_n05_a.get("succ_pct", 0), m_n10_a.get("succ_pct", 0), m_lag_a.get("succ_pct", 0), m_std_a.get("succ_pct", 0)]
    fall_solo = [m_n05_s.get("fall_pct", 0), m_n10_s.get("fall_pct", 0), m_lag_s.get("fall_pct", 0), m_std_s.get("fall_pct", 0)]
    fall_asst = [m_n05_a.get("fall_pct", 0), m_n10_a.get("fall_pct", 0), m_lag_a.get("fall_pct", 0), m_std_a.get("fall_pct", 0)]
    norm_asst = [m_n05_a.get("norm_m", 0), m_n10_a.get("norm_m", 0), m_lag_a.get("norm_m", 0), m_std_a.get("norm_m", 0)]
    jerk_asst = [m_n05_a.get("jerk_m", 0), m_n10_a.get("jerk_m", 0), m_lag_a.get("jerk_m", 0), m_std_a.get("jerk_m", 0)]

    if HAS_NUMPY and m_n05_s.get("total", 0) > 0:
        fig, axs = plt.subplots(2, 2, figsize=(13, 9))
        xp = np.arange(len(profiles))
        bw = 0.35

        # (0, 0): Tasa de Éxito
        axs[0, 0].bar(xp - bw/2, succ_solo, bw, label="Sin Copiloto (Solo)", color="#e74c3c", alpha=0.85)
        axs[0, 0].bar(xp + bw/2, succ_asst, bw, label="Con Copiloto Residual", color="#2ecc71", alpha=0.85)
        axs[0, 0].set_title("Tasa de Éxito (%)", fontweight="bold", fontsize=11)
        axs[0, 0].set_xticks(xp)
        axs[0, 0].set_xticklabels(profiles, rotation=15, ha="right", fontsize=9)
        axs[0, 0].set_ylim(0, 115)
        axs[0, 0].grid(axis="y", linestyle="--", alpha=0.4)
        axs[0, 0].legend(fontsize=9)

        # (0, 1): Caídas al suelo
        axs[0, 1].bar(xp - bw/2, fall_solo, bw, label="Sin Copiloto (Solo)", color="#c0392b", alpha=0.85)
        axs[0, 1].bar(xp + bw/2, fall_asst, bw, label="Con Copiloto Residual", color="#27ae60", alpha=0.85)
        axs[0, 1].set_title("Tasa de Caídas al Suelo (%)", fontweight="bold", fontsize=11)
        axs[0, 1].set_xticks(xp)
        axs[0, 1].set_xticklabels(profiles, rotation=15, ha="right", fontsize=9)
        axs[0, 1].set_ylim(0, 115)
        axs[0, 1].grid(axis="y", linestyle="--", alpha=0.4)
        axs[0, 1].legend(fontsize=9)

        # (1, 0): Norma Media de Intervención
        axs[1, 0].bar(xp, norm_asst, width=0.45, color="#3498db", alpha=0.85)
        axs[1, 0].set_title(r"Norma Media de Intervención Residual $\|a_r\|_2$", fontweight="bold", fontsize=11)
        axs[1, 0].set_xticks(xp)
        axs[1, 0].set_xticklabels(profiles, rotation=15, ha="right", fontsize=9)
        axs[1, 0].set_ylim(0, 1.05)
        axs[1, 0].grid(axis="y", linestyle="--", alpha=0.4)

        # (1, 1): Suavidad Jerk
        axs[1, 1].bar(xp, jerk_asst, width=0.45, color="#9b59b6", alpha=0.85)
        axs[1, 1].set_title(r"Suavidad de Acción: Jerk Residual $\|\Delta a_r\|_2$", fontweight="bold", fontsize=11)
        axs[1, 1].set_xticks(xp)
        axs[1, 1].set_xticklabels(profiles, rotation=15, ha="right", fontsize=9)
        axs[1, 1].grid(axis="y", linestyle="--", alpha=0.4)

        plt.suptitle("Dashboard Comparativo del Copiloto Residual Constreñido (Schaff & Walter, 2020)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        p4 = os.path.join(figs_dir, "copilot_performance_dashboard.png")
        fig.savefig(p4, dpi=300)
        plt.close(fig)

        # Sincronizar automáticamente con UEM_TFM/imagenes si existe
        if os.path.exists(uem_imgs):
            for pf in ["comparativa_tasa_exito_pilotos.png", "distribucion_terminaciones_copiloto.png", "distribucion_terminaciones_copiloto_lento.png", "copilot_performance_dashboard.png"]:
                src_f = os.path.join(figs_dir, pf)
                dst_f = os.path.join(uem_imgs, pf)
                if os.path.exists(src_f):
                    try:
                        import shutil
                        shutil.copy2(src_f, dst_f)
                    except OSError:
                        pass

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
        "--seeds-mode",
        type=str,
        default="100ep",
        choices=["100ep", "5seeds"],
        help="Modo de evaluación en Tabla 4.2: '100ep' (100 episodios simétricos por modelo) o '5seeds' (5 semillas x 100 eps = 500 episodios).",
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
        export_spanish_csv(
            h, rows,
            os.path.join(tables_dir, "tabla_4_1_degraded_configs.csv"),
        )

    # Tabla 4.2
    if args.table in ("all", "4.2"):
        print("\n--- TABLA 4.2: Comparativa de Pilotos Óptimos (PPO vs. SAC vs. TD3) ---")
        h, rows, m_opt_dict = get_table_4_2(args.results_dir, ppo_mode=args.ppo_mode, seeds_mode=args.seeds_mode, return_metrics=True)
        print(format_table_ascii(h, rows))
        export_spanish_csv(
            h, rows,
            os.path.join(tables_dir, "tabla_4_2_optimal_pilots.csv"),
        )

    # Tabla 4.3
    if args.table in ("all", "4.3"):
        print("\n--- TABLA 4.3: Desempeño de Pilotos Sintéticos Degradados y Heurístico ---")
        h, rows, m_deg_dict = get_table_4_3(args.results_dir, seeds_mode=args.seeds_mode, return_metrics=True)
        print(format_table_ascii(h, rows))
        export_spanish_csv(
            h, rows,
            os.path.join(tables_dir, "tabla_4_3_synthetic_pilots.csv"),
        )

    # Tabla 4.4
    if args.table in ("all", "4.4"):
        print("\n--- TABLA 4.4: Desempeño del Copiloto Residual (Piloto Ruidoso σ = 0.50 y σ = 1.00) ---")
        h, rows, m_cop_noisy_dict = get_table_4_4(args.results_dir, seeds_mode=args.seeds_mode, return_metrics=True)
        print(format_table_ascii(h, rows))
        export_spanish_csv(
            h, rows,
            os.path.join(tables_dir, "tabla_4_4_copilot_noisy.csv"),
        )

    m_cop_slow_dict = {}

    # Tabla 4.5
    if args.table in ("all", "4.5"):
        print("\n--- TABLA 4.5: Desempeño del Copiloto Residual (Pilotos Laggy y StandBy) ---")
        h, rows, m_cop_slow_dict = get_table_4_5(args.results_dir, seeds_mode=args.seeds_mode, return_metrics=True)
        print(format_table_ascii(h, rows))
        export_spanish_csv(
            h, rows,
            os.path.join(tables_dir, "tabla_4_5_copilot_slow.csv"),
        )

    # Generar gráficos usando métricas calculadas
    if args.plots and HAS_MATPLOTLIB:
        print("\n--- Regeneración Dinámica de Figuras Comparativas (Matplotlib) ---")
        generate_reproduced_plots(args.output_dir, args.results_dir, m_opt_dict, m_deg_dict, m_cop_noisy_dict, m_cop_slow_dict)

    print(f"\n{'='*78}")
    print(f" PROCESO COMPLETADO EXITOSAMENTE")
    print(f"  - Tablas CSV exportadas en   : {tables_dir}/")
    if HAS_MATPLOTLIB:
        print(f"  - Figuras PNG generadas en   : {os.path.join(args.output_dir, 'figures')}/")
    print(f"{'='*78}\n")


if __name__ == "__main__":
    main()
