# -*- coding: utf-8 -*-
"""
Comparación y análisis cuantitativo de pilotos (Fase 6).

Genera tablas estadísticas y gráficos comparativos a partir de los archivos CSV
producidos por src/evaluation/metrics.py, replicando y extendiendo el análisis
de las tablas 4.1 a 4.4 y figuras 4.8 a 4.12 de la tesis (Diedrichs, 2025):
    - Pilotos óptimos: SB3-PPO vs. RLtools-SAC vs. Heurístico determinista.
    - Pilotos degradados vs. asistidos: NoisyPilot vs. Noisy + Copiloto residual.
    - Pilotos con latencia vs. asistidos: LaggyPilot vs. Laggy + Copiloto residual.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

# Asegurar que la raíz del proyecto esté en el PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    import matplotlib
    # Usar backend 'Agg' para entornos sin servidor gráfico (headless / Docker)
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_evaluation_csv(csv_path: str) -> pd.DataFrame:
    """
    Carga un archivo CSV de resultados generado por evaluate_pilot() en metrics.py.

    Args:
        csv_path: Ruta al archivo CSV con columnas: episode, reward, length, success, termination, final_distance.

    Returns:
        pd.DataFrame: DataFrame cargado con tipos tipificados.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No se encontró el archivo de evaluación: {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = {"episode", "reward", "length", "success", "termination", "final_distance"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"El archivo {csv_path} no contiene las columnas requeridas: {required_cols}. "
            f"Columnas encontradas: {list(df.columns)}"
        )

    df["success"] = df["success"].astype(bool)
    df["reward"] = df["reward"].astype(float)
    df["length"] = df["length"].astype(int)
    df["final_distance"] = df["final_distance"].astype(float)
    return df


def calculate_pilot_metrics(df: pd.DataFrame, label: str = "pilot") -> dict:
    """
    Calcula los estadísticos agregados para un piloto evaluado.

    Args:
        df: DataFrame de episodios.
        label: Nombre o etiqueta descriptiva del piloto/experimento.

    Returns:
        dict: Diccionario con métricas agregadas clave.
    """
    n_total = len(df)
    if n_total == 0:
        return {
            "pilot": label,
            "episodes": 0,
            "success_rate": 0.0,
            "collision_rate": 0.0,
            "timeout_rate": 0.0,
            "mean_reward": 0.0,
            "std_reward": 0.0,
            "mean_length": 0.0,
            "std_length": 0.0,
            "mean_final_distance": 0.0,
        }

    n_success = int(df["success"].sum())
    success_rate = (n_success / n_total) * 100.0

    # Colisiones: terminación por caída o fuera de límites
    collision_mask = df["termination"].isin(["fall", "out_of_bounds"])
    n_collision = int(collision_mask.sum())
    collision_rate = (n_collision / n_total) * 100.0

    # Timeouts: límite máximo de pasos excedido
    timeout_mask = df["termination"] == "timeout"
    n_timeout = int(timeout_mask.sum())
    timeout_rate = (n_timeout / n_total) * 100.0

    return {
        "pilot": label,
        "episodes": n_total,
        "success_rate": float(success_rate),
        "collision_rate": float(collision_rate),
        "timeout_rate": float(timeout_rate),
        "mean_reward": float(df["reward"].mean()),
        "std_reward": float(df["reward"].std() if n_total > 1 else 0.0),
        "mean_length": float(df["length"].mean()),
        "std_length": float(df["length"].std() if n_total > 1 else 0.0),
        "mean_final_distance": float(df["final_distance"].mean()),
    }


def compare_pilots_from_csvs(
    csv_files: list[str],
    labels: list[str] | None = None,
) -> pd.DataFrame:
    """
    Carga múltiples CSVs y genera una tabla comparativa resumida.

    Args:
        csv_files: Lista de rutas a archivos CSV.
        labels: Lista opcional de nombres para cada archivo.

    Returns:
        pd.DataFrame: Tabla resumen con una fila por cada piloto.
    """
    records = []
    for i, path in enumerate(csv_files):
        if labels is not None and i < len(labels):
            lbl = labels[i]
        else:
            base = os.path.splitext(os.path.basename(path))[0]
            lbl = base.replace("eval_", "").replace("_results", "")

        df = load_evaluation_csv(path)
        stats = calculate_pilot_metrics(df, label=lbl)
        records.append(stats)

    df_summary = pd.DataFrame(records)
    return df_summary


def generate_comparison_plots(
    df_summary: pd.DataFrame,
    output_dir: str = "logs",
) -> list[str]:
    """
    Genera gráficos comparativos de barras para tasas de éxito, colisiones y duración.

    Args:
        df_summary: DataFrame generado por compare_pilots_from_csvs().
        output_dir: Carpeta destino donde guardar los archivos de imagen.

    Returns:
        list[str]: Rutas de los gráficos PNG generados.
    """
    if not HAS_MATPLOTLIB:
        print("[Comparación] Matplotlib/Seaborn no están disponibles; omitiendo gráficos.")
        return []

    os.makedirs(output_dir, exist_ok=True)
    generated_plots = []

    sns.set_theme(style="whitegrid", font_scale=1.1)

    # 1. Gráfico de Tasa de Éxito vs. Tasa de Colisión
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df_summary))
    width = 0.35

    rects1 = ax.bar(x - width / 2, df_summary["success_rate"], width, label="Éxito (%)", color="#2ca02c")
    rects2 = ax.bar(x + width / 2, df_summary["collision_rate"], width, label="Colisión (%)", color="#d62728")

    ax.set_ylabel("Porcentaje (%)")
    ax.set_title("Comparación de Desempeño: Éxito vs. Colisiones")
    ax.set_xticks(x)
    ax.set_xticklabels(df_summary["pilot"], rotation=15, ha="right")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right")

    # Etiquetas de valor en las barras
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plot_path1 = os.path.join(output_dir, "pilots_success_vs_collision.png")
    fig.savefig(plot_path1, dpi=300)
    plt.close(fig)
    generated_plots.append(plot_path1)

    # 2. Gráfico de Recompensa Media y Longitud de Episodios
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.bar(
        df_summary["pilot"],
        df_summary["mean_reward"],
        yerr=df_summary["std_reward"],
        capsize=5,
        color="#1f77b4",
        alpha=0.85,
    )
    ax1.set_ylabel("Recompensa media acumulada")
    ax1.set_title("Recompensa por Episodio (Media ± Desv.)")
    ax1.tick_params(axis="x", rotation=20)

    ax2.bar(
        df_summary["pilot"],
        df_summary["mean_length"],
        yerr=df_summary["std_length"],
        capsize=5,
        color="#ff7f0e",
        alpha=0.85,
    )
    ax2.set_ylabel("Pasos por episodio (steps)")
    ax2.set_title("Duración del Vuelo (Media ± Desv.)")
    ax2.tick_params(axis="x", rotation=20)

    plt.tight_layout()
    plot_path2 = os.path.join(output_dir, "pilots_rewards_and_lengths.png")
    fig.savefig(plot_path2, dpi=300)
    plt.close(fig)
    generated_plots.append(plot_path2)

    return generated_plots


def parse_args():
    parser = argparse.ArgumentParser(
        description="Comparación y agregación de métricas de evaluación de pilotos."
    )
    parser.add_argument(
        "--csv-files",
        type=str,
        nargs="+",
        required=True,
        help="Lista de rutas a archivos CSV generados por metrics.py.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=None,
        help="Etiquetas descriptivas correspondientes a cada archivo CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./logs",
        help="Directorio destino para guardar resumenes y gráficos (por defecto: ./logs).",
    )
    parser.add_argument(
        "--save-csv",
        type=str,
        default="pilots_comparison_summary.csv",
        help="Nombre del archivo CSV de resumen (relativo a output-dir).",
    )
    parser.add_argument(
        "--save-md",
        type=str,
        default="pilots_comparison_summary.md",
        help="Nombre del archivo Markdown de resumen (relativo a output-dir).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*75}")
    print(" REPORTE COMPARATIVO DE PILOTOS Y AUTONOMÍA COMPARTIDA")
    print(f"{'='*75}")

    df_summary = compare_pilots_from_csvs(args.csv_files, labels=args.labels)

    # Mostrar tabla en consola
    print("\nResumen Estadístico Agregado:")
    print(df_summary.to_string(index=False))

    # Guardar CSV y Markdown
    csv_out = os.path.join(args.output_dir, args.save_csv)
    df_summary.to_csv(csv_out, index=False)
    print(f"\n[Comparación] Tabla resumen guardada en CSV: {csv_out}")

    md_out = os.path.join(args.output_dir, args.save_md)
    with open(md_out, "w", encoding="utf-8") as f:
        f.write("# Resumen Comparativo de Pilotos\n\n")
        f.write(df_summary.to_markdown(index=False))
        f.write("\n")
    print(f"[Comparación] Tabla resumen guardada en Markdown: {md_out}")

    # Generar gráficos comparativos
    plots = generate_comparison_plots(df_summary, output_dir=args.output_dir)
    for p in plots:
        print(f"[Comparación] Gráfico generado: {p}")
    print(f"{'='*75}\n")


if __name__ == "__main__":
    main()
