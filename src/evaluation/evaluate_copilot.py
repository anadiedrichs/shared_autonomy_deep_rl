# -*- coding: utf-8 -*-
"""
Evaluation and benchmark script for the Constrained Residual Copilot (Schaff & Walter, 2020).

Evaluates the trained copilot (ConstrainedPPO) against multiple surrogate human pilot models
(NoisyPilot, LaggyPilot, StandByPilot, HeuristicPilot) in both:
    1. Placebo condition: Unassisted teleoperation (a_r = 0, a_composed = a_h).
    2. Assisted condition: Shared autonomy (a_composed = clip(a_h + beta * a_r, -1, 1)).

Quantitative metrics computed per configuration:
    - Success Rate (%)
    - Collision / Crash Rate (%) [fall, out_of_bounds]
    - Mean Intervention Norm (||a_r||_2)
    - Mean Jerk / Smoothness (||a_r(t) - a_r(t-1)||_2)
    - Cumulative Reward & Episode Length
    - Final Distance to Target (m)

Generates comparative CSV tables and publication-ready matplotlib figures in ./eval_results/.
"""

import argparse
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Headless matplotlib configuration for Docker / headless servers
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.copilot.constrained_ppo import ConstrainedPPO
from src.envs.shared_autonomy_env import SharedAutonomyEnv
from src.pilots.heuristic_pilot import HeuristicPilot
from src.pilots.laggy_pilot import LaggyPilot
from src.pilots.noisy_pilot import NoisyPilot
from src.pilots.optimal_pilot import OptimalPilot
from src.pilots.standby_pilot import StandByPilot


def compute_wilson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calcula el intervalo de confianza de Wilson para una proporción binomial (k éxitos en n ensayos).
    Proporciona cotas asimétricas formalmente rigurosas en [0.0, 100.0]%, evitando el colapso
    de la aproximación normal Wald cuando p = 0 o p = 1.
    """
    if n <= 0:
        return 0.0, 0.0
    z = 1.95996  # 95% de nivel de confianza bilateral
    p = float(k) / float(n)
    denom = 1.0 + (z**2) / float(n)
    center = (p + (z**2) / (2.0 * float(n))) / denom
    spread = (z * math.sqrt(max(0.0, p * (1.0 - p) / float(n) + (z**2) / (4.0 * (float(n)**2))))) / denom
    lower = max(0.0, center - spread) * 100.0
    upper = min(1.0, center + spread) * 100.0
    return round(lower, 2), round(upper, 2)


def build_surrogate_pilot(
    pilot_config_name: str,
    optimal_checkpoint: Optional[str] = None,
    optimal_backend: str = "sb3_ppo",
    corner_name: str = "cone_1",
    seed: int = 42,
) -> Tuple[object, str]:
    """
    Instantiates a surrogate pilot given a configuration key.

    Supported configurations:
        - noisy_0.5   : NoisyPilot with sigma = 0.50
        - noisy_1.0   : NoisyPilot with sigma = 1.00
        - laggy_0.8   : LaggyPilot with p_repeat = 0.80 (human latency)
        - standby_0.8 : StandByPilot with p_drop = 0.80 (NOP / signal drop)
        - heuristic   : HeuristicPilot (proportional corner pursuit)
        - optimal     : OptimalPilot (unperturbed expert)

    Returns:
        tuple: (pilot_instance, display_name)
    """
    key = pilot_config_name.lower().strip()

    if key == "heuristic":
        return HeuristicPilot(mode="proportional", corner_name=corner_name), "Piloto Heurístico"

    if key == "optimal":
        return (
            OptimalPilot(backend=optimal_backend, checkpoint_path=optimal_checkpoint, deterministic=True),
            "Piloto Óptimo",
        )

    # Base optimal policy used as reference for degraded pilots
    base_optimal = OptimalPilot(
        backend=optimal_backend,
        checkpoint_path=optimal_checkpoint,
        deterministic=True,
    )

    if key in ("noisy_0.5", "noisy", "noisy_sigma_0.5"):
        return NoisyPilot(optimal_pilot=base_optimal, sigma=0.50, seed=seed), "Noisy Pilot (σ=0.5)"
    elif key in ("noisy_1.0", "noisy_sigma_1.0"):
        return NoisyPilot(optimal_pilot=base_optimal, sigma=1.00, seed=seed), "Noisy Pilot (σ=1.0)"
    elif key in ("laggy_0.8", "laggy", "laggy_p_0.8"):
        return LaggyPilot(optimal_pilot=base_optimal, p_repeat=0.80, seed=seed), "Laggy Pilot (p=0.8)"
    elif key in ("standby_0.8", "standby", "standby_p_0.8"):
        return StandByPilot(optimal_pilot=base_optimal, p_drop=0.80, seed=seed), "StandBy Pilot (p=0.8)"
    else:
        raise ValueError(
            f"Configuración de piloto desconocida: '{pilot_config_name}'. "
            "Opciones: noisy_0.5, noisy_1.0, laggy_0.8, standby_0.8, heuristic, optimal."
        )


def evaluate_single_episode(
    env: SharedAutonomyEnv,
    copilot_model: Optional[object],
    is_assisted: bool,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Executes a single evaluation episode in the SharedAutonomyEnv.

    Args:
        env: The shared autonomy environment.
        copilot_model: Trained ConstrainedPPO / PPO model or None.
        is_assisted: True to apply residual action a_r, False for Placebo (a_r = 0).
        seed: Random seed for environment reset.

    Returns:
        dict: Detailed episode statistics.
    """
    copilot_obs, info = env.reset(seed=seed)
    terminated = False
    truncated = False

    ep_reward = 0.0
    ep_steps = 0
    intervention_norms: List[float] = []
    jerks: List[float] = []
    prev_residual = np.zeros(4, dtype=np.float32)

    while not (terminated or truncated):
        if is_assisted and copilot_model is not None:
            # Deterministic copilot inference: a_r in [-1, 1]^4
            action_r, _ = copilot_model.predict(copilot_obs, deterministic=True)
            action_r = np.asarray(action_r, dtype=np.float32)
        else:
            # Placebo condition: no assistance (a_r = 0)
            action_r = np.zeros(4, dtype=np.float32)

        # Compute step intervention metrics
        norm_r = float(np.linalg.norm(action_r))
        jerk_r = float(np.linalg.norm(action_r - prev_residual))
        intervention_norms.append(norm_r)
        jerks.append(jerk_r)
        prev_residual = action_r.copy()

        copilot_obs, reward, terminated, truncated, info = env.step(action_r)
        ep_reward += float(reward)
        ep_steps += 1

    # Analyze termination cause
    is_success = bool(info.get("is_success", False))
    corner_flag = info.get("corner")

    if is_success:
        term_reason = "goal"
        is_collision = False
    elif corner_flag in ("fall", "out_of_bounds") or getattr(env, "alt", 1.0) < 0.10:
        term_reason = "fall" if (corner_flag == "fall" or getattr(env, "alt", 1.0) < 0.10) else "out_of_bounds"
        is_collision = True
    elif truncated:
        term_reason = "timeout"
        is_collision = False
    else:
        term_reason = "terminated"
        is_collision = False

    final_dist = float(getattr(env, "dist_min_target", 0.0))

    # Distancia al obstáculo más cercano (menor de los 4 láseres ToF en metros)
    unwrapped_env = getattr(env, "unwrapped", env)
    d_front = getattr(env, "dist_front", getattr(unwrapped_env, "dist_front", 2000.0))
    d_back = getattr(env, "dist_back", getattr(unwrapped_env, "dist_back", 2000.0))
    d_right = getattr(env, "dist_right", getattr(unwrapped_env, "dist_right", 2000.0))
    d_left = getattr(env, "dist_left", getattr(unwrapped_env, "dist_left", 2000.0))
    min_obs_dist = round(float(min(d_front, d_back, d_right, d_left)) / 1000.0, 3)
    final_alt_m = float(getattr(env, "alt", getattr(unwrapped_env, "alt", getattr(env, "z_global", info.get("altitude", 0.0)))))

    mean_intervention = float(np.mean(intervention_norms)) if intervention_norms else 0.0
    mean_jerk = float(np.mean(jerks[1:])) if len(jerks) > 1 else (float(jerks[0]) if jerks else 0.0)

    return {
        "success": is_success,
        "collision": is_collision,
        "termination": term_reason,
        "steps": ep_steps,
        "length": ep_steps,
        "reward": ep_reward,
        "final_distance": final_dist,
        "min_obstacle_distance": min_obs_dist,
        "final_altitude_m": round(final_alt_m, 3),
        "mean_intervention_norm": mean_intervention,
        "mean_jerk": mean_jerk,
    }


def evaluate_copilot_suite(
    copilot_checkpoint: str,
    optimal_checkpoint: Optional[str] = None,
    optimal_backend: str = "sb3_ppo",
    pilot_configs: Optional[List[str]] = None,
    n_episodes: int = 20,
    seeds: Optional[List[int]] = None,
    residual_scale: float = 1.0,
    corner_name: str = "cone_1",
    output_dir: str = "./eval_results",
    skip_placebo: bool = False,
    skip_assisted: bool = False,
    verbose: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluates the trained copilot across all requested surrogate pilots and conditions.

    Args:
        copilot_checkpoint: Path to trained ConstrainedPPO model .zip.
        optimal_checkpoint: Path to optimal pilot checkpoint.
        optimal_backend: Backend for optimal pilot ('sb3_ppo', 'heuristic', etc.).
        pilot_configs: List of pilot keys to evaluate.
        n_episodes: Number of episodes per configuration.
        seeds: Explicit list of seeds (defaults to range(n_episodes) if None).
        residual_scale: Authority scaling factor beta.
        corner_name: Target corner cone in Webots.
        output_dir: Directory where CSVs and plots are saved.
        skip_placebo: If True, skips unassisted baseline.
        skip_assisted: If True, skips assisted condition.
        verbose: Print verbosity level.

    Returns:
        tuple: (episodes_df, summary_df)
    """
    os.makedirs(output_dir, exist_ok=True)

    if pilot_configs is None or len(pilot_configs) == 0:
        pilot_configs = ["noisy_0.5", "noisy_1.0", "laggy_0.8", "standby_0.8"]

    if seeds is None or len(seeds) == 0:
        seeds = [42 + i * 101 for i in range(n_episodes)]
    else:
        n_episodes = len(seeds)

    # Load trained Copilot model
    copilot_model = None
    if not skip_assisted:
        if not os.path.exists(copilot_checkpoint):
            raise FileNotFoundError(
                f"No se encontró el checkpoint del copiloto en: {copilot_checkpoint}"
            )
        if verbose > 0:
            print(f"[Evaluación] Cargando modelo del Copiloto desde: {copilot_checkpoint}")
        try:
            copilot_model = ConstrainedPPO.load(copilot_checkpoint)
        except Exception:
            # Fallback to standard SB3 PPO loader if ConstrainedPPO custom load needs fallback
            from stable_baselines3 import PPO
            copilot_model = PPO.load(copilot_checkpoint)

    modes: List[str] = []
    if not skip_placebo:
        modes.append("placebo")
    if not skip_assisted:
        modes.append("assisted")

    episode_records: List[Dict[str, Any]] = []

    # Instanciar el entorno una única vez (Webots solo permite un Robot/Supervisor por proceso)
    env = SharedAutonomyEnv(
        pilot=None,
        copilot=None,  # Manejado explícitamente en el bucle
        max_episode_steps=1_000,
        corner_name=corner_name,
        reward_mode="safety",
        residual_scale=residual_scale,
        include_pilot_action_in_obs=True,
        verbose=0,
    )

    try:
        for pilot_key in pilot_configs:
            for mode in modes:
                is_assisted = (mode == "assisted")
                mode_label = "Asistido (Copiloto)" if is_assisted else "Placebo (Sin Asistencia)"

                pilot_instance, display_name = build_surrogate_pilot(
                    pilot_config_name=pilot_key,
                    optimal_checkpoint=optimal_checkpoint,
                    optimal_backend=optimal_backend,
                    corner_name=corner_name,
                    seed=seeds[0],
                )

                if verbose > 0:
                    print(f"\n{'='*70}")
                    print(f" Evaluando: {display_name} | Modo: {mode_label} | {n_episodes} episodios")
                    print(f"{'='*70}")

                # Actualizar el piloto y la escala de autoridad residual en la instancia existente
                env.pilot = pilot_instance
                env.residual_scale = residual_scale if is_assisted else 0.0

                for ep_idx, ep_seed in enumerate(seeds):
                    # Ensure pilot seed is updated if it supports it
                    if hasattr(pilot_instance, "seed"):
                        try:
                            pilot_instance.seed(ep_seed)
                        except Exception:
                            pass

                    res = evaluate_single_episode(
                        env=env,
                        copilot_model=copilot_model,
                        is_assisted=is_assisted,
                        seed=ep_seed,
                    )

                    record = {
                        "pilot_key": pilot_key,
                        "pilot_name": display_name,
                        "mode": mode,
                        "mode_label": mode_label,
                        "episode": ep_idx + 1,
                        "seed": ep_seed,
                        "success": res["success"],
                        "collision": res["collision"],
                        "termination": res["termination"],
                        "steps": res["steps"],
                        "reward": res["reward"],
                        "final_distance": res["final_distance"],
                        "mean_intervention_norm": res["mean_intervention_norm"],
                        "mean_jerk": res["mean_jerk"],
                    }
                    episode_records.append(record)

                    if verbose >= 2 or (verbose == 1 and (ep_idx + 1) % 5 == 0):
                        icon = "✔ ÉXITO" if res["success"] else f"✘ ({res['termination']})"
                        print(
                            f"  Ep {ep_idx+1:2d}/{n_episodes} (seed {ep_seed}): {icon:12s} | "
                            f"Pasos: {res['steps']:3d} | Rew: {res['reward']:6.2f} | "
                            f"||a_r||: {res['mean_intervention_norm']:.3f} | "
                            f"Jerk: {res['mean_jerk']:.3f}"
                        )
    finally:
        try:
            env.close()
        except Exception:
            pass

    episodes_df = pd.DataFrame(episode_records)

    # Compute aggregate summary grouped by (pilot_name, mode)
    summary_rows = []
    for (pilot_name, mode), group in episodes_df.groupby(["pilot_name", "mode"]):
        n_tot = len(group)
        n_succ = int(group["success"].sum())
        n_coll = int(group["collision"].sum())
        succ_ci_low, succ_ci_high = compute_wilson_ci(n_succ, n_tot)
        coll_ci_low, coll_ci_high = compute_wilson_ci(n_coll, n_tot)
        succ_pct = float(n_succ / n_tot * 100.0) if n_tot > 0 else 0.0
        coll_pct = float(n_coll / n_tot * 100.0) if n_tot > 0 else 0.0

        summary_rows.append({
            "pilot_name": pilot_name,
            "mode": mode,
            "episodes": n_tot,
            "success_rate_pct": succ_pct,
            "success_wilson_ci_95": f"[{succ_ci_low:.1f}%, {succ_ci_high:.1f}%]",
            "success_ci_lower": succ_ci_low,
            "success_ci_upper": succ_ci_high,
            "collision_rate_pct": coll_pct,
            "collision_wilson_ci_95": f"[{coll_ci_low:.1f}%, {coll_ci_high:.1f}%]",
            "collision_ci_lower": coll_ci_low,
            "collision_ci_upper": coll_ci_high,
            "mean_reward": float(group["reward"].mean()),
            "std_reward": float(group["reward"].std()),
            "mean_steps": float(group["steps"].mean()),
            "std_steps": float(group["steps"].std()),
            "mean_intervention_norm": float(group["mean_intervention_norm"].mean()),
            "std_intervention_norm": float(group["mean_intervention_norm"].std()),
            "mean_jerk": float(group["mean_jerk"].mean()),
            "std_jerk": float(group["mean_jerk"].std()),
            "mean_final_distance": float(group["final_distance"].mean()),
        })

    summary_df = pd.DataFrame(summary_rows)

    # Export CSVs
    episodes_csv = os.path.join(output_dir, "eval_copilot_episodes.csv")
    summary_csv = os.path.join(output_dir, "eval_copilot_summary.csv")
    episodes_df.to_csv(episodes_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    if verbose > 0:
        print(f"\n{'='*70}")
        print(" RESUMEN CUANTITATIVO DE EVALUACIÓN DEL COPILOTO (Intervalos Wilson 95%)")
        print(f"{'='*70}")
        display_cols = [
            "pilot_name", "mode", "success_rate_pct", "success_wilson_ci_95",
            "collision_rate_pct", "collision_wilson_ci_95",
            "mean_intervention_norm", "mean_jerk"
        ]
        available_cols = [c for c in display_cols if c in summary_df.columns]
        print(summary_df[available_cols].to_string(index=False))
        print(f"\nResultados exportados a:")
        print(f"  - Episodios : {episodes_csv}")
        print(f"  - Resumen   : {summary_csv}")

    # Generate comparative plots
    if HAS_MATPLOTLIB:
        plot_copilot_comparisons(summary_df, output_dir=output_dir, verbose=verbose)

    return episodes_df, summary_df


def plot_copilot_comparisons(summary_df: pd.DataFrame, output_dir: str, verbose: int = 1):
    """
    Generates publication-quality comparative plots using matplotlib.
    """
    if summary_df.empty:
        return

    os.makedirs(output_dir, exist_ok=True)
    pilots = sorted(summary_df["pilot_name"].unique())
    x = np.arange(len(pilots))
    bar_width = 0.35

    # Prepare data for Placebo and Assisted
    def get_metric_series(mode_name: str, col_name: str) -> np.ndarray:
        vals = []
        for p in pilots:
            row = summary_df[(summary_df["pilot_name"] == p) & (summary_df["mode"] == mode_name)]
            vals.append(row[col_name].values[0] if len(row) > 0 else 0.0)
        return np.array(vals, dtype=np.float32)

    has_placebo = "placebo" in summary_df["mode"].values
    has_assisted = "assisted" in summary_df["mode"].values

    succ_placebo = get_metric_series("placebo", "success_rate_pct") if has_placebo else None
    succ_assisted = get_metric_series("assisted", "success_rate_pct") if has_assisted else None

    coll_placebo = get_metric_series("placebo", "collision_rate_pct") if has_placebo else None
    coll_assisted = get_metric_series("assisted", "collision_rate_pct") if has_assisted else None

    norm_assisted = get_metric_series("assisted", "mean_intervention_norm") if has_assisted else None
    jerk_assisted = get_metric_series("assisted", "mean_jerk") if has_assisted else None

    # 1. Success Rate Comparison Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    if has_placebo and has_assisted:
        ax.bar(x - bar_width / 2, succ_placebo, bar_width, label="Placebo (Sin Asistencia)", color="#e74c3c", alpha=0.85)
        ax.bar(x + bar_width / 2, succ_assisted, bar_width, label="Asistido (Constrained PPO)", color="#2ecc71", alpha=0.85)
    elif has_assisted:
        ax.bar(x, succ_assisted, bar_width, label="Asistido (Constrained PPO)", color="#2ecc71", alpha=0.85)
    else:
        ax.bar(x, succ_placebo, bar_width, label="Placebo", color="#e74c3c", alpha=0.85)

    ax.set_ylabel("Tasa de Éxito (%)", fontsize=11, fontweight="bold")
    ax.set_title("Eficacia de Asistencia: Tasa de Éxito por Perfil de Piloto", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(pilots, rotation=15, ha="right", fontsize=9)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper left")
    plt.tight_layout()
    p1 = os.path.join(output_dir, "copilot_success_rate_comparison.png")
    fig.savefig(p1, dpi=300)
    plt.close(fig)

    # 2. Collision Rate Comparison Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    if has_placebo and has_assisted:
        ax.bar(x - bar_width / 2, coll_placebo, bar_width, label="Placebo (Sin Asistencia)", color="#c0392b", alpha=0.85)
        ax.bar(x + bar_width / 2, coll_assisted, bar_width, label="Asistido (Constrained PPO)", color="#27ae60", alpha=0.85)
    elif has_assisted:
        ax.bar(x, coll_assisted, bar_width, label="Asistido", color="#27ae60", alpha=0.85)
    else:
        ax.bar(x, coll_placebo, bar_width, label="Placebo", color="#c0392b", alpha=0.85)

    ax.set_ylabel("Tasa de Colisión (%)", fontsize=11, fontweight="bold")
    ax.set_title("Seguridad Operacional: Tasa de Colisión por Perfil de Piloto", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(pilots, rotation=15, ha="right", fontsize=9)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper right")
    plt.tight_layout()
    p2 = os.path.join(output_dir, "copilot_collision_rate_comparison.png")
    fig.savefig(p2, dpi=300)
    plt.close(fig)

    # 3. Intervention Norm and Jerk Plot (for assisted mode)
    if has_assisted and norm_assisted is not None and jerk_assisted is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.bar(x, norm_assisted, width=0.45, color="#3498db", alpha=0.85)
        ax1.set_ylabel(r"Norma Media $\|a_r\|_2$", fontsize=11, fontweight="bold")
        ax1.set_title("Nivel de Intervención del Copiloto", fontsize=12, fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels(pilots, rotation=20, ha="right", fontsize=9)
        ax1.grid(axis="y", linestyle="--", alpha=0.4)

        ax2.bar(x, jerk_assisted, width=0.45, color="#9b59b6", alpha=0.85)
        ax2.set_ylabel(r"Jerk $\|a_r(t) - a_r(t-1)\|_2$", fontsize=11, fontweight="bold")
        ax2.set_title("Suavidad de la Asistencia (Jerk)", fontsize=12, fontweight="bold")
        ax2.set_xticks(x)
        ax2.set_xticklabels(pilots, rotation=20, ha="right", fontsize=9)
        ax2.grid(axis="y", linestyle="--", alpha=0.4)

        plt.tight_layout()
        p3 = os.path.join(output_dir, "copilot_intervention_and_jerk.png")
        fig.savefig(p3, dpi=300)
        plt.close(fig)

    # 4. Comprehensive 4-Panel Performance Dashboard
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # (0, 0): Success Rate
    if has_placebo and has_assisted:
        axs[0, 0].bar(x - bar_width/2, succ_placebo, bar_width, label="Placebo", color="#e74c3c", alpha=0.85)
        axs[0, 0].bar(x + bar_width/2, succ_assisted, bar_width, label="Asistido", color="#2ecc71", alpha=0.85)
        axs[0, 0].legend()
    axs[0, 0].set_title("Tasa de Éxito (%)", fontweight="bold")
    axs[0, 0].set_xticks(x)
    axs[0, 0].set_xticklabels(pilots, rotation=15, ha="right")
    axs[0, 0].set_ylim(0, 105)
    axs[0, 0].grid(axis="y", linestyle="--", alpha=0.4)

    # (0, 1): Collision Rate
    if has_placebo and has_assisted:
        axs[0, 1].bar(x - bar_width/2, coll_placebo, bar_width, label="Placebo", color="#c0392b", alpha=0.85)
        axs[0, 1].bar(x + bar_width/2, coll_assisted, bar_width, label="Asistido", color="#27ae60", alpha=0.85)
        axs[0, 1].legend()
    axs[0, 1].set_title("Tasa de Colisión (%)", fontweight="bold")
    axs[0, 1].set_xticks(x)
    axs[0, 1].set_xticklabels(pilots, rotation=15, ha="right")
    axs[0, 1].set_ylim(0, 105)
    axs[0, 1].grid(axis="y", linestyle="--", alpha=0.4)

    # (1, 0): Intervention Norm
    if has_assisted and norm_assisted is not None:
        axs[1, 0].bar(x, norm_assisted, width=0.45, color="#3498db", alpha=0.85)
    axs[1, 0].set_title(r"Norma Media de Intervención $\|a_r\|_2$", fontweight="bold")
    axs[1, 0].set_xticks(x)
    axs[1, 0].set_xticklabels(pilots, rotation=15, ha="right")
    axs[1, 0].grid(axis="y", linestyle="--", alpha=0.4)

    # (1, 1): Smoothness / Jerk
    if has_assisted and jerk_assisted is not None:
        axs[1, 1].bar(x, jerk_assisted, width=0.45, color="#9b59b6", alpha=0.85)
    axs[1, 1].set_title(r"Métrica de Jerk $\|\Delta a_r\|_2$", fontweight="bold")
    axs[1, 1].set_xticks(x)
    axs[1, 1].set_xticklabels(pilots, rotation=15, ha="right")
    axs[1, 1].grid(axis="y", linestyle="--", alpha=0.4)

    plt.suptitle("Dashboard Comparativo del Copiloto Residual Constreñido (Schaff & Walter, 2020)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    p4 = os.path.join(output_dir, "copilot_performance_dashboard.png")
    fig.savefig(p4, dpi=300)
    plt.close(fig)

    if verbose > 0:
        print(f"Gráficos comparativos guardados exitosamente en: {output_dir}/")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluación del Copiloto Residual Constreñido (Schaff & Walter, 2020)"
    )
    parser.add_argument(
        "--copilot-checkpoint",
        type=str,
        default="./models/copilot_constrained_ppo.zip",
        help="Ruta al modelo entrenado del copiloto (.zip).",
    )
    parser.add_argument(
        "--optimal-checkpoint",
        type=str,
        default=None,
        help="Ruta al checkpoint de la política óptima base (SB3 o RLtools).",
    )
    parser.add_argument(
        "--optimal-backend",
        type=str,
        default="sb3_ppo",
        choices=["sb3_ppo", "rltools_sac", "rltools_td3", "heuristic", "dummy"],
        help="Framework del piloto base.",
    )
    parser.add_argument(
        "--pilots",
        nargs="+",
        default=["noisy_0.5", "noisy_1.0", "laggy_0.8", "standby_0.8"],
        help="Lista de pilotos a evaluar (noisy_0.5, noisy_1.0, laggy_0.8, standby_0.8, heuristic, optimal).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Número de episodios por configuración de piloto y modo.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Lista explícita de semillas (ej: --seeds 42 123 256 512 789).",
    )
    parser.add_argument(
        "--residual-scale",
        type=float,
        default=1.0,
        help="Factor de modulación de autoridad beta en (0, 1].",
    )
    parser.add_argument(
        "--corner",
        type=str,
        default="cone_1",
        help="Cono objetivo en Webots (cone_1, cone_2, cone_3, cone_4).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./eval_results",
        help="Directorio donde se guardarán los CSVs y figuras comparativas.",
    )
    parser.add_argument(
        "--skip-placebo",
        action="store_true",
        help="Omitir evaluación de la condición Placebo (sin copiloto).",
    )
    parser.add_argument(
        "--skip-assisted",
        action="store_true",
        help="Omitir evaluación de la condición Asistida.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Nivel de detalle por consola (0: silencioso, 1: resumen, 2: paso a paso).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Expand 'all' keyword if provided in pilots argument
    if "all" in [p.lower() for p in args.pilots]:
        pilots = ["noisy_0.5", "noisy_1.0", "laggy_0.8", "standby_0.8", "heuristic"]
    else:
        pilots = args.pilots

    evaluate_copilot_suite(
        copilot_checkpoint=args.copilot_checkpoint,
        optimal_checkpoint=args.optimal_checkpoint,
        optimal_backend=args.optimal_backend,
        pilot_configs=pilots,
        n_episodes=args.episodes,
        seeds=args.seeds,
        residual_scale=args.residual_scale,
        corner_name=args.corner,
        output_dir=args.output_dir,
        skip_placebo=args.skip_placebo,
        skip_assisted=args.skip_assisted,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
