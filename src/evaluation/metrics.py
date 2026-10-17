# -*- coding: utf-8 -*-
"""
Evaluation metrics and protocol for pilots in CornerEnvContinuous.

Implements the 'evaluate_pilot' function and evaluation CLI analogous to Tables
4.1-4.4 from the thesis (Diedrichs, 2025):
    - Success rate (% of episodes reaching the cone in safe flight).
    - Crash / fall rate (terminated due to ground crash or out of bounds).
    - Timeout rate (episodes truncated for exceeding max_episode_steps).
    - Cumulative reward: mean and standard deviation per episode.
    - Episode step length: mean and standard deviation.
    - Optional export of metrics to DataFrame and CSV file.
"""

import argparse
import os
import sys
import numpy as np

# Ensure project root is in PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

from src.pilots.base_pilot import BasePilot


def evaluate_pilot(
    env,
    pilot: BasePilot,
    n_episodes: int = 10,
    seed: int | None = None,
    verbose: int = 1,
    save_csv_path: str | None = None,
) -> dict:
    """
    Executes 'n_episodes' deterministically evaluating a pilot on an environment.

    Args:
        env: Environment instance (e.g., CornerEnvContinuous or SharedAutonomyEnv).
        pilot: Instance of a pilot derived from BasePilot.
        n_episodes: Number of episodes to evaluate.
        seed: Initial seed for episode reproducibility.
        verbose: Detail level (0: silent, 1: per-episode and global summary, 2: step-by-step).
        save_csv_path: Optional path to save detailed results in CSV format.

    Returns:
        dict: Statistical summary with means, standard deviations, and success/crash rates.
    """
    rewards: list[float] = []
    lengths: list[int] = []
    successes: list[bool] = []
    terminations: list[str] = []
    final_distances: list[float] = []

    rng = np.random.default_rng(seed)

    if verbose >= 1:
        print(f"\n{'='*65}")
        print(f" Iniciando evaluación: {n_episodes} episodios")
        print(f" Piloto: {pilot.__class__.__name__}")
        print(f"{'='*65}")

    for ep in range(1, n_episodes + 1):
        ep_seed = int(rng.integers(0, 1_000_000)) if seed is not None else None
        obs, info = env.reset(seed=ep_seed)
        if hasattr(pilot, "reset"):
            pilot.reset()

        ep_reward = 0.0
        ep_steps = 0
        terminated = False
        truncated = False
        term_reason = "unknown"

        while not (terminated or truncated):
            # Deterministic inference according to common BasePilot contract
            action = pilot.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)

            ep_reward += float(reward)
            ep_steps += 1

            if verbose >= 2:
                print(f"  [Ep {ep} | Paso {ep_steps}] Acc: {action.round(3)} | Rew: {reward:.3f}")

        # Analyze episode termination causes
        is_success = bool(info.get("is_success", False))
        corner_flag = info.get("corner")

        if is_success:
            term_reason = "goal"
        elif corner_flag == "fall":
            term_reason = "fall"
        elif corner_flag == "out_of_bounds":
            term_reason = "out_of_bounds"
        elif truncated:
            term_reason = "timeout"
        else:
            term_reason = "terminated"

        final_dist = float(getattr(env, "dist_min_target", 0.0))

        rewards.append(ep_reward)
        lengths.append(ep_steps)
        successes.append(is_success)
        terminations.append(term_reason)
        final_distances.append(final_dist)

        if verbose >= 1:
            icon = "✔ ÉXITO" if is_success else f"✘ FIN ({term_reason})"
            print(
                f" Episodio {ep:3d}/{n_episodes}: {icon:15s} | "
                f"Pasos: {ep_steps:4d} | Recompensa: {ep_reward:8.2f} | "
                f"Dist. Final: {final_dist:.3f}m"
            )

    # Compute aggregate statistics
    total_eps = len(successes)
    success_count = sum(1 for s in successes if s)
    crash_count = sum(1 for t in terminations if t in ("fall", "out_of_bounds"))
    timeout_count = sum(1 for t in terminations if t == "timeout")

    metrics_summary = {
        "n_episodes": total_eps,
        "success_rate": float(success_count / total_eps) if total_eps > 0 else 0.0,
        "crash_rate": float(crash_count / total_eps) if total_eps > 0 else 0.0,
        "timeout_rate": float(timeout_count / total_eps) if total_eps > 0 else 0.0,
        "reward_mean": float(np.mean(rewards)) if rewards else 0.0,
        "reward_std": float(np.std(rewards)) if rewards else 0.0,
        "length_mean": float(np.mean(lengths)) if lengths else 0.0,
        "length_std": float(np.std(lengths)) if lengths else 0.0,
        "final_distance_mean": float(np.mean(final_distances)) if final_distances else 0.0,
        "final_distance_std": float(np.std(final_distances)) if final_distances else 0.0,
        "rewards": rewards,
        "lengths": lengths,
        "successes": successes,
        "terminations": terminations,
        "final_distances": final_distances,
    }

    if verbose >= 1:
        print(f"\n{'-'*65}")
        print(" RESUMEN DE EVALUACIÓN (Patrón Tesis 2025):")
        print(f"  Tasa de Éxito   (Success Rate) : {metrics_summary['success_rate'] * 100:.1f}% ({success_count}/{total_eps})")
        print(f"  Tasa de Caídas  (Crash Rate)   : {metrics_summary['crash_rate'] * 100:.1f}% ({crash_count}/{total_eps})")
        print(f"  Tasa de Tiempo  (Timeout Rate) : {metrics_summary['timeout_rate'] * 100:.1f}% ({timeout_count}/{total_eps})")
        print(f"  Recompensa Media               : {metrics_summary['reward_mean']:.2f} ± {metrics_summary['reward_std']:.2f}")
        print(f"  Longitud Media (pasos)         : {metrics_summary['length_mean']:.1f} ± {metrics_summary['length_std']:.1f}")
        print(f"  Distancia Final Media          : {metrics_summary['final_distance_mean']:.3f}m ± {metrics_summary['final_distance_std']:.3f}m")
        print(f"{'='*65}\n")

    # Optional export to CSV file
    if save_csv_path is not None:
        save_dir = os.path.dirname(os.path.abspath(save_csv_path))
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        data = {
            "episode": list(range(1, total_eps + 1)),
            "reward": rewards,
            "length": lengths,
            "success": successes,
            "termination": terminations,
            "final_distance": final_distances,
        }

        if pd is not None:
            df = pd.DataFrame(data)
            df.to_csv(save_csv_path, index=False)
        else:
            # Fallback to standard CSV writer if pandas is unavailable
            import csv
            with open(save_csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["episode", "reward", "length", "success", "termination", "final_distance"])
                for i in range(total_eps):
                    writer.writerow([i + 1, rewards[i], lengths[i], successes[i], terminations[i], final_distances[i]])

        if verbose >= 1:
            print(f"[Métricas] Resultados guardados en: {save_csv_path}")

    return metrics_summary


def create_pilot_instance(
    pilot_type: str = "optimal",
    checkpoint: str = "./models/ppo_optimal_pilot.zip",
    backend: str = "sb3_ppo",
    heuristic_mode: str = "proportional",
    corner: str = "cone_1",
    sigma: float = 0.1,
    p_repeat: float = 0.2,
    seed: int = 42,
    copilot_checkpoint: str | None = None,
    copilot_backend: str = "auto",
    residual_scale: float = 1.0,
    base_pilot_type: str = "noisy",
) -> BasePilot:
    """
    Unified instantiation factory for any pilot in the ecosystem.
    """
    from src.pilots.heuristic_pilot import HeuristicPilot
    from src.pilots.laggy_pilot import LaggyPilot
    from src.pilots.noisy_pilot import NoisyPilot
    from src.pilots.optimal_pilot import OptimalPilot

    pt = pilot_type.lower().strip()

    if pt == "optimal":
        return OptimalPilot(
            backend=backend,
            checkpoint_path=checkpoint,
            deterministic=True,
        )
    elif pt == "heuristic":
        return HeuristicPilot(
            mode=heuristic_mode,
            corner_name=corner,
        )
    elif pt == "noisy":
        # Optimal base pilot with injected noise
        base = OptimalPilot(backend=backend, checkpoint_path=checkpoint, deterministic=True)
        return NoisyPilot(optimal_pilot=base, sigma=sigma, seed=seed)
    elif pt == "laggy":
        # Optimal base pilot with injected delay
        base = OptimalPilot(backend=backend, checkpoint_path=checkpoint, deterministic=True)
        return LaggyPilot(optimal_pilot=base, p_repeat=p_repeat, seed=seed)
    elif pt == "assisted":
        from src.copilot.residual_policy import AssistedPilot, ResidualPolicy

        # Create the assisted base pilot (noisy by default or as configured)
        base = create_pilot_instance(
            pilot_type=base_pilot_type,
            checkpoint=checkpoint,
            backend=backend,
            heuristic_mode=heuristic_mode,
            corner=corner,
            sigma=sigma,
            p_repeat=p_repeat,
            seed=seed,
        )
        copilot = ResidualPolicy(
            checkpoint_path=copilot_checkpoint,
            backend=copilot_backend,
            residual_scale=residual_scale,
        )
        return AssistedPilot(pilot=base, copilot=copilot)
    else:
        raise ValueError(
            f"Tipo de piloto desconocido: '{pilot_type}'. "
            "Opciones: optimal, heuristic, noisy, laggy, assisted."
        )


def parse_args():
    """Command-line argument parser for evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluación de pilotos en CornerEnvContinuous con métricas de la tesis."
    )
    parser.add_argument(
        "--pilot-type",
        type=str,
        default="optimal",
        choices=["optimal", "heuristic", "noisy", "laggy", "assisted"],
        help="Tipo de piloto a evaluar (optimal, heuristic, noisy, laggy, assisted).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./models/ppo_optimal_pilot.zip",
        help="Ruta al checkpoint del modelo (.zip para SB3, .h para RLtools).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="sb3_ppo",
        choices=["sb3_ppo", "rltools_sac", "heuristic", "dummy"],
        help="Backend para OptimalPilot.",
    )
    parser.add_argument(
        "--heuristic-mode",
        type=str,
        default="proportional",
        choices=["open_loop", "proportional", "fixed", "closed_loop"],
        help="Modo de control para HeuristicPilot.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Cantidad de episodios a evaluar (por defecto: 10).",
    )
    parser.add_argument(
        "--corner",
        type=str,
        default="cone_1",
        help="Nombre del cono objetivo en Webots ('cone_1').",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=1000,
        help="Límite máximo de pasos por episodio antes de truncar.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para reproducibilidad.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.1,
        help="Desviación de ruido para NoisyPilot (por defecto: 0.1).",
    )
    parser.add_argument(
        "--p-repeat",
        type=float,
        default=0.2,
        help="Probabilidad de repetición de acción para LaggyPilot (por defecto: 0.2).",
    )
    parser.add_argument(
        "--copilot-checkpoint",
        type=str,
        default=None,
        help="Ruta al checkpoint del copiloto (.zip o .h) si pilot-type=assisted.",
    )
    parser.add_argument(
        "--copilot-backend",
        type=str,
        default="auto",
        choices=["auto", "sb3_ppo", "sb3_sac", "rltools_sac", "dummy"],
        help="Backend para el copiloto si pilot-type=assisted.",
    )
    parser.add_argument(
        "--residual-scale",
        type=float,
        default=1.0,
        help="Factor de escala para la acción residual del copiloto (por defecto: 1.0).",
    )
    parser.add_argument(
        "--base-pilot-type",
        type=str,
        default="noisy",
        choices=["optimal", "heuristic", "noisy", "laggy"],
        help="Tipo de piloto base a asistir si pilot-type=assisted (por defecto: noisy).",
    )
    parser.add_argument(
        "--save-csv",
        type=str,
        default="./logs/eval_results.csv",
        help="Ruta destino para guardar resultados CSV detallados.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Nivel de logging (0: silencioso, 1: resumen por ep, 2: paso a paso).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    from src.envs.corner_env_continuous import CornerEnvContinuous

    print(f"[Evaluación] Inicializando entorno CornerEnvContinuous (objetivo: {args.corner})...")
    env = CornerEnvContinuous(
        corner_name=args.corner,
        max_episode_steps=args.max_episode_steps,
        verbose=0,
    )

    print(f"[Evaluación] Instanciando piloto tipo '{args.pilot_type}'...")
    pilot = create_pilot_instance(
        pilot_type=args.pilot_type,
        checkpoint=args.checkpoint,
        backend=args.backend,
        heuristic_mode=args.heuristic_mode,
        corner=args.corner,
        sigma=args.sigma,
        p_repeat=args.p_repeat,
        seed=args.seed,
        copilot_checkpoint=args.copilot_checkpoint,
        copilot_backend=args.copilot_backend,
        residual_scale=args.residual_scale,
        base_pilot_type=args.base_pilot_type,
    )

    try:
        results = evaluate_pilot(
            env=env,
            pilot=pilot,
            n_episodes=args.episodes,
            seed=args.seed,
            verbose=args.verbose,
            save_csv_path=args.save_csv,
        )
    finally:
        env.close()
