# -*- coding: utf-8 -*-
"""
Métricas y protocolo de evaluación para pilotos en CornerEnvContinuous.

Implementa la función 'evaluate_pilot' y CLI de evaluación análogos a las tablas
4.1-4.4 de la tesis (Diedrichs, 2025):
    - Tasa de éxito (% de episodios que alcanzaron el cono en vuelo seguro).
    - Tasa de caídas / choques (terminados por caída al suelo o límites).
    - Tasa de tiempo agotado (episodios truncados por exceder max_episode_steps).
    - Recompensa acumulada: promedio y desviación estándar por episodio.
    - Longitud de pasos del episodio: promedio y desviación estándar.
    - Exportación opcional de métricas a DataFrame y archivo CSV.
"""

import argparse
import os
import sys
import numpy as np

# Asegurar que la raíz del proyecto esté en el PYTHONPATH
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
    Ejecuta 'n_episodes' deterministas evaluando un piloto sobre un entorno.

    Args:
        env: Instancia del entorno (ej. CornerEnvContinuous o SharedAutonomyEnv).
        pilot: Instancia de un piloto derivado de BasePilot.
        n_episodes: Número de episodios a evaluar.
        seed: Semilla inicial para reproducibilidad de los episodios.
        verbose: Nivel de detalle (0: silencioso, 1: resumen por episodio y global, 2: paso a paso).
        save_csv_path: Ruta opcional para guardar los resultados detallados en formato CSV.

    Returns:
        dict: Resumen estadístico con medias, desviaciones estándar y tasas de éxito/caída.
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
            # Inferencia determinista según contrato común BasePilot
            action = pilot.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)

            ep_reward += float(reward)
            ep_steps += 1

            if verbose >= 2:
                print(f"  [Ep {ep} | Paso {ep_steps}] Acc: {action.round(3)} | Rew: {reward:.3f}")

        # Análisis de causas de finalización del episodio
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

    # Computar estadísticas agregadas
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

    # Exportación opcional a archivo CSV
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
            # Fallback a escritura CSV estándar si pandas no estuviera disponible
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
) -> BasePilot:
    """
    Fábrica de instanciación unificada para cualquier piloto del ecosistema.
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
        # Piloto base óptimo al que se le inyecta ruido
        base = OptimalPilot(backend=backend, checkpoint_path=checkpoint, deterministic=True)
        return NoisyPilot(optimal_pilot=base, sigma=sigma, seed=seed)
    elif pt == "laggy":
        # Piloto base óptimo al que se le inyecta retardo
        base = OptimalPilot(backend=backend, checkpoint_path=checkpoint, deterministic=True)
        return LaggyPilot(optimal_pilot=base, p_repeat=p_repeat, seed=seed)
    else:
        raise ValueError(f"Tipo de piloto desconocido: '{pilot_type}'. Opciones: optimal, heuristic, noisy, laggy.")


def parse_args():
    """Analizador de argumentos para la evaluación en línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Evaluación de pilotos en CornerEnvContinuous con métricas de la tesis."
    )
    parser.add_argument(
        "--pilot-type",
        type=str,
        default="optimal",
        choices=["optimal", "heuristic", "noisy", "laggy"],
        help="Tipo de piloto a evaluar (optimal, heuristic, noisy, laggy).",
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
