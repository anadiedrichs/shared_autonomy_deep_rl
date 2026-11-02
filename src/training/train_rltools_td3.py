# -*- coding: utf-8 -*-
"""
Continuous optimal pilot training with TD3 (RLtools wrapper).

Trains a continuous policy with Twin Delayed DDPG (TD3) using the RLtools
interface conventions on CornerEnvContinuous, and exports the trained policy
directly as a static C/C++ header (.h) for embedded deployment on the Crazyflie.
"""

import argparse
import os
import sys
import time
import numpy as np

# Disable Intel MKL conflicts with PyTorch if applicable
os.environ.setdefault("RL_TOOLS_DISABLE_MKL", "1")

# Ensure the project root is in PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.training.train_rltools_sac import make_env_factory, CStdoutTee
from src.training.rltools_td3 import TD3


def train_rltools_td3(
    max_steps: int = 100_000,
    seed: int = 42,
    gamma: float = 0.97,
    corner_name: str = "cone_1",
    max_episode_steps: int = 1_000,
    header_save_path: str = "./models/optimal_pilot_td3_checkpoint.h",
    log_dir: str | None = "./logs/td3",
    verbose: bool = True,
):
    """
    Executes TD3 training loop for the optimal pilot and exports the policy.

    Args:
        max_steps: Maximum number of training steps to execute.
        seed: Random seed for numerical reproducibility.
        gamma: Discount factor.
        corner_name: DEF identifier of the target cone in Webots.
        max_episode_steps: Maximum step limit per episode.
        header_save_path: Path to the C header file (.h) to export the trained policy.
        log_dir: Directory to save TensorBoard logs and metrics.
        verbose: If True, prints periodic progress to the console.

    Returns:
        tuple: (final training state, C source code of the exported header).
    """
    os.makedirs(os.path.dirname(os.path.abspath(header_save_path)), exist_ok=True)

    # TensorBoard setup and hyperparameter logging
    writer = None
    if log_dir is not None:
        try:
            from torch.utils.tensorboard import SummaryWriter
            run_dir = os.path.join(log_dir, f"td3_seed{seed}")
            os.makedirs(run_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=run_dir)
            hparams = {
                "algorithm": "RLtools-TD3",
                "max_steps": max_steps,
                "seed": seed,
                "gamma": gamma,
                "corner_name": corner_name,
                "max_episode_steps": max_episode_steps,
            }
            metrics = {
                "train/step": 0,
            }
            writer.add_hparams(hparams, metrics)
        except ImportError:
            writer = None
            run_dir = log_dir
    else:
        run_dir = None

    print(f"[RLtools-TD3] Creando fábrica de entorno con métricas (esquina: {corner_name})...")
    metrics_log_dir = run_dir if run_dir is not None else log_dir
    env_factory = make_env_factory(
        corner_name=corner_name,
        max_episode_steps=max_episode_steps,
        log_dir=metrics_log_dir,
        writer=writer,
        verbose=verbose,
        algorithm_name="RLtools-TD3",
    )

    print(f"[RLtools-TD3] Inicializando algoritmo TD3 con semilla {seed} y gamma {gamma}...")
    eval_interval = min(1_000, max(1, max_steps // 10))
    td3 = TD3(
        env_factory,
        STEP_LIMIT=max_steps,
        EPISODE_STEP_LIMIT=max_episode_steps,
        GAMMA=gamma,
        evaluation_interval=eval_interval,
        num_evaluation_episodes=5,
        verbose=verbose,
    )
    state = td3.State(seed)

    best_eval_return = float("-inf")
    best_step = None
    best_policy_c_header = None
    best_header_path = header_save_path.replace(".h", "_best.h")
    final_header_path = header_save_path.replace(".h", "_final.h")

    stdout_tee = CStdoutTee()
    print(f"[RLtools-TD3] Iniciando bucle de entrenamiento (hasta {max_steps} pasos)...")
    step_count = 0
    finished = False

    try:
        while not finished and step_count < max_steps:
            finished = state.step()
            step_count += 1

            # Check for periodic evaluation outputs from training state
            eval_returns = stdout_tee.read_eval_returns()
            for ret in eval_returns:
                if writer is not None:
                    writer.add_scalar("evaluation/mean_return", ret, step_count)
                if ret > best_eval_return:
                    best_eval_return = ret
                    best_step = step_count
                    best_policy_c_header = state.export_policy()

                    # Save best model to primary path and dedicated best checkpoint path
                    with open(header_save_path, "w", encoding="utf-8") as f:
                        f.write(best_policy_c_header)
                    with open(best_header_path, "w", encoding="utf-8") as f:
                        f.write(best_policy_c_header)

                    if verbose:
                        print(
                            f"\n[RLtools-TD3] ★ ¡Nuevo mejor modelo guardado en paso {step_count}! "
                            f"Retorno medio de evaluación: {ret:.4f} (guardado en {header_save_path})\n"
                        )
                    if writer is not None:
                        writer.add_scalar("evaluation/best_mean_return", best_eval_return, step_count)

            if step_count % 10_000 == 0 or finished:
                if verbose:
                    print(f"[RLtools-TD3] Progreso: paso {step_count}/{max_steps} completado.")
                if writer is not None:
                    writer.add_scalar("train/step", step_count, step_count)
    finally:
        stdout_tee.close()

    print(f"[RLtools-TD3] Entrenamiento finalizado. Exportando política del paso final a {final_header_path}...")
    final_policy_c_header = state.export_policy()
    with open(final_header_path, "w", encoding="utf-8") as f:
        f.write(final_policy_c_header)

    # Ensure primary path has the best checkpoint, fallback to final if no evaluation was superior
    if best_policy_c_header is None:
        best_policy_c_header = final_policy_c_header
        best_step = step_count
        best_eval_return = 0.0
        with open(header_save_path, "w", encoding="utf-8") as f:
            f.write(best_policy_c_header)

    if writer is not None:
        writer.close()

    # Flush CSV logs and close environment
    try:
        env_instance = env_factory()
        if hasattr(env_instance, "close"):
            env_instance.close()
    except Exception:
        pass

    if verbose:
        print(
            f"[RLtools-TD3] Resumen de checkpoints:\n"
            f"  - Checkpoint óptimo activo: {header_save_path} (paso {best_step}, retorno {best_eval_return:.4f})\n"
            f"  - Checkpoint mejor guardado: {best_header_path}\n"
            f"  - Checkpoint paso final:     {final_header_path}"
        )
    return state, best_policy_c_header


def parse_args():
    """Command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Entrenamiento del piloto óptimo con TD3 (RLtools wrapper) y exportación a header C."
    )
    parser.add_argument(
        "--steps", "--timesteps",
        dest="steps",
        type=int,
        default=100_000,
        help="Número máximo de pasos de entrenamiento.",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=1_000,
        help="Límite máximo de pasos por episodio antes de truncar.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Semilla para reproducibilidad.")
    parser.add_argument(
        "--gamma", type=float, default=0.97, help="Factor de descuento gamma (por defecto 0.97)."
    )
    parser.add_argument(
        "--corner", type=str, default="cone_1", help="Nombre del cono objetivo en Webots."
    )
    parser.add_argument(
        "--export-path", "--save-path", "--model-path",
        dest="export_path",
        type=str,
        default="./models/optimal_pilot_td3_checkpoint.h",
        help="Ruta destino para el header C exportado.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs/td3",
        help="Directorio de logs y TensorBoard para TD3.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_rltools_td3(
        max_steps=args.steps,
        max_episode_steps=args.max_episode_steps,
        seed=args.seed,
        gamma=args.gamma,
        corner_name=args.corner,
        header_save_path=args.export_path,
        log_dir=args.log_dir,
    )
