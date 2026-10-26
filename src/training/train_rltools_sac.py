# -*- coding: utf-8 -*-
"""
Continuous optimal pilot training with SAC (RLtools).

Trains a continuous policy with Soft Actor-Critic (SAC) using the RLtools Python
interface on CornerEnvContinuous, and exports the trained policy directly as
a C/C++ header (.h) for embedded deployment on the Crazyflie.
"""

import argparse
import os
import sys

# RLtools refuses to link Intel MKL when PyTorch is already imported (e.g. via
# torch.utils.tensorboard). Disable MKL before rltools is loaded to avoid the
# "PyTorch is imported which is known to cause issues..." RuntimeError.
os.environ.setdefault("RL_TOOLS_DISABLE_MKL", "1")

# Ensure the project root is in PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.envs.corner_env_continuous import CornerEnvContinuous

try:
    import gymnasium as gym
except ImportError:
    gym = None


def make_env_factory(corner_name: str = "cone_1", max_episode_steps: int = 1_000):
    """
    Create the environment factory required by the RLtools interface.
    Uses the singleton/persistent pattern to avoid multiple Webots Supervisor
    initializations within the same Python process.
    """
    _env_instance = None

    def env_factory():
        nonlocal _env_instance
        if _env_instance is None:
            _env_instance = CornerEnvContinuous(
                corner_name=corner_name,
                max_episode_steps=max_episode_steps,
                enable_trajectory_logging=False,
                verbose=0,
            )
            if gym is not None and hasattr(gym, "wrappers"):
                _env_instance = gym.wrappers.RescaleAction(
                    _env_instance, min_action=-1.0, max_action=1.0
                )
        return _env_instance

    return env_factory


def train_rltools_sac(
    max_steps: int = 100_000,
    seed: int = 7,
    gamma: float = 0.97,
    corner_name: str = "cone_1",
    max_episode_steps: int = 1_000,
    header_save_path: str = "./models/optimal_pilot_sac_checkpoint.h",
    log_dir: str = "./logs/sac",
    verbose: bool = True,
):
    """
    Execute the SAC training loop with RLtools.

    Args:
        max_steps: Maximum number of training steps to execute.
        seed: Random seed for numerical reproducibility.
        corner_name: DEF identifier of the target cone in Webots.
        max_episode_steps: Maximum step limit per episode.
        header_save_path: Path to the C header file (.h) to export the trained policy.
        log_dir: Directory to save TensorBoard logs and metrics.
        verbose: If True, prints periodic progress to the console.

    Returns:
        tuple: (final training state, C source code of the exported header).
    """
    try:
        from rltools import SAC
    except ImportError as e:
        raise ImportError(
            f"No se pudo importar rltools: {e}. "
            "Asegúrate de instalarlo con 'pip install rltools' (requiere C++ compiler y pybind11)."
        )

    os.makedirs(os.path.dirname(os.path.abspath(header_save_path)), exist_ok=True)

    # TensorBoard setup and hyperparameter logging
    writer = None
    if log_dir is not None:
        try:
            from torch.utils.tensorboard import SummaryWriter
            run_dir = os.path.join(log_dir, f"sac_seed{seed}")
            os.makedirs(run_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=run_dir)
            hparams = {
                "algorithm": "RLtools-SAC",
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

    print(f"[RLtools-SAC] Creando fábrica de entorno (esquina: {corner_name})...")
    env_factory = make_env_factory(corner_name=corner_name, max_episode_steps=max_episode_steps)

    print(f"[RLtools-SAC] Inicializando algoritmo SAC con semilla {seed} y gamma {gamma}...")
    eval_interval = min(1_000, max(1, max_steps // 10))
    sac = SAC(
        env_factory,
        STEP_LIMIT=max_steps,
        EPISODE_STEP_LIMIT=max_episode_steps,
        GAMMA=gamma,
        evaluation_interval=eval_interval,
        num_evaluation_episodes=5,
        verbose=verbose,
    )
    state = sac.State(seed)

    print(f"[RLtools-SAC] Iniciando bucle de entrenamiento (hasta {max_steps} pasos)...")
    step_count = 0
    finished = False

    while not finished and step_count < max_steps:
        finished = state.step()
        step_count += 1

        if step_count % 10_000 == 0 or finished:
            if verbose:
                print(f"[RLtools-SAC] Progreso: paso {step_count}/{max_steps} completado.")
            if writer is not None:
                writer.add_scalar("train/step", step_count, step_count)

    print(f"[RLtools-SAC] Entrenamiento finalizado. Exportando política a {header_save_path}...")
    policy_c_header = state.export_policy()

    with open(header_save_path, "w", encoding="utf-8") as f:
        f.write(policy_c_header)

    if writer is not None:
        writer.close()

    print(f"[RLtools-SAC] Header C exportado exitosamente en: {header_save_path}")
    return state, policy_c_header


def parse_args():
    """Command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Entrenamiento del piloto óptimo con SAC (RLtools) y exportación a header C."
    )
    parser.add_argument(
        "--steps",
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
    parser.add_argument("--seed", type=int, default=7, help="Semilla para reproducibilidad.")
    parser.add_argument(
        "--gamma", type=float, default=0.97, help="Factor de descuento gamma (por defecto 0.97)."
    )
    parser.add_argument(
        "--corner", type=str, default="cone_1", help="Nombre del cono objetivo en Webots."
    )
    parser.add_argument(
        "--export-path",
        type=str,
        default="./models/optimal_pilot_sac_checkpoint.h",
        help="Ruta destino para el header C exportado.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs/sac",
        help="Directorio de logs y TensorBoard para RLtools.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_rltools_sac(
        max_steps=args.steps,
        max_episode_steps=args.max_episode_steps,
        seed=args.seed,
        gamma=args.gamma,
        corner_name=args.corner,
        header_save_path=args.export_path,
        log_dir=args.log_dir,
    )
