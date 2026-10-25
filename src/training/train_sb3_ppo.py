# -*- coding: utf-8 -*-
"""
Continuous optimal pilot training with PPO (Stable-Baselines3).

Trains a continuous Gaussian policy with PPO on CornerEnvContinuous,
using the hyperparameters validated in the thesis (Diedrichs, 2025, Params2):
    - Learning rate: 1e-4
    - Discount factor (gamma): 0.97
    - Network architecture: MLP [64, 64]
    - GAE lambda: 0.95
    - Value function coefficient (vf_coef): 0.5
    - Monitoring and logging with TensorBoard and Monitor wrapper
"""

import argparse
import os
import sys

# Ensure the project root is in PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.envs.corner_env_continuous import CornerEnvContinuous


def train_sb3_ppo(
    total_timesteps: int = 100_000,
    seed: int = 7,
    learning_rate: float = 1e-4,
    gamma: float = 0.97,
    ent_coef: float = 0.01,
    corner_name: str = "cone_1",
    max_episode_steps: int = 1_000,
    log_dir: str = "./logs/ppo",
    model_save_path: str = "./models/ppo_optimal_pilot",
    checkpoint_freq: int = 20_000,
    verbose: int = 1,
    progress_bar: bool = True,
):
    """
    Execute the continuous PPO training loop with Stable-Baselines3.

    Args:
        total_timesteps: Total timesteps for training.
        seed: Random seed for numerical reproducibility.
        learning_rate: Learning rate of the Adam optimizer.
        gamma: Discount factor for future returns.
        corner_name: DEF identifier of the target cone in Webots ("cone_1").
        max_episode_steps: Maximum step limit per episode (default 1000).
        log_dir: Directory to save TensorBoard logs and Monitor metrics.
        model_save_path: Destination path for the trained model .zip file.
        checkpoint_freq: Step frequency for saving intermediate checkpoints.
        verbose: Standard output verbosity level (0: none, 1: info, 2: debug).
        progress_bar: If True, displays the progress bar during learn() (requires rich and tqdm).

    Returns:
        PPO: Trained model instance.
    """
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
        from stable_baselines3.common.logger import HParam
        from stable_baselines3.common.monitor import Monitor
    except ImportError as e:
        raise ImportError(
            f"No se pudo importar stable-baselines3: {e}. "
            "Asegúrate de tenerlo instalado con 'pip install stable-baselines3'."
        )

    class HParamCallback(BaseCallback):
        """Callback to log hyperparameters in the TensorBoard HPARAMS tab."""

        def __init__(self, hparams: dict, metrics: dict | None = None, verbose: int = 0):
            super().__init__(verbose)
            self.hparams = hparams
            self.metrics = metrics or {"rollout/ep_rew_mean": 0.0}

        def _on_training_start(self) -> None:
            self.logger.record(
                "hparams",
                HParam(self.hparams, self.metrics),
                exclude=("stdout", "log", "json", "csv"),
            )

        def _on_step(self) -> bool:
            return True

    # Create output directories
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(model_save_path)), exist_ok=True)

    print(f"[PPO Training] Inicializando CornerEnvContinuous (objetivo: {corner_name})...")
    raw_env = CornerEnvContinuous(
        corner_name=corner_name,
        max_episode_steps=max_episode_steps,
        enable_trajectory_logging=False,
        verbose=0,
    )
    env = Monitor(raw_env, filename=os.path.join(log_dir, "monitor.csv"))

    # Neural architecture configuration: 2 hidden layers of 64 neurons (Params2)
    # log_std_init=-0.5 (~0.61 std) avoids extreme [-1, 1] bang-bang saturation during early exploration
    policy_kwargs = dict(
        net_arch=dict(pi=[64, 64], vf=[64, 64]),
        log_std_init=-0.5,
    )

    print(
        f"[PPO Training] Configurando PPO: lr={learning_rate}, gamma={gamma}, ent_coef={ent_coef}, "
        f"timesteps={total_timesteps}, seed={seed}"
    )
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        gamma=gamma,
        gae_lambda=0.95,
        ent_coef=ent_coef,
        vf_coef=0.5,
        seed=seed,
        tensorboard_log=log_dir,
        policy_kwargs=policy_kwargs,
        verbose=verbose,
    )

    # Training callbacks
    callbacks = []

    # 1. Hyperparameter logging in TensorBoard (HPARAMS)
    hparams_dict = {
        "algorithm": "SB3-PPO",
        "learning_rate": learning_rate,
        "gamma": gamma,
        "gae_lambda": 0.95,
        "ent_coef": ent_coef,
        "vf_coef": 0.5,
        "seed": seed,
        "total_timesteps": total_timesteps,
        "corner_name": corner_name,
        "max_episode_steps": max_episode_steps,
    }
    metrics_dict = {
        "rollout/ep_rew_mean": 0.0,
    }
    callbacks.append(HParamCallback(hparams=hparams_dict, metrics=metrics_dict, verbose=verbose))

    # 2. Optional periodic checkpoints
    if checkpoint_freq > 0:
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=log_dir,
            name_prefix="ppo_checkpoint",
            verbose=verbose,
        )
        callbacks.append(checkpoint_callback)

    # Check availability of libraries for the progress bar
    if progress_bar:
        try:
            import rich  # noqa: F401
            import tqdm  # noqa: F401
        except ImportError:
            print(
                "[PPO Training] Aviso: tqdm y rich son requeridos para la barra de progreso. Desactivando progress_bar."
            )
            progress_bar = False

    print(f"[PPO Training] Iniciando entrenamiento por {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        tb_log_name=f"ppo_lr{learning_rate}_gamma{gamma}_seed{seed}",
        progress_bar=progress_bar,
    )

    print(f"[PPO Training] Guardando modelo final en: {model_save_path}.zip")
    model.save(model_save_path)
    env.close()

    print("[PPO Training] Entrenamiento completado con éxito.")
    return model


def parse_args():
    """Command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Entrenamiento de piloto óptimo con PPO (Stable-Baselines3)."
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100_000,
        help="Número total de pasos de simulación (timesteps).",
    )
    parser.add_argument("--seed", type=int, default=7, help="Semilla para reproducibilidad.")
    parser.add_argument("--verbose", type=int, default=1, help="Nivel de logging: 0, 1 o 2.")
    parser.add_argument(
        "--progress-bar",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Muestra u oculta la barra de progreso en consola (requiere rich y tqdm).",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Tasa de aprendizaje.")
    parser.add_argument("--gamma", type=float, default=0.97, help="Factor de descuento gamma.")
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.01,
        help="Coeficiente de entropía para fomentar la exploración continua (por defecto 0.01).",
    )
    parser.add_argument(
        "--corner", type=str, default="cone_1", help="Nombre del cono objetivo (cone_1)."
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=1000,
        help="Límite máximo de pasos por episodio para CornerEnvContinuous (por defecto 1000).",
    )
    parser.add_argument(
        "--log-dir", type=str, default="./logs/ppo", help="Directorio de logs y TensorBoard."
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="./models/ppo_optimal_pilot",
        help="Ruta destino para guardar el modelo .zip.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_sb3_ppo(
        total_timesteps=args.timesteps,
        seed=args.seed,
        learning_rate=args.lr,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        corner_name=args.corner,
        max_episode_steps=args.max_episode_steps,
        log_dir=args.log_dir,
        model_save_path=args.save_path,
        verbose=args.verbose,
        progress_bar=args.progress_bar,
    )
