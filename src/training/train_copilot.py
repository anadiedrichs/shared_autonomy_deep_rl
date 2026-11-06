# -*- coding: utf-8 -*-
"""
Training script for the Constrained Residual Copilot (Schaff & Walter, 2020).

Trains a residual policy pi_r(s_t, a_h) using ConstrainedPPO with an adaptive Lagrange
multiplier to minimally assist a surrogate pilot (Noisy, Laggy, StandBy, Heuristic, Optimal):
    a_total = a_h + beta * a_r

Features:
1. Warm-Up Phase:
   Forces a_r = 0 for the first N timesteps (e.g. 10_000 to 100_000) so the Critic
   pre-learns the value function V(s, a_h) on unassisted surrogate trajectories.
2. Constrained Optimization:
   Adaptive dual Lagrange multiplier balances safety satisfaction against minimal intervention.
3. Full TensorFlow / TensorBoard Logging:
   Logs loss_lambda, lambda, mean_residual_norm, ppo_loss, and episode rewards to .tfevents.
4. Reproducible Model and Checkpoint Saving:
   Saves models to ./models/copilot_constrained_ppo.zip and logs hyperparameters in JSON.
"""

import argparse
import json
import os
import sys

# Ensure repository root is on sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np

try:
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
except ImportError as err:
    raise ImportError(
        f"Stable-Baselines3 is required to train the copilot: {err}. "
        "Please install it via 'pip install stable-baselines3'."
    )

from src.copilot.constrained_ppo import ConstrainedPPO
from src.envs.shared_autonomy_env import SharedAutonomyEnv
from src.pilots.heuristic_pilot import HeuristicPilot
from src.pilots.laggy_pilot import LaggyPilot
from src.pilots.noisy_pilot import NoisyPilot
from src.pilots.optimal_pilot import OptimalPilot
from src.pilots.standby_pilot import StandByPilot


class ResidualWarmupCallback(BaseCallback):
    """
    Forces residual actions to zero (a_r = 0) during the initial warm-up timesteps
    by setting residual_scale = 0.0 on the underlying SharedAutonomyEnv.
    """

    def __init__(
        self,
        warmup_steps: int = 10_000,
        target_residual_scale: float = 1.0,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.warmup_steps = int(warmup_steps)
        self.target_residual_scale = float(target_residual_scale)
        self._warmup_logged = False

    def _on_step(self) -> bool:
        # Resolve the underlying SharedAutonomyEnv through Gym/SB3 wrappers
        env = self.training_env.envs[0]
        while hasattr(env, "env"):
            if isinstance(env, SharedAutonomyEnv):
                break
            env = env.env

        if not isinstance(env, SharedAutonomyEnv):
            return True

        if self.num_timesteps < self.warmup_steps:
            # During warm-up: force residual intervention to zero
            env.residual_scale = 0.0
            if not self._warmup_logged and self.verbose > 0:
                print(
                    f"[Warm-Up] Fase de calentamiento activa (timesteps < {self.warmup_steps}). "
                    "Forzando a_r = 0 para pre-entrenamiento del Crítico..."
                )
                self._warmup_logged = True
            self.logger.record("train/is_warmup", 1.0)
        else:
            # After warm-up: restore authority scaling factor beta
            if env.residual_scale != self.target_residual_scale:
                env.residual_scale = self.target_residual_scale
                if self.verbose > 0:
                    print(
                        f"\n[Warm-Up] Fin del calentamiento en t={self.num_timesteps}. "
                        f"Restaurando beta = {self.target_residual_scale}. Copiloto activo."
                    )
            self.logger.record("train/is_warmup", 0.0)

        return True


def create_surrogate_pilot(
    pilot_type: str,
    checkpoint: str | None = None,
    backend: str = "sb3_ppo",
    noise_std: float = 0.50,
    lag_prob: float = 0.80,
    corner_name: str = "cone_1",
    seed: int = 42,
):
    """Factory creating the surrogate pilot acting as the human operator."""
    pt = pilot_type.lower().strip()

    if pt == "optimal":
        return OptimalPilot(backend=backend, checkpoint_path=checkpoint, deterministic=True)
    elif pt == "heuristic":
        return HeuristicPilot(mode="proportional", corner_name=corner_name)
    elif pt == "noisy":
        base = OptimalPilot(backend=backend, checkpoint_path=checkpoint, deterministic=True)
        return NoisyPilot(optimal_pilot=base, sigma=noise_std, seed=seed)
    elif pt == "laggy":
        base = OptimalPilot(backend=backend, checkpoint_path=checkpoint, deterministic=True)
        return LaggyPilot(optimal_pilot=base, p_repeat=lag_prob, seed=seed)
    elif pt == "standby":
        base = OptimalPilot(backend=backend, checkpoint_path=checkpoint, deterministic=True)
        return StandByPilot(optimal_pilot=base, p_drop=lag_prob, seed=seed)
    else:
        raise ValueError(
            f"Tipo de piloto desconocido: '{pilot_type}'. "
            "Opciones: noisy, laggy, standby, heuristic, optimal."
        )


def train_constrained_copilot(
    total_timesteps: int = 100_000,
    warmup_steps: int = 10_000,
    pilot_type: str = "noisy",
    optimal_checkpoint: str | None = None,
    optimal_backend: str = "sb3_ppo",
    noise_std: float = 0.50,
    lag_prob: float = 0.80,
    target_threshold: float = 0.0,
    init_lambda: float = 1.0,
    lr_lambda: float = 3e-4,
    learning_rate: float = 3e-4,
    residual_scale: float = 1.0,
    batch_size: int = 64,
    gamma: float = 0.97,
    corner_name: str = "cone_1",
    max_episode_steps: int = 1_000,
    save_path: str = "./models/copilot_constrained_ppo.zip",
    log_dir: str = "./logs/copilot_constrained_ppo",
    checkpoint_freq: int = 20_000,
    seed: int = 42,
    verbose: int = 1,
):
    """Executes the constrained residual policy training workflow."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 1. Instantiate the surrogate pilot
    print(f"[Copilot Training] Instanciando piloto primario: {pilot_type.upper()}")
    pilot = create_surrogate_pilot(
        pilot_type=pilot_type,
        checkpoint=optimal_checkpoint,
        backend=optimal_backend,
        noise_std=noise_std,
        lag_prob=lag_prob,
        corner_name=corner_name,
        seed=seed,
    )

    # 2. Build the SharedAutonomyEnv (augmented observation space s_ext = [s_t, a_h])
    print(
        f"[Copilot Training] Inicializando SharedAutonomyEnv (objetivo: {corner_name}, "
        f"beta={residual_scale})..."
    )
    raw_env = SharedAutonomyEnv(
        pilot=pilot,
        copilot=None,  # RL agent learns the copilot policy directly
        corner_name=corner_name,
        max_episode_steps=max_episode_steps,
        residual_scale=residual_scale,
        reward_mode="safety",  # Goal-agnostic general safety reward
        include_pilot_action_in_obs=True,  # 15 dimensions: [s_t (11), a_h (4)]
        verbose=0,
    )
    env = Monitor(raw_env, filename=os.path.join(log_dir, "monitor.csv"))

    # 3. Configure network architecture (MLP [64, 64] for actor and critic)
    policy_kwargs = dict(
        net_arch=dict(pi=[64, 64], vf=[64, 64]),
    )

    print(
        f"[Copilot Training] Configurando ConstrainedPPO:\n"
        f"  Total timesteps   : {total_timesteps}\n"
        f"  Warm-up steps     : {warmup_steps}\n"
        f"  Target threshold  : {target_threshold}\n"
        f"  Init lambda       : {init_lambda} (lr_lambda: {lr_lambda})\n"
        f"  Learning rate     : {learning_rate}\n"
        f"  Gamma             : {gamma}\n"
        f"  TensorBoard dir   : {log_dir}\n"
        f"  Save path         : {save_path}"
    )

    # 4. Instantiate ConstrainedPPO with TensorBoard logging (.tfevents)
    model = ConstrainedPPO(
        policy="MlpPolicy",
        env=env,
        target_threshold=target_threshold,
        init_lambda=init_lambda,
        lr_lambda=lr_lambda,
        learning_rate=learning_rate,
        gamma=gamma,
        batch_size=batch_size,
        policy_kwargs=policy_kwargs,
        tensorboard_log=log_dir,
        seed=seed,
        verbose=verbose,
    )

    # 5. Build callbacks
    callbacks = []
    # Warm-up callback (a_r = 0 during warm-up timesteps)
    if warmup_steps > 0:
        callbacks.append(
            ResidualWarmupCallback(
                warmup_steps=warmup_steps,
                target_residual_scale=residual_scale,
                verbose=verbose,
            )
        )

    # Periodic checkpoint callback
    if checkpoint_freq > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=checkpoint_freq,
                save_path=log_dir,
                name_prefix="copilot_constrained_step",
                verbose=verbose,
            )
        )

    # 6. Execute training
    print("\n[Copilot Training] Iniciando entrenamiento con ConstrainedPPO...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    # 7. Save trained model and hyperparameters
    print(f"\n[Copilot Training] Guardando modelo en: {save_path}")
    model.save(save_path)

    hparams_path = save_path.replace(".zip", "_hparams.json")
    metadata = {
        "algorithm": "ConstrainedPPO",
        "reference": "Schaff & Walter (RSS 2020)",
        "pilot_type": pilot_type,
        "total_timesteps": total_timesteps,
        "warmup_steps": warmup_steps,
        "target_threshold": target_threshold,
        "final_lambda": float(model.lambda_val.item()),
        "learning_rate": learning_rate,
        "lr_lambda": lr_lambda,
        "residual_scale": residual_scale,
        "gamma": gamma,
        "batch_size": batch_size,
        "corner_name": corner_name,
        "seed": seed,
    }
    with open(hparams_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"[Copilot Training] Hiperparámetros guardados en: {hparams_path}")

    env.close()
    return model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Entrenamiento del Copiloto Residual con ConstrainedPPO (Schaff & Walter, 2020)."
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100_000,
        help="Número total de timesteps de entrenamiento (por defecto: 100000).",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=10_000,
        help="Timesteps de calentamiento forzando a_r = 0 para el Crítico (por defecto: 10000).",
    )
    parser.add_argument(
        "--pilot-type",
        type=str,
        default="noisy",
        choices=["noisy", "laggy", "standby", "heuristic", "optimal"],
        help="Tipo de piloto humano/subrogado a asistir (por defecto: noisy).",
    )
    parser.add_argument(
        "--optimal-checkpoint",
        type=str,
        default="./models/ppo_optimal_pilot.zip",
        help="Ruta al checkpoint del piloto óptimo base.",
    )
    parser.add_argument(
        "--optimal-backend",
        type=str,
        default="sb3_ppo",
        choices=["sb3_ppo", "rltools_sac", "rltools_td3"],
        help="Backend para el piloto óptimo base.",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.50,
        help="Desviación sigma del ruido para NoisyPilot (por defecto: 0.50).",
    )
    parser.add_argument(
        "--lag-prob",
        type=float,
        default=0.80,
        help="Probabilidad de repetición/omisión para LaggyPilot o StandByPilot (por defecto: 0.80).",
    )
    parser.add_argument(
        "--target-threshold",
        type=float,
        default=0.0,
        help="Umbral target_d de la restricción de retorno esperado (por defecto: 0.0).",
    )
    parser.add_argument(
        "--init-lambda",
        type=float,
        default=1.0,
        help="Valor inicial para el multiplicador de Lagrange lambda (por defecto: 1.0).",
    )
    parser.add_argument(
        "--lr-lambda",
        type=float,
        default=3e-4,
        help="Tasa de aprendizaje para el optimizador de lambda (por defecto: 3e-4).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Tasa de aprendizaje del actor y crítico (por defecto: 3e-4).",
    )
    parser.add_argument(
        "--residual-scale",
        type=float,
        default=1.0,
        help="Factor de autoridad/escala residual beta (por defecto: 1.0).",
    )
    parser.add_argument(
        "--corner",
        type=str,
        default="cone_1",
        help="Nombre del cono objetivo en Webots (por defecto: cone_1).",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="./models/copilot_constrained_ppo.zip",
        help="Ruta donde guardar el modelo entrenado .zip.",
    )
    parser.add_argument(
        "--tensorboard-log",
        type=str,
        default="./logs/copilot_constrained_ppo",
        help="Directorio de logs para TensorBoard (.tfevents de TensorFlow).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla aleatoria (por defecto: 42).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_constrained_copilot(
        total_timesteps=args.timesteps,
        warmup_steps=args.warmup_steps,
        pilot_type=args.pilot_type,
        optimal_checkpoint=args.optimal_checkpoint,
        optimal_backend=args.optimal_backend,
        noise_std=args.noise_std,
        lag_prob=args.lag_prob,
        target_threshold=args.target_threshold,
        init_lambda=args.init_lambda,
        lr_lambda=args.lr_lambda,
        learning_rate=args.lr,
        residual_scale=args.residual_scale,
        corner_name=args.corner,
        save_path=args.save_path,
        log_dir=args.tensorboard_log,
        seed=args.seed,
    )
