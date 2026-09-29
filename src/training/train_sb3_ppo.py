# -*- coding: utf-8 -*-
"""
Entrenamiento del piloto óptimo continuo con PPO (Stable-Baselines3).

Entrena una política gaussiana continua con PPO sobre CornerEnvContinuous,
utilizando los hiperparámetros validados en la tesis (Diedrichs, 2025, Params2):
    - Tasa de aprendizaje: 1e-4
    - Factor de descuento (gamma): 0.97
    - Arquitectura de la red: MLP [64, 64]
    - GAE lambda: 0.95
    - Coeficiente de valor (vf_coef): 0.5
    - Monitoreo y registro con TensorBoard y Monitor wrapper
"""

import argparse
import os
import sys

# Asegurar que la raíz del proyecto esté en el PYTHONPATH
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
):
    """
    Ejecuta el ciclo de entrenamiento continuo de PPO con Stable-Baselines3.

    Args:
        total_timesteps: Pasos de tiempo totales para el entrenamiento.
        seed: Semilla para reproducibilidad numérica.
        learning_rate: Tasa de aprendizaje del optimizador Adam.
        gamma: Factor de descuento para retornos futuros.
        corner_name: Identificador DEF del cono objetivo en Webots ("cone_1").
        max_episode_steps: Límite máximo de pasos por episodio (por defecto 1000).
        log_dir: Directorio para guardar logs de TensorBoard y métricas de Monitor.
        model_save_path: Ruta destino para el archivo .zip del modelo entrenado.
        checkpoint_freq: Frecuencia de pasos para guardar checkpoints intermedios.
        verbose: Nivel de detalle en salida estándar (0: nada, 1: info, 2: debug).

    Returns:
        PPO: Instancia del modelo entrenado.
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
        """Callback para registrar hiperparámetros en la pestaña HPARAMS de TensorBoard."""

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

    # Crear directorios de salida
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

    # Configuración de arquitectura neuronal: 2 capas ocultas de 64 neuronas (Params2)
    policy_kwargs = dict(
        net_arch=dict(pi=[64, 64], vf=[64, 64])
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

    # Callbacks de entrenamiento
    callbacks = []

    # 1. Registro de hiperparámetros en TensorBoard (HPARAMS)
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

    # 2. Checkpoints periódicos opcionales
    if checkpoint_freq > 0:
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=log_dir,
            name_prefix="ppo_checkpoint",
            verbose=verbose,
        )
        callbacks.append(checkpoint_callback)

    print(f"[PPO Training] Iniciando entrenamiento por {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        tb_log_name=f"ppo_lr{learning_rate}_gamma{gamma}_seed{seed}",
        progress_bar=False,
    )

    print(f"[PPO Training] Guardando modelo final en: {model_save_path}.zip")
    model.save(model_save_path)
    env.close()

    print("[PPO Training] Entrenamiento completado con éxito.")
    return model


def parse_args():
    """Analizador de argumentos de línea de comandos."""
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
    )
