# -*- coding: utf-8 -*-
"""
Script de entrenamiento para la política residual del copiloto (Autonomía Compartida).

Entrena una política residual pi_r(s, a_h) -> a_r contra un piloto humano o sintético
congelado (NoisyPilot, LaggyPilot, HeuristicPilot u OptimalPilot), utilizando la función de
recompensa goal-agnostic general_safety_reward() (Schaff & Walter, 2020).

Soporta algoritmos PPO y SAC mediante Stable-Baselines3, con registro en TensorBoard.
"""

import argparse
import os
import sys
import numpy as np

# Asegurar que el directorio raíz del proyecto esté en PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.envs.shared_autonomy_env import SharedAutonomyEnv
from src.pilots.heuristic_pilot import HeuristicPilot
from src.pilots.laggy_pilot import LaggyPilot
from src.pilots.noisy_pilot import NoisyPilot
from src.pilots.optimal_pilot import OptimalPilot


def create_base_pilot_for_training(
    pilot_type: str = "noisy",
    corner_name: str = "cone_1",
    noise_std: float = 0.20,
    lag_prob: float = 0.30,
    heuristic_mode: str = "proportional",
    optimal_model_path: str | None = None,
    seed: int = 42,
):
    """
    Crea el piloto base (congelado) con el que interactuará el copiloto durante el entrenamiento.

    Args:
        pilot_type: Tipo de piloto ("noisy", "laggy", "heuristic", "optimal").
        corner_name: Nombre del cono objetivo ("cone_1", etc.).
        noise_std: Desviación típica del ruido gaussiano para NoisyPilot.
        lag_prob: Probabilidad de retardo/repetición para LaggyPilot.
        heuristic_mode: Modo de HeuristicPilot ("proportional" u "open_loop").
        optimal_model_path: Ruta al checkpoint entrenado para OptimalPilot.
        seed: Semilla para reproducibilidad de los pilotos estocásticos.

    Returns:
        BasePilot: Instancia del piloto congelado listo para usar en SharedAutonomyEnv.
    """
    pilot_type = pilot_type.lower()

    if pilot_type == "heuristic":
        return HeuristicPilot(mode=heuristic_mode, corner_name=corner_name)

    elif pilot_type == "optimal":
        if optimal_model_path is None:
            # Si no se indica ruta, se usa HeuristicPilot como aproximación óptima determinista
            return HeuristicPilot(mode="proportional", corner_name=corner_name)
        return OptimalPilot(checkpoint_path=optimal_model_path)

    elif pilot_type == "noisy":
        # Piloto de referencia base sobre el cual aplicar ruido
        if optimal_model_path is not None and os.path.exists(optimal_model_path):
            base_pilot = OptimalPilot(checkpoint_path=optimal_model_path)
        else:
            base_pilot = HeuristicPilot(mode=heuristic_mode, corner_name=corner_name)
        return NoisyPilot(optimal_pilot=base_pilot, sigma=noise_std, seed=seed)

    elif pilot_type == "laggy":
        # Piloto de referencia base sobre el cual aplicar latencia
        if optimal_model_path is not None and os.path.exists(optimal_model_path):
            base_pilot = OptimalPilot(checkpoint_path=optimal_model_path)
        else:
            base_pilot = HeuristicPilot(mode=heuristic_mode, corner_name=corner_name)
        return LaggyPilot(optimal_pilot=base_pilot, p_repeat=lag_prob, seed=seed)

    else:
        raise ValueError(
            f"Tipo de piloto desconocido: '{pilot_type}'. "
            "Opciones válidas: 'noisy', 'laggy', 'heuristic', 'optimal'."
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Entrenamiento de la política residual del copiloto con Stable-Baselines3."
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="ppo",
        choices=["ppo", "sac"],
        help="Algoritmo de RL para entrenar el copiloto: 'ppo' o 'sac' (por defecto: ppo).",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100_000,
        help="Número total de timesteps de entrenamiento (por defecto: 100000).",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=1_000,
        help="Límite máximo de pasos por episodio (por defecto: 1000).",
    )
    parser.add_argument(
        "--pilot-type",
        type=str,
        default="noisy",
        choices=["noisy", "laggy", "heuristic", "optimal"],
        help="Tipo de piloto humano/sintético congelado (por defecto: noisy).",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.20,
        help="Desviación típica sigma del ruido para NoisyPilot (por defecto: 0.20).",
    )
    parser.add_argument(
        "--lag-prob",
        type=float,
        default=0.30,
        help="Probabilidad de repetición p_repeat para LaggyPilot (por defecto: 0.30).",
    )
    parser.add_argument(
        "--heuristic-mode",
        type=str,
        default="proportional",
        choices=["proportional", "open_loop"],
        help="Modo del HeuristicPilot si se usa como base (por defecto: proportional).",
    )
    parser.add_argument(
        "--optimal-model-path",
        type=str,
        default=None,
        help="Ruta al modelo entrenado de OptimalPilot si pilot-type=optimal o para base de noisy/laggy.",
    )
    parser.add_argument(
        "--corner",
        type=str,
        default="cone_1",
        help="Cono objetivo del escenario (por defecto: cone_1).",
    )
    parser.add_argument(
        "--reward-mode",
        type=str,
        default="safety",
        choices=["safety", "goal_directed", "combined"],
        help="Tipo de recompensa: 'safety' (goal-agnostic), 'goal_directed' o 'combined' (por defecto: safety).",
    )
    parser.add_argument(
        "--lambda-intervention",
        type=float,
        default=0.10,
        help="Factor de penalización por intervención ||a_r||^2 (por defecto: 0.10).",
    )
    parser.add_argument(
        "--residual-scale",
        type=float,
        default=1.0,
        help="Factor de atenuación para la acción residual (por defecto: 1.0).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Tasa de aprendizaje (por defecto: 3e-4).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Tamaño de lote (por defecto: 64 para PPO, 256 para SAC).",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Ruta de guardado del modelo .zip entrenado (por defecto: models/copilot_residual_<algo>.zip).",
    )
    parser.add_argument(
        "--tensorboard-log",
        type=str,
        default="logs/copilot",
        help="Directorio de logs para TensorBoard (por defecto: logs/copilot).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla aleatoria (por defecto: 42).",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Nivel de verbosidad (0: silencioso, 1: info estándar).",
    )
    return parser.parse_args()


def train():
    args = parse_args()

    # Configuración de rutas por defecto
    if args.save_path is None:
        save_dir = os.path.join(project_root, "models")
        os.makedirs(save_dir, exist_ok=True)
        args.save_path = os.path.join(
            save_dir, f"copilot_residual_{args.algorithm.lower()}_{args.pilot_type}.zip"
        )
    else:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    os.makedirs(args.tensorboard_log, exist_ok=True)

    print(f"\n{'='*70}")
    print(" ENTRENAMIENTO DE COPILOTO RESIDUAL (AUTONOMÍA COMPARTIDA)")
    print(f"{'='*70}")
    print(f" Algoritmo:                {args.algorithm.upper()}")
    print(f" Piloto congelado:         {args.pilot_type.upper()}")
    if args.pilot_type == "noisy":
        print(f"   - Ruido sigma:          {args.noise_std}")
    elif args.pilot_type == "laggy":
        print(f"   - Lag p_repeat:         {args.lag_prob}")
    print(f" Modo de recompensa:       {args.reward_mode}")
    print(f" Lambda intervención:      {args.lambda_intervention}")
    print(f" Escala residual:          {args.residual_scale}")
    print(f" Timesteps totales:        {args.timesteps:,}")
    print(f" Objetivo (esquina):       {args.corner}")
    print(f" Guardado de modelo en:    {args.save_path}")
    print(f" TensorBoard logdir:       {args.tensorboard_log}")
    print(f"{'='*70}\n")

    # 1. Crear piloto congelado
    pilot = create_base_pilot_for_training(
        pilot_type=args.pilot_type,
        corner_name=args.corner,
        noise_std=args.noise_std,
        lag_prob=args.lag_prob,
        heuristic_mode=args.heuristic_mode,
        optimal_model_path=args.optimal_model_path,
        seed=args.seed,
    )

    # 2. Instanciar entorno de autonomía compartida
    env = SharedAutonomyEnv(
        pilot=pilot,
        copilot=None,  # El algoritmo RL actuará como copiloto en entrenamiento
        max_episode_steps=args.max_episode_steps,
        corner_name=args.corner,
        reward_mode=args.reward_mode,
        lambda_intervention=args.lambda_intervention,
        residual_scale=args.residual_scale,
        include_pilot_action_in_obs=True,  # Obs extendida: [obs(11), a_h(4)] -> 15 dimensiones
        enable_trajectory_logging=False,
        verbose=0,
    )

    # 3. Configurar algoritmo de aprendizaje por refuerzo
    try:
        from stable_baselines3 import PPO, SAC
    except ImportError:
        raise ImportError(
            "stable-baselines3 no está instalado. Ejecuta: pip install stable-baselines3"
        )

    algo_name = args.algorithm.lower()
    run_name = f"copilot_{algo_name}_{args.pilot_type}_s{args.seed}"

    if algo_name == "ppo":
        batch_size = args.batch_size if args.batch_size is not None else 64
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=args.lr,
            n_steps=2048,
            batch_size=batch_size,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.005,
            verbose=args.verbose,
            tensorboard_log=args.tensorboard_log,
            seed=args.seed,
        )
    elif algo_name == "sac":
        batch_size = args.batch_size if args.batch_size is not None else 256
        model = SAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=args.lr,
            buffer_size=100_000,
            batch_size=batch_size,
            gamma=0.99,
            tau=0.005,
            ent_coef="auto",
            verbose=args.verbose,
            tensorboard_log=args.tensorboard_log,
            seed=args.seed,
        )
    else:
        raise ValueError(f"Algoritmo no soportado: {algo_name}")

    # 4. Bucle de aprendizaje
    print(f"Comenzando entrenamiento de {algo_name.upper()}...")
    try:
        model.learn(
            total_timesteps=args.timesteps,
            tb_log_name=run_name,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nEntrenamiento interrumpido manualmente por el usuario.")

    # 5. Guardar modelo entrenado
    model.save(args.save_path)
    print(f"\n¡Entrenamiento finalizado! Modelo de copiloto guardado en: {args.save_path}")

    env.close()


if __name__ == "__main__":
    train()
