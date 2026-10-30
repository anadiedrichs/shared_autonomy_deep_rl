# -*- coding: utf-8 -*-
"""
Training script for the copilot residual policy (Shared Autonomy).

Trains a residual policy pi_r(s, a_h) -> a_r against a frozen human or synthetic
pilot (NoisyPilot, LaggyPilot, HeuristicPilot, or OptimalPilot), using the
goal-agnostic reward function general_safety_reward() (Schaff & Walter, 2020).

Supports PPO and SAC algorithms via Stable-Baselines3, with TensorBoard logging.
"""

import argparse
import os
import sys
import numpy as np

# Ensure the project root directory is in PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.envs.shared_autonomy_env import SharedAutonomyEnv
from src.pilots.heuristic_pilot import HeuristicPilot
from src.pilots.laggy_pilot import LaggyPilot
from src.pilots.noisy_pilot import NoisyPilot
from src.pilots.optimal_pilot import OptimalPilot


try:
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError:  # pragma: no cover
    BaseCallback = object


class BestTrainingRewardCallback(BaseCallback):
    """
    Saves the best model according to the moving average of training episode rewards.

    Needed because Webots allows a single Robot instance per process, so a separate
    evaluation environment (as used by EvalCallback) cannot be created.
    """

    def __init__(self, check_freq: int, window: int, save_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = int(check_freq)
        self.window = max(int(window), 1)
        self.save_dir = save_dir
        self.best_mean = -np.inf
        self.episode_rewards: list[float] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is not None:
                self.episode_rewards.append(float(ep["r"]))

        if self.n_calls % self.check_freq == 0 and len(self.episode_rewards) >= self.window:
            mean_r = float(np.mean(self.episode_rewards[-self.window:]))
            if mean_r > self.best_mean:
                self.best_mean = mean_r
                self.model.save(os.path.join(self.save_dir, "best_model.zip"))
                if self.verbose >= 1:
                    print(
                        f"[Checkpoint] Nuevo mejor modelo (paso {self.num_timesteps}): "
                        f"recompensa media últimos {self.window} eps = {mean_r:.2f}"
                    )
        return True


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
    Create the base (frozen) pilot that the copilot will interact with during training.

    Args:
        pilot_type: Pilot type ("noisy", "laggy", "heuristic", "optimal").
        corner_name: Target cone name ("cone_1", etc.).
        noise_std: Standard deviation of Gaussian noise for NoisyPilot.
        lag_prob: Delay/repetition probability for LaggyPilot.
        heuristic_mode: HeuristicPilot mode ("proportional" or "open_loop").
        optimal_model_path: Path to the trained checkpoint for OptimalPilot.
        seed: Random seed for reproducibility of stochastic pilots.

    Returns:
        BasePilot: Frozen pilot instance ready for use in SharedAutonomyEnv.
    """
    pilot_type = pilot_type.lower()

    # Resolve optimal_model_path relative to cwd and project_root
    resolved_path = optimal_model_path
    if resolved_path is not None and not os.path.exists(resolved_path):
        candidate = os.path.join(project_root, resolved_path)
        if os.path.exists(candidate):
            resolved_path = candidate

    if pilot_type == "heuristic":
        return HeuristicPilot(mode=heuristic_mode, corner_name=corner_name)

    elif pilot_type == "optimal":
        if resolved_path is None or not os.path.exists(resolved_path):
            # If no path is provided or doesn't exist, use HeuristicPilot
            print(
                f"[Copilot Training] Advertencia: Modelo óptimo no encontrado ({optimal_model_path}). Usando HeuristicPilot."
            )
            return HeuristicPilot(mode="proportional", corner_name=corner_name)
        print(f"[Copilot Training] Base pilot: OptimalPilot ({resolved_path})")
        return OptimalPilot(checkpoint_path=resolved_path)

    elif pilot_type == "noisy":
        # Base reference pilot on top of which noise is applied
        if resolved_path is not None and os.path.exists(resolved_path):
            print(f"[Copilot Training] Base pilot de NoisyPilot: OptimalPilot ({resolved_path})")
            base_pilot = OptimalPilot(checkpoint_path=resolved_path)
        else:
            print(f"[Copilot Training] Base pilot de NoisyPilot: HeuristicPilot ({heuristic_mode})")
            base_pilot = HeuristicPilot(mode=heuristic_mode, corner_name=corner_name)
        return NoisyPilot(optimal_pilot=base_pilot, sigma=noise_std, seed=seed)

    elif pilot_type == "laggy":
        # Base reference pilot on top of which latency is applied
        if resolved_path is not None and os.path.exists(resolved_path):
            print(f"[Copilot Training] Base pilot de LaggyPilot: OptimalPilot ({resolved_path})")
            base_pilot = OptimalPilot(checkpoint_path=resolved_path)
        else:
            print(f"[Copilot Training] Base pilot de LaggyPilot: HeuristicPilot ({heuristic_mode})")
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
        default=0.50,
        help="Factor de penalización por intervención ||a_r||^2 (por defecto: 0.50).",
    )
    parser.add_argument(
        "--action-smoothness-weight",
        type=float,
        default=0.05,
        help="Factor de penalización por brusquedad/jerk ||a_r(t) - a_r(t-1)||^2 (por defecto: 0.05).",
    )
    parser.add_argument(
        "--obstacle-safe-distance",
        type=float,
        default=0.35,
        help="Distancia de seguridad a obstáculos en metros para zona muerta (por defecto: 0.35 m).",
    )
    parser.add_argument(
        "--safe-altitude",
        type=float,
        default=0.30,
        help="Altitud de seguridad en metros por debajo de la cual se activa penalización de suelo (por defecto: 0.30 m).",
    )
    parser.add_argument(
        "--proximity-mode",
        type=str,
        default="quadratic",
        choices=["quadratic", "exponential"],
        help="Forma funcional de la penalización de proximidad: 'quadratic' (con zona muerta) o 'exponential' (por defecto: quadratic).",
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
        "--ent-coef",
        type=str,
        default="auto",
        help="Coeficiente de entropía para SAC: 'auto' o valor float (e.g. '0.01') (por defecto: auto).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Tamaño de lote (por defecto: 64 para PPO, 256 para SAC).",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=0,
        help="Cada cuántos timesteps comprobar y guardar el mejor modelo (recompensa media móvil de entrenamiento); 0 desactiva.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=5,
        help="Ventana de episodios para la recompensa media móvil del mejor modelo (por defecto: 5).",
    )
    parser.add_argument(
        "--altitude-weight",
        type=float,
        default=1.0,
        help="Peso de la penalización de proximidad al suelo en la recompensa de seguridad (por defecto: 1.0).",
    )
    parser.add_argument(
        "--altitude-decay",
        type=float,
        default=0.10,
        help="Decaimiento espacial en metros para la penalización de altitud (por defecto: 0.10).",
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

    # Default path configuration
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
    print(f" Forma de proximidad:      {args.proximity_mode} (d_safe: {args.obstacle_safe_distance}m, z_safe: {args.safe_altitude}m)")
    print(f" Lambda intervención:      {args.lambda_intervention}")
    print(f" Peso jerk / suavidad:     {args.action_smoothness_weight}")
    print(f" Escala residual:          {args.residual_scale}")
    print(f" Peso penalización suelo:  {args.altitude_weight}")
    if args.algorithm.lower() == "sac":
        print(f" SAC ent_coef:             {args.ent_coef}")
    if args.eval_freq > 0:
        print(f" Checkpoint mejor modelo:  cada {args.eval_freq} timesteps ({args.eval_episodes} eps)")
    print(f" Timesteps totales:        {args.timesteps:,}")
    print(f" Objetivo (esquina):       {args.corner}")
    print(f" Guardado de modelo en:    {args.save_path}")
    print(f" TensorBoard logdir:       {args.tensorboard_log}")
    print(f"{'='*70}\n")

    # 1. Create frozen pilot
    pilot = create_base_pilot_for_training(
        pilot_type=args.pilot_type,
        corner_name=args.corner,
        noise_std=args.noise_std,
        lag_prob=args.lag_prob,
        heuristic_mode=args.heuristic_mode,
        optimal_model_path=args.optimal_model_path,
        seed=args.seed,
    )

    # 2. Instantiate shared autonomy environment
    env = SharedAutonomyEnv(
        pilot=pilot,
        copilot=None,  # The RL algorithm will act as the copilot during training
        max_episode_steps=args.max_episode_steps,
        corner_name=args.corner,
        reward_mode=args.reward_mode,
        lambda_intervention=args.lambda_intervention,
        residual_scale=args.residual_scale,
        include_pilot_action_in_obs=True,  # Extended obs: [obs(11), a_h(4)] -> 15 dimensions
        action_smoothness_weight=args.action_smoothness_weight,
        obstacle_safe_distance_m=args.obstacle_safe_distance,
        altitude_weight=args.altitude_weight,
        safe_altitude_m=args.safe_altitude,
        proximity_mode=args.proximity_mode,
        enable_trajectory_logging=False,
        verbose=0,
    )

    # 3. Configure reinforcement learning algorithm
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
        sac_ent_coef = args.ent_coef
        try:
            sac_ent_coef = float(args.ent_coef)
        except (ValueError, TypeError):
            sac_ent_coef = "auto"

        model = SAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=args.lr,
            buffer_size=100_000,
            batch_size=batch_size,
            gamma=0.99,
            tau=0.005,
            ent_coef=sac_ent_coef,
            verbose=args.verbose,
            tensorboard_log=args.tensorboard_log,
            seed=args.seed,
        )
    else:
        raise ValueError(f"Algoritmo no soportado: {algo_name}")

    # 4. Optional evaluation callback
    # Webots permite una única instancia de Robot por proceso, por lo que no se puede crear
    # un entorno de evaluación separado. Se guarda el mejor modelo según la recompensa media
    # móvil de los episodios de entrenamiento (reutilizando el mismo entorno).
    callbacks = []
    if args.eval_freq > 0:
        best_model_dir = os.path.join(save_dir, f"best_copilot_{algo_name}_{args.pilot_type}")
        os.makedirs(best_model_dir, exist_ok=True)
        callbacks.append(
            BestTrainingRewardCallback(
                check_freq=args.eval_freq,
                window=args.eval_episodes,
                save_dir=best_model_dir,
                verbose=args.verbose,
            )
        )
        print(f"[Checkpoint] Mejor modelo (recompensa media móvil) en {best_model_dir}/best_model.zip")

    # 5. Learning loop
    print(f"Comenzando entrenamiento de {algo_name.upper()}...")
    try:
        model.learn(
            total_timesteps=args.timesteps,
            tb_log_name=run_name,
            callback=callbacks if callbacks else None,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nEntrenamiento interrumpido manualmente por el usuario.")

    # 6. Save trained model
    if args.save_path.endswith(".h"):
        print(
            f"[Aviso] Se guardará en formato ZIP de SB3 a pesar de la extensión .h ({args.save_path})."
        )
    model.save(args.save_path)
    print(f"\n¡Entrenamiento finalizado! Modelo de copiloto guardado en: {args.save_path}")

    env.close()


if __name__ == "__main__":
    train()
