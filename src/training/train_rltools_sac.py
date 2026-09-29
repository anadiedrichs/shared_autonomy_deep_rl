# -*- coding: utf-8 -*-
"""
Entrenamiento del piloto óptimo continuo con SAC (RLtools).

Entrena una política continua con Soft Actor-Critic (SAC) usando la interfaz Python
de RLtools sobre CornerEnvContinuous, y exporta la política entrenada directamente como
un header de C/C++ (.h) para despliegue embebido en el Crazyflie.
"""

import argparse
import os
import sys

# Asegurar que la raíz del proyecto esté en el PYTHONPATH
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
    Crea la factoría del entorno requerida por la interfaz de RLtools.
    """
    def env_factory():
        env = CornerEnvContinuous(
            corner_name=corner_name,
            max_episode_steps=max_episode_steps,
            enable_trajectory_logging=False,
            verbose=0,
        )
        if gym is not None and hasattr(gym, "wrappers"):
            env = gym.wrappers.RescaleAction(env, min_action=-1.0, max_action=1.0)
        return env

    return env_factory


def train_rltools_sac(
    max_steps: int = 100_000,
    seed: int = 7,
    corner_name: str = "cone_1",
    max_episode_steps: int = 1_000,
    header_save_path: str = "./models/optimal_pilot_sac_checkpoint.h",
    log_dir: str = "./logs/sac",
    verbose: bool = True,
):
    """
    Ejecuta el ciclo de entrenamiento de SAC con RLtools.

    Args:
        max_steps: Número máximo de pasos de entrenamiento a ejecutar.
        seed: Semilla para reproducibilidad numérica.
        corner_name: Identificador DEF del cono objetivo en Webots.
        max_episode_steps: Límite máximo de pasos por episodio.
        header_save_path: Ruta del archivo header C (.h) para exportar la política entrenada.
        log_dir: Directorio para guardar logs y métricas de TensorBoard.
        verbose: Si es True, imprime progreso periódico en consola.

    Returns:
        tuple: (estado final del entrenamiento, código fuente C del header exportado).
    """
    try:
        from rltools import SAC
    except ImportError as e:
        raise ImportError(
            f"No se pudo importar rltools: {e}. "
            "Asegúrate de instalarlo con 'pip install rltools' (requiere C++ compiler y pybind11)."
        )

    os.makedirs(os.path.dirname(os.path.abspath(header_save_path)), exist_ok=True)

    # Configuración de TensorBoard y registro de hiperparámetros
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

    print(f"[RLtools-SAC] Inicializando algoritmo SAC con semilla {seed}...")
    sac = SAC(env_factory)
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
    """Analizador de argumentos de línea de comandos."""
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
        corner_name=args.corner,
        header_save_path=args.export_path,
        log_dir=args.log_dir,
    )
