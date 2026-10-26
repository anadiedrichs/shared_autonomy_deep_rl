# -*- coding: utf-8 -*-
"""
Continuous optimal pilot training with SAC (RLtools).

Trains a continuous policy with Soft Actor-Critic (SAC) using the RLtools Python
interface on CornerEnvContinuous, and exports the trained policy directly as
a C/C++ header (.h) for embedded deployment on the Crazyflie.
"""

import argparse
import csv
import os
import re
import select
import sys
import time
import numpy as np

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


class RLtoolsEpisodeMetricsWrapper(gym.Wrapper if gym is not None and hasattr(gym, "Wrapper") else object):
    """
    Episode-level metrics wrapper for RLtools training.
    Captures episode rewards, lengths, and success flags, writing them to:
    1. monitor.csv (SB3-compatible format: r, l, t)
    2. TensorBoard scalars (rollout/ep_rew_mean, rollout/ep_len_mean, rollout/success_rate)
    """

    def __init__(self, env, log_dir: str | None = None, writer=None, verbose: bool = True):
        if gym is not None and hasattr(gym, "Wrapper"):
            super().__init__(env)
        self.env = env
        self.log_dir = log_dir
        self.writer = writer
        self.verbose = verbose

        self.episode_count = 0
        self.total_env_steps = 0
        self.episode_reward = 0.0
        self.episode_length = 0
        self.start_time = time.time()

        self.recent_rewards: list[float] = []
        self.recent_successes: list[float] = []
        self.recent_lengths: list[int] = []

        # Ensure spaces are directly accessible for the pybind11 RLtools layer
        self.observation_space = getattr(env, "observation_space", None)
        self.action_space = getattr(env, "action_space", None)

        # Setup CSV writer for monitor.csv (SB3 format)
        self.csv_file = None
        self.csv_writer = None
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)
            csv_path = os.path.join(self.log_dir, "monitor.csv")
            self.csv_file = open(csv_path, "w", newline="", encoding="utf-8")
            self.csv_file.write(f'#{{"t_start": {self.start_time}, "env_id": "CornerEnvContinuous"}}\n')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(["r", "l", "t"])
            self.csv_file.flush()

    def reset(self, **kwargs):
        self.episode_reward = 0.0
        self.episode_length = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.total_env_steps += 1
        self.episode_reward += float(reward)
        self.episode_length += 1

        if terminated or truncated:
            self.episode_count += 1
            is_success = bool(info.get("is_success", False))
            corner_flag = info.get("corner", "unknown")
            if is_success:
                term_reason = "goal"
            elif corner_flag == "fall":
                term_reason = "fall"
            elif corner_flag == "out_of_bounds":
                term_reason = "out_of_bounds"
            elif truncated:
                term_reason = "timeout"
            else:
                term_reason = str(corner_flag)

            elapsed_time = time.time() - self.start_time
            if self.csv_writer is not None:
                self.csv_writer.writerow(
                    [round(self.episode_reward, 6), self.episode_length, round(elapsed_time, 4)]
                )
                self.csv_file.flush()

            self.recent_rewards.append(self.episode_reward)
            self.recent_successes.append(1.0 if is_success else 0.0)
            self.recent_lengths.append(self.episode_length)
            if len(self.recent_rewards) > 20:
                self.recent_rewards.pop(0)
                self.recent_successes.pop(0)
                self.recent_lengths.pop(0)

            if self.writer is not None:
                self.writer.add_scalar("rollout/ep_rew", self.episode_reward, self.total_env_steps)
                self.writer.add_scalar("rollout/ep_len", self.episode_length, self.total_env_steps)
                self.writer.add_scalar("rollout/is_success", 1.0 if is_success else 0.0, self.total_env_steps)
                self.writer.add_scalar("rollout/ep_rew_mean", float(np.mean(self.recent_rewards)), self.total_env_steps)
                self.writer.add_scalar("rollout/ep_len_mean", float(np.mean(self.recent_lengths)), self.total_env_steps)
                self.writer.add_scalar("rollout/success_rate", float(np.mean(self.recent_successes)), self.total_env_steps)

            if self.verbose:
                status_icon = "✔ ÉXITO" if is_success else f"✘ {term_reason}"
                print(
                    f"[RLtools-SAC] Ep {self.episode_count:3d} ({status_icon}) | "
                    f"Paso: {self.total_env_steps:6d} | "
                    f"Rew: {self.episode_reward:7.2f} | "
                    f"Long: {self.episode_length:4d} | "
                    f"Dist: {getattr(self.env, 'dist_min_target', 0.0):.2f}m"
                )

        return obs, reward, terminated, truncated, info

    def close(self):
        if self.csv_file is not None and not self.csv_file.closed:
            self.csv_file.close()
            self.csv_file = None
        if hasattr(self.env, "close"):
            return self.env.close()

    def __getattr__(self, name):
        return getattr(self.env, name)


def make_env_factory(
    corner_name: str = "cone_1",
    max_episode_steps: int = 1_000,
    log_dir: str | None = None,
    writer=None,
    verbose: bool = True,
):
    """
    Create the environment factory required by the RLtools interface.
    Uses the singleton/persistent pattern to avoid multiple Webots Supervisor
    initializations within the same Python process.
    """
    _env_instance = None

    def env_factory():
        nonlocal _env_instance
        if _env_instance is None:
            raw_env = CornerEnvContinuous(
                corner_name=corner_name,
                max_episode_steps=max_episode_steps,
                enable_trajectory_logging=False,
                verbose=0,
            )
            rescaled_env = raw_env
            if gym is not None and hasattr(gym, "wrappers"):
                rescaled_env = gym.wrappers.RescaleAction(
                    raw_env, min_action=np.float32(-1.0), max_action=np.float32(1.0)
                )
            _env_instance = RLtoolsEpisodeMetricsWrapper(
                rescaled_env,
                log_dir=log_dir,
                writer=writer,
                verbose=verbose,
            )
        return _env_instance

    return env_factory


class CStdoutTee:
    """
    Captures C++ stdout output (e.g. from RLtools log statements) while
    simultaneously forwarding it to the real stdout so console logs remain intact.
    Extracts periodic evaluation returns: 'Mean return: <float>'.
    """

    def __init__(self):
        self.enabled = False
        try:
            self.orig_stdout_fd = os.dup(1)
            self.pipe_r, self.pipe_w = os.pipe()
            os.dup2(self.pipe_w, 1)
            self.buffer = ""
            self.enabled = True
        except Exception:
            self.enabled = False

    def read_eval_returns(self) -> list[float]:
        if not self.enabled:
            return []
        try:
            import ctypes
            libc = ctypes.CDLL(None)
            libc.fflush(None)
        except Exception:
            pass
        sys.stdout.flush()
        returns = []
        try:
            while True:
                r, _, _ = select.select([self.pipe_r], [], [], 0.0)
                if not r:
                    break
                chunk = os.read(self.pipe_r, 4096).decode("utf-8", errors="ignore")
                if not chunk:
                    break
                os.write(self.orig_stdout_fd, chunk.encode("utf-8"))
                self.buffer += chunk

            matches = re.findall(
                r"Mean return:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", self.buffer
            )
            for m in matches:
                try:
                    returns.append(float(m))
                except ValueError:
                    pass
            self.buffer = ""
        except Exception:
            pass
        return returns

    def close(self):
        if not self.enabled:
            return
        try:
            sys.stdout.flush()
            os.dup2(self.orig_stdout_fd, 1)
            os.close(self.orig_stdout_fd)
            os.close(self.pipe_w)
            os.close(self.pipe_r)
        except Exception:
            pass
        finally:
            self.enabled = False


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

    print(f"[RLtools-SAC] Creando fábrica de entorno con métricas (esquina: {corner_name})...")
    metrics_log_dir = run_dir if run_dir is not None else log_dir
    env_factory = make_env_factory(
        corner_name=corner_name,
        max_episode_steps=max_episode_steps,
        log_dir=metrics_log_dir,
        writer=writer,
        verbose=verbose,
    )

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

    best_eval_return = float("-inf")
    best_step = None
    best_policy_c_header = None
    best_header_path = header_save_path.replace(".h", "_best.h")
    final_header_path = header_save_path.replace(".h", "_final.h")

    stdout_tee = CStdoutTee()
    print(f"[RLtools-SAC] Iniciando bucle de entrenamiento (hasta {max_steps} pasos)...")
    step_count = 0
    finished = False

    try:
        while not finished and step_count < max_steps:
            finished = state.step()
            step_count += 1

            # Check for periodic evaluation outputs from RLtools C++ engine
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
                            f"\n[RLtools-SAC] ★ ¡Nuevo mejor modelo guardado en paso {step_count}! "
                            f"Retorno medio de evaluación: {ret:.4f} (guardado en {header_save_path})\n"
                        )
                    if writer is not None:
                        writer.add_scalar("evaluation/best_mean_return", best_eval_return, step_count)

            if step_count % 10_000 == 0 or finished:
                if verbose:
                    print(f"[RLtools-SAC] Progreso: paso {step_count}/{max_steps} completado.")
                if writer is not None:
                    writer.add_scalar("train/step", step_count, step_count)
    finally:
        stdout_tee.close()

    print(f"[RLtools-SAC] Entrenamiento finalizado. Exportando política del paso final a {final_header_path}...")
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
            f"[RLtools-SAC] Resumen de checkpoints:\n"
            f"  - Checkpoint óptimo activo: {header_save_path} (paso {best_step}, retorno {best_eval_return:.4f})\n"
            f"  - Checkpoint mejor guardado: {best_header_path}\n"
            f"  - Checkpoint paso final:     {final_header_path}"
        )
    return state, best_policy_c_header


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
