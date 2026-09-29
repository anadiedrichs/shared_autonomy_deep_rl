# -*- coding: utf-8 -*-
"""
Script de evaluación y validación de vuelo de HeuristicPilot en CornerEnvContinuous.

Permite ejecutar un episodio individual con el piloto heurístico en modo lazo abierto
(enfoque 1) o lazo cerrado proporcional (enfoque 2), imprimiendo la telemetría en
tiempo real y reportando la cinemática exacta hasta el objetivo.

Uso:
    python src/evaluation/run_heuristic_episode.py --mode proportional --corner cone_1
    python src/evaluation/run_heuristic_episode.py --mode open_loop --corner cone_1
"""

import argparse
import os
import sys
import time

# Asegurar que la raíz del proyecto esté en el PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.envs.corner_env_continuous import CornerEnvContinuous
from src.pilots.heuristic_pilot import HeuristicPilot


def run_heuristic_episode(
    mode: str = "proportional",
    corner_name: str = "cone_1",
    max_episode_steps: int = 800,
    log_interval: int = 25,
    enable_trajectory_logging: bool = False,
    verbose: int = 1,
) -> dict:
    """
    Ejecuta un episodio completo con HeuristicPilot y reporta métricas.

    Args:
        mode: 'proportional' (enfoque 2) o 'open_loop' (enfoque 1).
        corner_name: Identificador del cono objetivo ("cone_1", "cone_2", etc.).
        max_episode_steps: Límite de pasos del episodio.
        log_interval: Frecuencia de pasos para imprimir telemetría en consola.
        enable_trajectory_logging: Si es True, guarda CSV con la trayectoria física.
        verbose: Nivel de detalle de salida del entorno.

    Returns:
        dict: Resumen con métricas de la ejecución (éxito, pasos, recompensa, etc.).
    """
    print("=" * 70)
    print(f"[Heuristic Flight] Iniciando evaluación de HeuristicPilot")
    print(f"  Modo: {mode}")
    print(f"  Objetivo: {corner_name}")
    print(f"  Límite de pasos: {max_episode_steps}")
    print("=" * 70)

    # 1. Instanciar entorno
    env = CornerEnvContinuous(
        corner_name=corner_name,
        max_episode_steps=max_episode_steps,
        enable_trajectory_logging=enable_trajectory_logging,
        verbose=verbose,
    )

    # 2. Instanciar piloto heurístico vinculado al entorno
    pilot = HeuristicPilot(
        mode=mode,
        corner_name=corner_name,
        env=env,
        target_altitude=0.55,
        target_radius=0.15,
    )

    # 3. Reset y despegue
    obs, info = env.reset()
    start_time = time.time()

    initial_dist = float(info.get("dist_min_target", env.get_distance_to_target()))
    print(f"[Start] Posición inicial: [x={env.x_global:.2f}, y={env.y_global:.2f}, alt={env.alt:.2f}m]")
    print(f"[Start] Distancia inicial al objetivo: {initial_dist:.3f} m")
    print("-" * 70)
    print(f"{'Paso':>6} | {'X':>6} {'Y':>6} {'Alt':>6} | {'Dist Meta':>10} | {'Cmd [vx, vy, vz]':>18} | {'Rew':>7} | {'Acum':>8}")
    print("-" * 70)

    total_reward = 0.0
    step_count = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        step_count += 1

        # Obtener acción del piloto heurístico
        action = pilot.get_action(obs, env=env)

        # Aplicar paso en la simulación
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Imprimir telemetría periódica o en pasos terminales
        if step_count % log_interval == 0 or terminated or truncated:
            dist = float(info.get("dist_min_target", env.get_distance_to_target()))
            cmd_str = f"[{action[0]:+.2f}, {action[1]:+.2f}, {action[3]:+.2f}]"
            print(
                f"{step_count:6d} | "
                f"{env.x_global:6.2f} {env.y_global:6.2f} {env.alt:6.2f} | "
                f"{dist:9.3f} m | "
                f"{cmd_str:>18} | "
                f"{reward:+7.2f} | "
                f"{total_reward:+8.2f}"
            )

    elapsed = time.time() - start_time
    is_success = bool(info.get("is_success", env.is_success))
    final_dist = float(info.get("dist_min_target", env.get_distance_to_target()))
    termination_reason = info.get("corner", env.corner)

    print("=" * 70)
    print("[Heuristic Flight] Resumen del Episodio:")
    print(f"  Éxito alcanzado (is_success) : {'SÍ (EXITO)' if is_success else 'NO (FALLO)'}")
    print(f"  Motivo de terminación        : {termination_reason}")
    print(f"  Pasos empleados              : {step_count} / {max_episode_steps}")
    print(f"  Tiempo de simulación         : {step_count * 0.032:.2f} s ({elapsed:.2f} s reloj real)")
    print(f"  Distancia final al cono      : {final_dist:.3f} m (umbral: {env.DISTANCE_THRESHOLD} m)")
    print(f"  Altitud final                : {env.alt:.2f} m (umbral éxito: > 0.50 m)")
    print(f"  Recompensa acumulada         : {total_reward:+.2f}")
    print("=" * 70)

    env.close()

    return {
        "is_success": is_success,
        "step_count": step_count,
        "total_reward": total_reward,
        "final_distance": final_dist,
        "final_altitude": env.alt,
        "termination_reason": termination_reason,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ejecución de un episodio con HeuristicPilot (Enfoque 1 u Enfoque 2)."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="proportional",
        choices=["proportional", "open_loop", "enfoque1", "enfoque2", "fixed", "closed_loop"],
        help="Modo del piloto: 'proportional' (lazo cerrado con frenado) o 'open_loop' (acción constante).",
    )
    parser.add_argument(
        "--corner",
        type=str,
        default="cone_1",
        help="Identificador del cono objetivo en Webots ('cone_1', 'cone_2', etc.).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=800,
        help="Número máximo de pasos del episodio (por defecto 800).",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=25,
        help="Intervalo de pasos para imprimir telemetría en consola.",
    )
    parser.add_argument(
        "--trajectory-log",
        action="store_true",
        help="Habilita guardado de trayectoria en CSV.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Nivel de verbosidad del entorno (0: silencioso, 1: eventos clave).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_heuristic_episode(
        mode=args.mode,
        corner_name=args.corner,
        max_episode_steps=args.steps,
        log_interval=args.log_interval,
        enable_trajectory_logging=args.trajectory_log,
        verbose=args.verbose,
    )
