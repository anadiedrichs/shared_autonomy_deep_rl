#!/usr/bin/env bash
# ==============================================================================
# Script de Evaluación Multisemilla para Pilotos Degradados
# TFM: Autonomía Compartida para UAVs con Webots y RLtools
# Autora: Ana Laura Diedrichs
# ==============================================================================
# Evalúa cuatro perfiles de pilotos humanos/sintéticos degradados sobre un modelo
# base óptimo, a lo largo de 5 semillas independientes:
#   1) NoisyPilot con sigma = 0.5   (ruido motriz moderado)
#   2) NoisyPilot con sigma = 1.0   (ruido motriz al 100% / sigma = 1.0)
#   3) LaggyPilot con p_repeat = 0.8 (retardo humano ZOH / Schaff & Walter 2020)
#   4) StandByPilot con p_drop = 0.8 (pérdida de paquetes 80% con modo seguro hover)
#
# Todos los resultados (CSV detallados, gráficos de trayectoria y tabla resumen)
# se almacenan organizados dentro de results_TFM/logs/degraded/.
#
# Uso básico:
#   bash scripts/evaluate_degraded_pilots.sh
#
# Filtrar perfiles específicos:
#   bash scripts/evaluate_degraded_pilots.sh noisy       # Solo Noisy (0.5 y 1.0)
#   bash scripts/evaluate_degraded_pilots.sh laggy       # Solo Laggy (0.8)
#   bash scripts/evaluate_degraded_pilots.sh standby     # Solo StandBy (0.8)
#
# Variables de entorno opcionales:
#   SEEDS="42 123 256 512 789"                   # 5 semillas maestras de evaluación
#   EPISODES=20                                  # Episodios por semilla (por defecto: 20)
#   CORNER="cone_1"                              # Cono objetivo
#   CHECKPOINT="models/optimal_pilot_sac_seed_123_best.h"    # Checkpoint del piloto óptimo base
#   BACKEND="rltools_sac"                            # Backend (sb3_ppo, rltools_td3, rltools_sac)
#   RESULTS_DIR="results_TFM"                    # Directorio raíz de resultados
# ==============================================================================

set -o pipefail

# Ubicar raíz del repositorio (0001-webots-rl)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Configuración por defecto
SEEDS=(${SEEDS:-42 123 256 512 789})
EPISODES=${EPISODES:-20}
CORNER=${CORNER:-cone_1}
RESULTS_DIR=${RESULTS_DIR:-results_TFM}
OUT_DIR="${RESULTS_DIR}/logs/degraded"

# Detección automática del checkpoint base si no se definió explícitamente
if [ -z "${CHECKPOINT}" ]; then
    if [ -f "${RESULTS_DIR}/models/optimal_pilot_td3_checkpoint.h" ]; then
        CHECKPOINT="${RESULTS_DIR}/models/optimal_pilot_td3_checkpoint.h"
        BACKEND="rltools_td3"
    elif [ -f "${RESULTS_DIR}/models/optimal_pilot_sac_checkpoint.h" ]; then
        CHECKPOINT="${RESULTS_DIR}/models/optimal_pilot_sac_checkpoint.h"
        BACKEND="rltools_sac"
    elif [ -f "models/ppo_optimal_pilot.zip" ]; then
        CHECKPOINT="models/ppo_optimal_pilot.zip"
        BACKEND="sb3_ppo"
    elif [ -f "${RESULTS_DIR}/models/ppo_optimal_pilot.zip" ]; then
        CHECKPOINT="${RESULTS_DIR}/models/ppo_optimal_pilot.zip"
        BACKEND="sb3_ppo"
    elif [ -f "models/optimal_pilot_sac_checkpoint.h" ]; then
        CHECKPOINT="models/optimal_pilot_sac_checkpoint.h"
        BACKEND="rltools_sac"
    else
        # Si no hay checkpoint de RL pero solo se va a evaluar heurístico:
        if [ "$1" == "heuristic" ]; then
            CHECKPOINT="none"
            BACKEND="sb3_ppo"
        else
            echo "ERROR: No se encontró ningún checkpoint de piloto óptimo."
            echo "Por favor define la variable CHECKPOINT, ej:"
            echo "  CHECKPOINT=models/ppo_optimal_pilot.zip bash scripts/evaluate_degraded_pilots.sh"
            exit 1
        fi
    fi
fi

# Detectar backend según extensión si no se fijó
if [ -z "${BACKEND}" ]; then
    if [[ "${CHECKPOINT}" == *.zip ]]; then
        BACKEND="sb3_ppo"
    elif [[ "${CHECKPOINT}" == *td3* ]]; then
        BACKEND="rltools_td3"
    elif [[ "${CHECKPOINT}" == *sac* ]]; then
        BACKEND="rltools_sac"
    else
        BACKEND="sb3_ppo"
    fi
fi

# Definir las configuraciones a evaluar:
# Formato: TAG|PILOT_TYPE|PARAM_FLAG|PARAM_VALUE|DESCRIPCION
ALL_CONFIGS=(
    "heuristic_prop|heuristic|--heuristic-mode|proportional|HeuristicPilot (Control P / Referencia Geometrica)"
    "noisy_sigma_0.5|noisy|--sigma|0.5|NoisyPilot (sigma = 0.50)"
    "noisy_sigma_1.0|noisy|--sigma|1.0|NoisyPilot (sigma = 1.00 / 100%)"
    "laggy_p_0.8|laggy|--p-repeat|0.8|LaggyPilot (p_repeat = 0.80 / Schaff & Walter)"
    "standby_p_0.8|standby|--p-repeat|0.8|StandByPilot (p_drop = 0.80 / Hover NOP)"
)

# Filtrar si el usuario pasó argumentos de línea de comandos o mostrar ayuda
ACTIVE_CONFIGS=()
if [ $# -gt 0 ]; then
    if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
        echo "Uso: bash scripts/evaluate_degraded_pilots.sh [FILTRO]"
        echo ""
        echo "Filtros disponibles:"
        echo "  (sin argumentos) : Evalúa todas las configuraciones (Heurístico + 4 Degradados)"
        echo "  heuristic        : Evalúa solo HeuristicPilot (Control Proporcional P)"
        echo "  noisy            : Evalúa solo NoisyPilot (sigma = 0.5 y 1.0)"
        echo "  laggy            : Evalúa solo LaggyPilot (p_repeat = 0.8 / Schaff & Walter)"
        echo "  standby          : Evalúa solo StandByPilot (p_drop = 0.8 / Hover NOP)"
        echo ""
        echo "Variables de entorno:"
        echo "  SEEDS=\"42 123 256 512 789\"       5 semillas maestras"
        echo "  EPISODES=20                      Episodios por semilla"
        echo "  CHECKPOINT=ruta/modelo.zip       Checkpoint del piloto óptimo base"
        echo "  BACKEND=sb3_ppo|rltools_td3      Backend de inferencia"
        echo "  RESULTS_DIR=results_TFM          Carpeta destino"
        exit 0
    fi

    FILTER="$1"
    for cfg in "${ALL_CONFIGS[@]}"; do
        if [[ "${cfg}" == *"${FILTER}"* ]]; then
            ACTIVE_CONFIGS+=("${cfg}")
        fi
    done
    if [ ${#ACTIVE_CONFIGS[@]} -eq 0 ]; then
        echo "ERROR: Ninguna configuración coincide con el filtro: '${FILTER}'"
        echo "Opciones válidas: heuristic, noisy, laggy, standby"
        exit 1
    fi
else
    ACTIVE_CONFIGS=("${ALL_CONFIGS[@]}")
fi

mkdir -p "${OUT_DIR}"

echo "=============================================================================="
echo "      CAMPAÑA DE EVALUACIÓN MULTISEMILLA PARA PILOTOS DEGRADADOS"
echo "=============================================================================="
echo " Raíz del proyecto   : ${PROJECT_ROOT}"
echo " Checkpoint base     : ${CHECKPOINT}"
echo " Backend base        : ${BACKEND}"
echo " Cono objetivo       : ${CORNER}"
echo " Semillas fijadas    : ${SEEDS[*]} (Total: ${#SEEDS[@]} semillas)"
echo " Episodios por run   : ${EPISODES}"
echo " Salida de datos     : ${OUT_DIR}/"
echo " Configuraciones (${#ACTIVE_CONFIGS[@]}) :"
for cfg in "${ACTIVE_CONFIGS[@]}"; do
    IFS='|' read -r TAG PTYPE PFLAG PVAL PDESC <<< "${cfg}"
    echo "   - [${TAG}] ${PDESC}"
done
echo "=============================================================================="
echo ""

START_TOTAL_TIME=$(date +%s)
TOTAL_RUNS=$(( ${#ACTIVE_CONFIGS[@]} * ${#SEEDS[@]} ))
CURRENT_RUN=0

for cfg in "${ACTIVE_CONFIGS[@]}"; do
    IFS='|' read -r TAG PTYPE PFLAG PVAL PDESC <<< "${cfg}"

    echo "##############################################################################"
    echo "  EVALUANDO CONFIGURACIÓN: ${TAG} (${PDESC})"
    echo "##############################################################################"

    for SEED in "${SEEDS[@]}"; do
        CURRENT_RUN=$((CURRENT_RUN + 1))
        RUN_NAME="${TAG}_seed_${SEED}"
        CSV_FILE="${OUT_DIR}/${RUN_NAME}.csv"
        TRAJ_PNG="${OUT_DIR}/${RUN_NAME}_traj.png"
        LOG_FILE="${OUT_DIR}/${RUN_NAME}.log"

        echo "[${CURRENT_RUN}/${TOTAL_RUNS}] Ejecutando: ${RUN_NAME} (${EPISODES} episodios, seed=${SEED})..."
        RUN_START=$(date +%s)

        CHECKPOINT_ARGS=()
        if [ "${PTYPE}" != "heuristic" ] && [ "${CHECKPOINT}" != "none" ]; then
            CHECKPOINT_ARGS+=(--checkpoint "${CHECKPOINT}" --backend "${BACKEND}")
        fi

        python3 src/evaluation/metrics.py \
            --pilot-type "${PTYPE}" \
            "${CHECKPOINT_ARGS[@]}" \
            "${PFLAG}" "${PVAL}" \
            --corner "${CORNER}" \
            --episodes "${EPISODES}" \
            --seed "${SEED}" \
            --save-csv "${CSV_FILE}" \
            --plot-trajectory-png "${TRAJ_PNG}" \
            --verbose 1 > "${LOG_FILE}" 2>&1

        EXIT_CODE=$?
        RUN_END=$(date +%s)
        RUN_DURATION=$((RUN_END - RUN_START))

        if [ ${EXIT_CODE} -eq 0 ]; then
            echo "   -> [OK] Completado en ${RUN_DURATION}s. CSV: ${CSV_FILE}"
        else
            echo "   -> [ERROR code ${EXIT_CODE}] Falló ${RUN_NAME}. Ver detalles en: ${LOG_FILE}"
        fi
    done
    echo ""
done

END_TOTAL_TIME=$(date +%s)
TOTAL_DURATION=$((END_TOTAL_TIME - START_TOTAL_TIME))

echo "=============================================================================="
echo "  GENERANDO TABLA RESUMEN CONSOLIDADA MULTISEMILLA"
echo "=============================================================================="

# Script auxiliar de consolidación en Python (sin dependencias externas)
python3 - << 'EOF'
import os
import glob
import csv
import math

out_dir = os.environ.get("OUT_DIR", "results_TFM/logs/degraded")
csv_files = sorted(glob.glob(os.path.join(out_dir, "*_seed_*.csv")))

if not csv_files:
    print(f"[Resumen] No se encontraron archivos CSV en: {out_dir}")
    exit(0)

records = []
for f in csv_files:
    base = os.path.basename(f)
    parts = base.replace(".csv", "").split("_seed_")
    if len(parts) != 2:
        continue
    config_name, seed_str = parts[0], parts[1]
    
    rows = []
    try:
        with open(f, 'r', encoding='utf-8') as fp:
            reader = csv.DictReader(fp)
            for r in reader:
                rows.append(r)
    except Exception:
        continue
    
    n_eps = len(rows)
    if n_eps == 0:
        continue

    success_rate = sum(1 for r in rows if r.get("success", "").strip().lower() in ("true", "1")) / n_eps * 100.0
    collision_rate = sum(1 for r in rows if r.get("termination") in ("collision", "fall", "out_of_bounds")) / n_eps * 100.0
    timeout_rate = sum(1 for r in rows if r.get("termination") == "timeout") / n_eps * 100.0
    survival_rate = 100.0 - collision_rate

    rewards = [float(r["reward"]) for r in rows if r.get("reward")]
    mean_reward = sum(rewards)/len(rewards) if rewards else 0.0
    std_reward = math.sqrt(sum((x - mean_reward)**2 for x in rewards)/(len(rewards)-1)) if len(rewards) > 1 else 0.0

    dists = [float(r["final_distance"]) for r in rows if r.get("final_distance")]
    mean_dist = sum(dists)/len(dists) if dists else 0.0
    std_dist = math.sqrt(sum((x - mean_dist)**2 for x in dists)/(len(dists)-1)) if len(dists) > 1 else 0.0

    obs_vals = []
    for r in rows:
        val = r.get("min_obstacle_distance") or r.get("final_obstacle_distance")
        if val is not None and val != "" and val.lower() != "nan":
            try:
                obs_vals.append(float(val))
            except ValueError:
                pass
    mean_obs = sum(obs_vals)/len(obs_vals) if obs_vals else None

    alt_vals = []
    for r in rows:
        val = r.get("final_altitude_m") or r.get("final_altitude_mm")
        if val is not None and val != "" and val.lower() != "nan":
            try:
                fval = float(val)
                if fval > 10.0:
                    fval /= 1000.0
                alt_vals.append(fval)
            except ValueError:
                pass
    mean_alt = sum(alt_vals)/len(alt_vals) if alt_vals else None
    
    records.append({
        "config": config_name,
        "seed": int(seed_str),
        "episodes": n_eps,
        "success_pct": success_rate,
        "collision_pct": collision_rate,
        "timeout_pct": timeout_rate,
        "survival_pct": survival_rate,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "mean_dist_m": mean_dist,
        "std_dist_m": std_dist,
        "mean_obs_m": mean_obs,
        "mean_alt_m": mean_alt,
    })

if not records:
    print("[Resumen] No se pudieron parsear registros válidos.")
    exit(0)

summary_csv = os.path.join(out_dir, "summary_degraded_pilots.csv")
keys = list(records[0].keys())
try:
    with open(summary_csv, 'w', newline='', encoding='utf-8') as fp:
        writer = csv.DictWriter(fp, fieldnames=keys)
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)
    print(f"\n[Resumen] Tabla detallada guardada en: {summary_csv}\n")
except OSError:
    print(f"\n[Aviso] No se pudo escribir en {summary_csv} (modo solo lectura). Mostrando resumen en pantalla:\n")

header = f"{'Configuración':<22} | {'Seed':<5} | {'Éxito %':<8} | {'Superv %':<8} | {'Dist Meta (m)':<14} | {'Dist Obs (m)':<12} | {'Alt (m)':<9} | {'Recompensa':<14}"
print(header)
print("-" * len(header))
for r in records:
    obs_str = f"{r['mean_obs_m']:>6.3f}m" if r['mean_obs_m'] is not None else "    N/D     "
    alt_str = f"{r['mean_alt_m']:>6.3f}m" if r['mean_alt_m'] is not None else "   N/D   "
    print(f"{r['config']:<22} | {r['seed']:<5} | {r['success_pct']:>7.1f}% | {r['survival_pct']:>7.1f}% | {r['mean_dist_m']:>6.3f} ± {r['std_dist_m']:>4.2f}m | {obs_str:<12} | {alt_str:<9} | {r['mean_reward']:>6.1f} ± {r['std_reward']:>4.1f}")

# Agrupación por configuración (media sobre las semillas)
print("\n" + "=" * 95)
print("  PROMEDIOS GLOBALES SOBRE LAS SEMILLAS EVALUADAS")
print("=" * 95)
configs_set = sorted(list(set(r['config'] for r in records)))
agg_header = f"{'Configuración':<22} | {'Éxito Promedio':<15} | {'Supervivencia':<15} | {'Dist Meta (m)':<15} | {'Recompensa':<14}"
print(agg_header)
print("-" * len(agg_header))
for c in configs_set:
    sub = [r for r in records if r['config'] == c]
    succ_avg = sum(r['success_pct'] for r in sub) / len(sub)
    surv_avg = sum(r['survival_pct'] for r in sub) / len(sub)
    dist_avg = sum(r['mean_dist_m'] for r in sub) / len(sub)
    rew_avg = sum(r['mean_reward'] for r in sub) / len(sub)
    print(f"{c:<22} | {succ_avg:>13.1f}%  | {surv_avg:>13.1f}%  | {dist_avg:>13.3f}m  | {rew_avg:>12.1f}")
EOF

echo ""
echo "=============================================================================="
echo "  EVALUACIÓN MULTISEMILLA FINALIZADA"
echo "=============================================================================="
echo " Tiempo total transcurrido : ${TOTAL_DURATION} segundos"
echo " Archivos generados en     : ${OUT_DIR}/"
echo "=============================================================================="
