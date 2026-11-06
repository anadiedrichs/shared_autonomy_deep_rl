#!/usr/bin/env bash
# ==============================================================================
# Script de Evaluación de Pilotos Óptimos Pre-entrenados (Sin Re-entrenamiento)
# TFM: Autonomía Compartida para UAVs con Webots y RLtools
# Autora: Ana Laura Diedrichs
# ==============================================================================
# Evalúa los modelos ya entrenados de pilotos óptimos (TD3, SAC y PPO) a lo largo
# de las 5 semillas maestras (42, 123, 256, 512, 789).
#
# Genera las métricas actualizadas de la tesis:
#   - Tasa de éxito (%)
#   - Tasa de colisión / caídas (%)
#   - Tasa de tiempo agotado (timeout %)
#   - Supervivencia / Integridad estructural (%)
#   - Recompensa acumulada media (mu +- sigma)
#   - Longitud media de episodio y tiempo de vuelo (s)
#   - Distancia final a la meta (m)
#   - Distancia al obstáculo más cercano (m)  [NUEVA MÉTRICA ToF Multi-ranger]
#   - Altitud final media (m)                [ESTANDARIZADA EN METROS]
#
# Los resultados se almacenan en results_TFM/logs/{algo}/ y figures en results_TFM/figures/
#
# Uso básico (evalúa TD3, SAC y PPO):
#   bash scripts/evaluate_optimal_pilots.sh
#
# Evaluar un solo algoritmo:
#   bash scripts/evaluate_optimal_pilots.sh td3
#   bash scripts/evaluate_optimal_pilots.sh sac
#   bash scripts/evaluate_optimal_pilots.sh ppo
#
# Variables de entorno opcionales:
#   SEEDS="42 123 256 512 789"    # Semillas de modelos a evaluar
#   EPISODES=100                   # Episodios por evaluación (por defecto: 100)
#   EVAL_SEED=42                   # Semilla maestra del generador de evaluación
#   CORNER="cone_1"                # Cono meta
#   RESULTS_DIR="results_TFM"      # Directorio raíz de resultados
#   SKIP_EXISTING=0                # 1: salta si ya existe el CSV, 0: re-evalúa (defecto: 0)
# ==============================================================================

set -o pipefail

# Ubicar raíz del repositorio (0001-webots-rl)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Configuración por defecto
SEEDS=(${SEEDS:-42 123 256 512 789})
EPISODES=${EPISODES:-100}
EVAL_SEED=${EVAL_SEED:-42}
CORNER=${CORNER:-cone_1}
RESULTS_DIR=${RESULTS_DIR:-results_TFM}
SKIP_EXISTING=${SKIP_EXISTING:-0}

MODELS_DIR="${RESULTS_DIR}/models"
LOGS_DIR="${RESULTS_DIR}/logs"
FIGURES_DIR="${RESULTS_DIR}/figures"
TABLES_DIR="${RESULTS_DIR}/tables"

mkdir -p "${FIGURES_DIR}" "${TABLES_DIR}"

# Selección de algoritmos
TARGET_ALGO="${1:-all}"
TARGET_ALGO="$(echo "${TARGET_ALGO}" | tr '[:upper:]' '[:lower:]')"

if [ "${TARGET_ALGO}" == "-h" ] || [ "${TARGET_ALGO}" == "--help" ]; then
    echo "Uso: bash scripts/evaluate_optimal_pilots.sh [td3|sac|ppo|all]"
    echo ""
    echo "Argumentos:"
    echo "  all (defecto) : Evalúa todos los modelos disponibles (TD3, SAC y PPO)"
    echo "  td3           : Evalúa únicamente los modelos RLtools TD3"
    echo "  sac           : Evalúa únicamente los modelos RLtools SAC"
    echo "  ppo           : Evalúa únicamente los modelos Stable-Baselines3 PPO"
    echo ""
    echo "Variables de entorno:"
    echo "  SEEDS=\"42 123 256 512 789\"    Semillas a evaluar"
    echo "  EPISODES=100                  Episodios por semilla (defecto: 100)"
    echo "  EVAL_SEED=42                  Semilla maestra de simulación"
    echo "  CORNER=cone_1                 Cono objetivo"
    echo "  RESULTS_DIR=results_TFM       Directorio de resultados"
    echo "  SKIP_EXISTING=1               Omitir si el archivo CSV ya existe"
    exit 0
fi

ALGOS=()
case "${TARGET_ALGO}" in
    td3)  ALGOS=("td3") ;;
    sac)  ALGOS=("sac") ;;
    ppo)  ALGOS=("ppo") ;;
    all)  ALGOS=("td3" "sac" "ppo") ;;
    *)
        echo "ERROR: Algoritmo desconocido '${TARGET_ALGO}'. Opciones válidas: td3, sac, ppo, all"
        exit 1
        ;;
esac

echo "=============================================================================="
echo "    EVALUACIÓN DE PILOTOS ÓPTIMOS PRE-ENTRENADOS (SIN RE-ENTRENAMIENTO)"
echo "=============================================================================="
echo " Raíz del proyecto      : ${PROJECT_ROOT}"
echo " Directorio modelos     : ${MODELS_DIR}"
echo " Algoritmos a evaluar   : ${ALGOS[*]}"
echo " Semillas a evaluar     : ${SEEDS[*]} (Total: ${#SEEDS[@]} semillas)"
echo " Episodios por corrida  : ${EPISODES}"
echo " Semilla de evaluación  : ${EVAL_SEED}"
echo " Cono objetivo          : ${CORNER}"
echo " Salida de logs CSV     : ${LOGS_DIR}/"
echo " Salida de figuras      : ${FIGURES_DIR}/"
echo " Omitir existentes      : ${SKIP_EXISTING}"
echo "=============================================================================="
echo ""

START_TOTAL_TIME=$(date +%s)
TOTAL_RUNS=$(( ${#ALGOS[@]} * ${#SEEDS[@]} ))
CURRENT_RUN=0
EVALUATED_CSVS=()

for ALGO in "${ALGOS[@]}"; do
    ALGO_LOG_DIR="${LOGS_DIR}/${ALGO}"
    mkdir -p "${ALGO_LOG_DIR}"

    echo "##############################################################################"
    echo "  EVALUANDO ALGORITMO: ${ALGO^^}"
    echo "##############################################################################"

    for SEED in "${SEEDS[@]}"; do
        CURRENT_RUN=$((CURRENT_RUN + 1))
        RUN_NAME="eval_${ALGO}_seed_${SEED}_${EPISODES}ep"
        CSV_FILE="${ALGO_LOG_DIR}/${RUN_NAME}.csv"
        PNG_FILE="${FIGURES_DIR}/trayectoria_${ALGO}_seed_${SEED}_${EPISODES}ep.png"
        PNG_LEGACY="${ALGO_LOG_DIR}/trayectoria_${ALGO}_seed_${SEED}_${EPISODES}ep.png"
        LOG_FILE="${ALGO_LOG_DIR}/${RUN_NAME}.log"

        # Detectar checkpoint del modelo
        MODEL_FILE=""
        BACKEND=""
        case "${ALGO}" in
            td3)
                BACKEND="rltools_td3"
                if [ -f "${MODELS_DIR}/optimal_pilot_td3_seed_${SEED}_best.h" ]; then
                    MODEL_FILE="${MODELS_DIR}/optimal_pilot_td3_seed_${SEED}_best.h"
                elif [ -f "${MODELS_DIR}/optimal_pilot_td3_seed_${SEED}.h" ]; then
                    MODEL_FILE="${MODELS_DIR}/optimal_pilot_td3_seed_${SEED}.h"
                elif [ -f "${MODELS_DIR}/optimal_pilot_td3_checkpoint.h" ]; then
                    MODEL_FILE="${MODELS_DIR}/optimal_pilot_td3_checkpoint.h"
                elif [ -f "models/optimal_pilot_td3_checkpoint.h" ]; then
                    MODEL_FILE="models/optimal_pilot_td3_checkpoint.h"
                fi
                ;;
            sac)
                BACKEND="rltools_sac"
                if [ -f "${MODELS_DIR}/optimal_pilot_sac_seed_${SEED}_best.h" ]; then
                    MODEL_FILE="${MODELS_DIR}/optimal_pilot_sac_seed_${SEED}_best.h"
                elif [ -f "${MODELS_DIR}/optimal_pilot_sac_seed_${SEED}.h" ]; then
                    MODEL_FILE="${MODELS_DIR}/optimal_pilot_sac_seed_${SEED}.h"
                elif [ -f "${MODELS_DIR}/optimal_pilot_sac_checkpoint.h" ]; then
                    MODEL_FILE="${MODELS_DIR}/optimal_pilot_sac_checkpoint.h"
                elif [ -f "models/optimal_pilot_sac_checkpoint.h" ]; then
                    MODEL_FILE="models/optimal_pilot_sac_checkpoint.h"
                fi
                ;;
            ppo)
                BACKEND="sb3_ppo"
                if [ -f "${MODELS_DIR}/ppo_optimal_pilot_seed_${SEED}.zip" ]; then
                    MODEL_FILE="${MODELS_DIR}/ppo_optimal_pilot_seed_${SEED}.zip"
                elif [ -f "${MODELS_DIR}/ppo_optimal_pilot.zip" ]; then
                    MODEL_FILE="${MODELS_DIR}/ppo_optimal_pilot.zip"
                elif [ -f "models/ppo_optimal_pilot.zip" ]; then
                    MODEL_FILE="models/ppo_optimal_pilot.zip"
                fi
                ;;
        esac

        if [ -z "${MODEL_FILE}" ] || [ ! -f "${MODEL_FILE}" ]; then
            echo "[${CURRENT_RUN}/${TOTAL_RUNS}] [ALERTA] Modelo no encontrado para ${ALGO^^} (seed ${SEED}). Omitiendo..."
            continue
        fi

        if [ "${SKIP_EXISTING}" -eq 1 ] && [ -f "${CSV_FILE}" ]; then
            echo "[${CURRENT_RUN}/${TOTAL_RUNS}] [OMITIDO] CSV ya existe: ${CSV_FILE}"
            EVALUATED_CSVS+=("${CSV_FILE}")
            continue
        fi

        echo "[${CURRENT_RUN}/${TOTAL_RUNS}] Evaluando: ${ALGO^^} seed=${SEED} (${EPISODES} eps)..."
        echo "   Modelo: ${MODEL_FILE}"
        RUN_START=$(date +%s)

        python3 src/evaluation/metrics.py \
            --pilot-type optimal \
            --backend "${BACKEND}" \
            --checkpoint "${MODEL_FILE}" \
            --episodes "${EPISODES}" \
            --seed "${SEED}" \
            --corner "${CORNER}" \
            --save-csv "${CSV_FILE}" \
            --plot-trajectory-png "${PNG_FILE}" \
            --verbose 1 > "${LOG_FILE}" 2>&1

        EXIT_CODE=$?
        RUN_END=$(date +%s)
        RUN_DURATION=$((RUN_END - RUN_START))

        if [ ${EXIT_CODE} -eq 0 ]; then
            cp -f "${PNG_FILE}" "${PNG_LEGACY}" 2>/dev/null || true
            echo "   -> [OK] Completado en ${RUN_DURATION}s. CSV: ${CSV_FILE}"
            EVALUATED_CSVS+=("${CSV_FILE}")
        else
            echo "   -> [ERROR code ${EXIT_CODE}] Falló ${RUN_NAME}. Ver detalles en: ${LOG_FILE}"
        fi
    done
    echo ""
done

END_TOTAL_TIME=$(date +%s)
TOTAL_DURATION=$((END_TOTAL_TIME - START_TOTAL_TIME))

echo "=============================================================================="
echo "      TABLA RESUMEN DE MÉTRICAS CONSOLIDADAS (PILOTOS ÓPTIMOS)"
echo "=============================================================================="

# Consolidación estadística en Python (sin dependencias externas, usa módulos nativos csv y math)
python3 - << EOF
import os
import glob
import csv
import math

logs_dir = "${LOGS_DIR}"
algos = "${ALGOS[*]}".split()
records = []

for algo in algos:
    csvs = sorted(glob.glob(os.path.join(logs_dir, algo, "eval_" + algo + "_seed_*_*.csv")))
    for f in csvs:
        base = os.path.basename(f)
        try:
            seed_part = base.split("_seed_")[1].split("_")[0]
            seed = int(seed_part)
        except Exception:
            seed = -1

        rows = []
        try:
            with open(f, 'r', encoding='utf-8') as fp:
                reader = csv.DictReader(fp)
                for r in reader:
                    rows.append(r)
        except Exception:
            continue

        if not rows:
            continue

        n = len(rows)
        succ = sum(1 for r in rows if r.get('success', '').strip().lower() in ('true', '1')) / n * 100.0
        fall = sum(1 for r in rows if r.get('termination') == 'fall') / n * 100.0
        oob = sum(1 for r in rows if r.get('termination') == 'out_of_bounds') / n * 100.0
        crash = fall + oob
        timeout = sum(1 for r in rows if r.get('termination') == 'timeout') / n * 100.0
        survival = 100.0 - crash

        rewards = [float(r['reward']) for r in rows if r.get('reward')]
        rew_m = sum(rewards)/len(rewards) if rewards else 0.0
        rew_s = math.sqrt(sum((x - rew_m)**2 for x in rewards)/(len(rewards)-1)) if len(rewards) > 1 else 0.0

        lengths = [float(r['length']) for r in rows if r.get('length')]
        len_m = sum(lengths)/len(lengths) if lengths else 0.0
        time_flight = len_m * 0.032

        dists = [float(r['final_distance']) for r in rows if r.get('final_distance')]
        dist_m = sum(dists)/len(dists) if dists else 0.0

        obs_vals = []
        for r in rows:
            val = r.get('min_obstacle_distance') or r.get('final_obstacle_distance')
            if val is not None and val != '' and val.lower() != 'nan':
                try:
                    obs_vals.append(float(val))
                except ValueError:
                    pass
        obs_m = sum(obs_vals)/len(obs_vals) if obs_vals else None

        alt_vals = []
        for r in rows:
            val = r.get('final_altitude_m') or r.get('final_altitude_mm')
            if val is not None and val != '' and val.lower() != 'nan':
                try:
                    fval = float(val)
                    if fval > 10.0:  # compatibilidad con milímetros anteriores
                        fval /= 1000.0
                    alt_vals.append(fval)
                except ValueError:
                    pass
        alt_m = sum(alt_vals)/len(alt_vals) if alt_vals else None

        records.append({
            "algo": algo.upper(),
            "seed": seed,
            "episodes": n,
            "success_pct": succ,
            "crash_pct": crash,
            "timeout_pct": timeout,
            "survival_pct": survival,
            "reward_mean": rew_m,
            "reward_std": rew_s,
            "length_mean": len_m,
            "flight_time_s": time_flight,
            "goal_dist_m": dist_m,
            "obs_dist_m": obs_m,
            "alt_m": alt_m,
        })

if not records:
    print("[Resumen] No se encontraron datos CSV válidos para resumir.")
    exit(0)

# Guardar CSV resumen
summary_csv = os.path.join("${TABLES_DIR}", "summary_optimal_pilots.csv")
keys = list(records[0].keys())
try:
    with open(summary_csv, 'w', newline='', encoding='utf-8') as fp:
        writer = csv.DictWriter(fp, fieldnames=keys)
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)
    print(f"Resumen detallado guardado en: {summary_csv}\n")
except OSError:
    print(f"[Aviso] No se pudo escribir en {summary_csv} (modo solo lectura). Mostrando resumen en pantalla:\n")

header = f"{'ALGO':<6} | {'Seed':<5} | {'Éxito %':<8} | {'Superv %':<8} | {'Dist Meta (m)':<13} | {'Dist Obs (m)':<12} | {'Alt Final (m)':<13} | {'Recompensa':<14}"
print(header)
print("-" * len(header))
for r in records:
    obs_str = f"{r['obs_dist_m']:>6.3f}m" if r['obs_dist_m'] is not None else "    N/D     "
    alt_str = f"{r['alt_m']:>6.3f}m" if r['alt_m'] is not None else "    N/D     "
    print(f"{r['algo']:<6} | {r['seed']:<5} | {r['success_pct']:>7.1f}% | {r['survival_pct']:>7.1f}% | {r['goal_dist_m']:>7.3f}m    | {obs_str:<12} | {alt_str:<13} | {r['reward_mean']:>6.1f} ± {r['reward_std']:>4.1f}")

# Promedios globales multisemilla por algoritmo
print("\n" + "=" * 90)
print("  PROMEDIOS GLOBALES MULTISEMILLA POR ALGORITMO")
print("=" * 90)
algos_set = sorted(list(set(r['algo'] for r in records)))
agg_header = f"{'ALGO':<6} | {'Éxito Promedio':<15} | {'Supervivencia':<15} | {'Tiempo Vuelo (s)':<17} | {'Recompensa Prom.':<18}"
print(agg_header)
print("-" * len(agg_header))
for a in algos_set:
    sub = [r for r in records if r['algo'] == a]
    succ_avg = sum(r['success_pct'] for r in sub) / len(sub)
    surv_avg = sum(r['survival_pct'] for r in sub) / len(sub)
    time_avg = sum(r['flight_time_s'] for r in sub) / len(sub)
    rew_avg = sum(r['reward_mean'] for r in sub) / len(sub)
    print(f"{a:<6} | {succ_avg:>13.1f}%  | {surv_avg:>13.1f}%  | {time_avg:>15.2f}s  | {rew_avg:>16.1f}")
EOF

echo ""
echo "=============================================================================="
echo "  EVALUACIÓN FINALIZADA EN ${TOTAL_DURATION} SEGUNDOS"
echo "=============================================================================="
echo " Para regenerar la Tabla 4.2 y las demás tablas de la tesis en LaTeX:"
echo "   python3 scripts/reproduce_results.py"
echo "=============================================================================="
