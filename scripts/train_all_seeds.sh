#!/usr/bin/env bash
# ==============================================================================
# Script de Entrenamiento y Evaluación Multisemilla para Pilotos Óptimos
# TFM: Autonomía Compartida para UAVs con Webots y RLtools
# Autora: Ana Laura Diedrichs
# ==============================================================================
# Entrena y evalúa los tres algoritmos continuos (TD3, SAC y PPO) sobre 5 semillas
# independientes, garantizando la reproducibilidad estadística del piloto óptimo.
# Todos los artefactos generados se guardan dentro de results_TFM/.
#
# Uso básico:
#   bash scripts/train_all_seeds.sh             # Entrena y evalúa los 3 modelos (TD3, SAC, PPO)
#   bash scripts/train_all_seeds.sh td3         # Entrena y evalúa solo TD3 (5 semillas)
#   bash scripts/train_all_seeds.sh sac         # Entrena y evalúa solo SAC (5 semillas)
#   bash scripts/train_all_seeds.sh ppo         # Entrena y evalúa solo PPO (5 semillas)
#   bash scripts/train_all_seeds.sh td3 sac     # Entrena y evalúa TD3 y SAC
#
# Variables configurables por entorno:
#   SEEDS="42 123 256 512 789"  # 5 semillas independientes
#   STEPS=150000                 # Pasos de entrenamiento por corrida
#   EVAL_EPISODES=100            # Episodios de evaluación por modelo
#   GAMMA=0.97                   # Factor de descuento unificado
#   CORNER="cone_1"              # Cono objetivo en Webots
#   SKIP_EXISTING=1              # 1: omite reentrenar si ya existe el checkpoint
# ==============================================================================

set -o pipefail

# Ubicar raíz del repositorio
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Configuración por defecto
SEEDS=(${SEEDS:-42 123 256 512 789})
STEPS=${STEPS:-150000}
EVAL_EPISODES=${EVAL_EPISODES:-100}
GAMMA=${GAMMA:-0.97}
CORNER=${CORNER:-cone_1}
SKIP_EXISTING=${SKIP_EXISTING:-1}
EVAL_SEED=${EVAL_SEED:-1000}

# Modelos a ejecutar (por defecto los 3)
if [ $# -gt 0 ]; then
    REQUESTED_MODELS=("$@")
else
    REQUESTED_MODELS=("td3" "sac" "ppo")
fi

# Directorios de destino en results_TFM
RESULTS_DIR="results_TFM"
MODELS_DIR="${RESULTS_DIR}/models"
LOGS_DIR="${RESULTS_DIR}/logs"

mkdir -p "${MODELS_DIR}"
mkdir -p "${LOGS_DIR}/td3" "${LOGS_DIR}/sac" "${LOGS_DIR}/ppo"

echo "=============================================================================="
echo "    CAMPAÑA DE ENTRENAMIENTO Y EVALUACIÓN MULTISEMILLA (5 SEMILLAS)"
echo "=============================================================================="
echo " Raíz del proyecto : ${PROJECT_ROOT}"
echo " Modelos elegidos  : ${REQUESTED_MODELS[*]}"
echo " Semillas fijadas  : ${SEEDS[*]} (Total: ${#SEEDS[@]} semillas)"
echo " Pasos por modelo  : ${STEPS}"
echo " Factor gamma      : ${GAMMA}"
echo " Cono objetivo     : ${CORNER}"
echo " Episodios de eval : ${EVAL_EPISODES} (Semilla maestra eval: ${EVAL_SEED})"
echo " Directorio salida : ${RESULTS_DIR}/"
echo " Omitir existentes : $([ "${SKIP_EXISTING}" -eq 1 ] && echo "Sí" || echo "No")"
echo "=============================================================================="
echo ""

START_TOTAL_TIME=$(date +%s)

for MODEL in "${REQUESTED_MODELS[@]}"; do
    MODEL_LOWER=$(echo "${MODEL}" | tr '[:upper:]' '[:lower:]')
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    echo "  INICIANDO BLOQUE PARA EL MODELO: ${MODEL_LOWER^^}"
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

    for SEED in "${SEEDS[@]}"; do
        echo ""
        echo "------------------------------------------------------------------------------"
        echo "  [${MODEL_LOWER^^}] Corrida con Semilla: ${SEED}"
        echo "------------------------------------------------------------------------------"

        case "${MODEL_LOWER}" in
            td3)
                MODEL_FILE="${MODELS_DIR}/optimal_pilot_td3_seed_${SEED}.h"
                LOG_DIR_RUN="${LOGS_DIR}/td3"
                EVAL_CSV="${LOGS_DIR}/td3/eval_td3_seed_${SEED}_${EVAL_EPISODES}ep.csv"
                EVAL_PNG="${LOGS_DIR}/td3/trayectoria_td3_seed_${SEED}_${EVAL_EPISODES}ep.png"

                # 1. Entrenamiento
                if [ "${SKIP_EXISTING}" -eq 1 ] && [ -f "${MODEL_FILE}" ]; then
                    echo "  [INFO] Checkpoint ya existe: ${MODEL_FILE}. Omitiendo entrenamiento."
                else
                    echo "  [PASO 1/2] Entrenando TD3 (${STEPS} pasos, seed ${SEED})..."
                    python3 src/training/train_rltools_td3.py \
                        --steps "${STEPS}" \
                        --gamma "${GAMMA}" \
                        --corner "${CORNER}" \
                        --seed "${SEED}" \
                        --export-path "${MODEL_FILE}" \
                        --log-dir "${LOG_DIR_RUN}"
                fi

                # 2. Evaluación
                if [ "${SKIP_EXISTING}" -eq 1 ] && [ -f "${EVAL_CSV}" ]; then
                    echo "  [INFO] Evaluación ya existe: ${EVAL_CSV}. Omitiendo evaluación."
                else
                    echo "  [PASO 2/2] Evaluando TD3 (${EVAL_EPISODES} episodios, seed maestro ${EVAL_SEED})..."
                    python3 src/evaluation/metrics.py \
                        --pilot-type optimal \
                        --backend rltools_td3 \
                        --checkpoint "${MODEL_FILE}" \
                        --episodes "${EVAL_EPISODES}" \
                        --seed "${EVAL_SEED}" \
                        --corner "${CORNER}" \
                        --save-csv "${EVAL_CSV}" \
                        --plot-trajectory-png "${EVAL_PNG}"
                fi
                ;;

            sac)
                MODEL_FILE="${MODELS_DIR}/optimal_pilot_sac_seed_${SEED}.h"
                LOG_DIR_RUN="${LOGS_DIR}/sac"
                EVAL_CSV="${LOGS_DIR}/sac/eval_sac_seed_${SEED}_${EVAL_EPISODES}ep.csv"
                EVAL_PNG="${LOGS_DIR}/sac/trayectoria_sac_seed_${SEED}_${EVAL_EPISODES}ep.png"

                # 1. Entrenamiento
                if [ "${SKIP_EXISTING}" -eq 1 ] && [ -f "${MODEL_FILE}" ]; then
                    echo "  [INFO] Checkpoint ya existe: ${MODEL_FILE}. Omitiendo entrenamiento."
                else
                    echo "  [PASO 1/2] Entrenando SAC (${STEPS} pasos, seed ${SEED})..."
                    python3 src/training/train_rltools_sac.py \
                        --steps "${STEPS}" \
                        --gamma "${GAMMA}" \
                        --corner "${CORNER}" \
                        --seed "${SEED}" \
                        --export-path "${MODEL_FILE}" \
                        --log-dir "${LOG_DIR_RUN}"
                fi

                # 2. Evaluación
                if [ "${SKIP_EXISTING}" -eq 1 ] && [ -f "${EVAL_CSV}" ]; then
                    echo "  [INFO] Evaluación ya existe: ${EVAL_CSV}. Omitiendo evaluación."
                else
                    echo "  [PASO 2/2] Evaluando SAC (${EVAL_EPISODES} episodios, seed maestro ${EVAL_SEED})..."
                    python3 src/evaluation/metrics.py \
                        --pilot-type optimal \
                        --backend rltools_sac \
                        --checkpoint "${MODEL_FILE}" \
                        --episodes "${EVAL_EPISODES}" \
                        --seed "${EVAL_SEED}" \
                        --corner "${CORNER}" \
                        --save-csv "${EVAL_CSV}" \
                        --plot-trajectory-png "${EVAL_PNG}"
                fi
                ;;

            ppo)
                MODEL_PATH_BASE="${MODELS_DIR}/ppo_optimal_pilot_seed_${SEED}"
                MODEL_FILE="${MODEL_PATH_BASE}.zip"
                LOG_DIR_RUN="${LOGS_DIR}/ppo"
                EVAL_CSV="${LOGS_DIR}/ppo/eval_ppo_seed_${SEED}_${EVAL_EPISODES}ep.csv"
                EVAL_PNG="${LOGS_DIR}/ppo/trayectoria_ppo_seed_${SEED}_${EVAL_EPISODES}ep.png"

                # 1. Entrenamiento
                if [ "${SKIP_EXISTING}" -eq 1 ] && [ -f "${MODEL_FILE}" ]; then
                    echo "  [INFO] Checkpoint ya existe: ${MODEL_FILE}. Omitiendo entrenamiento."
                else
                    echo "  [PASO 1/2] Entrenando PPO (${STEPS} pasos, seed ${SEED})..."
                    python3 src/training/train_sb3_ppo.py \
                        --timesteps "${STEPS}" \
                        --gamma "${GAMMA}" \
                        --learning-rate 3e-4 \
                        --corner "${CORNER}" \
                        --seed "${SEED}" \
                        --model-path "${MODEL_PATH_BASE}" \
                        --log-dir "${LOG_DIR_RUN}"
                fi

                # 2. Evaluación
                if [ "${SKIP_EXISTING}" -eq 1 ] && [ -f "${EVAL_CSV}" ]; then
                    echo "  [INFO] Evaluación ya existe: ${EVAL_CSV}. Omitiendo evaluación."
                else
                    echo "  [PASO 2/2] Evaluando PPO (${EVAL_EPISODES} episodios, seed maestro ${EVAL_SEED})..."
                    python3 src/evaluation/metrics.py \
                        --pilot-type optimal \
                        --backend sb3_ppo \
                        --checkpoint "${MODEL_FILE}" \
                        --episodes "${EVAL_EPISODES}" \
                        --seed "${EVAL_SEED}" \
                        --corner "${CORNER}" \
                        --save-csv "${EVAL_CSV}" \
                        --plot-trajectory-png "${EVAL_PNG}"
                fi
                ;;

            *)
                echo "  [ERROR] Modelo desconocido: '${MODEL}'. Opciones válidas: td3, sac, ppo."
                ;;
        esac
    done
done

END_TOTAL_TIME=$(date +%s)
ELAPSED_TOTAL=$((END_TOTAL_TIME - START_TOTAL_TIME))

echo ""
echo "=============================================================================="
echo "    ¡CAMPAÑA MULTISEMILLA FINALIZADA CON ÉXITO!"
echo "=============================================================================="
echo " Tiempo total transcurrido : ${ELAPSED_TOTAL} segundos"
echo " Modelos procesados        : ${REQUESTED_MODELS[*]}"
echo " Semillas procesadas       : ${SEEDS[*]}"
echo ""
echo " Resumen de artefactos en ${RESULTS_DIR}/:"
echo "   - Modelos exportados    : ${MODELS_DIR}/"
echo "   - Logs y TensorBoard    : ${LOGS_DIR}/"
echo "   - Evaluaciones (CSVs)   : ${LOGS_DIR}/*/*.csv"
echo "   - Gráficos trayectorias : ${LOGS_DIR}/*/*.png"
echo "=============================================================================="
