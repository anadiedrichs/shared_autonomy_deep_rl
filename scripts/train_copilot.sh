#!/usr/bin/env bash
# ==============================================================================
# Script de Entrenamiento del Copiloto Residual Constreñido (Schaff & Walter, 2020)
# TFM: Autonomía Compartida para UAVs con Webots y RLtools
# Autora: Ana Laura Diedrichs
# ==============================================================================
# Entrena la política residual a_r mediante ConstrainedPPO con multiplicador de
# Lagrange adaptable y fase de calentamiento (Warm-Up) donde a_r = 0.
#
# Registra automáticamente métricas en TensorBoard:
#   - train/lambda
#   - train/loss_lambda
#   - train/mean_residual_norm
#   - train/ppo_loss
#   - train/is_warmup
#
# Uso:
#   bash scripts/train_copilot.sh
#
# Para visualizar el entrenamiento en TensorBoard:
#   tensorboard --logdir logs/copilot_constrained_ppo --host 0.0.0.0 --port 6006
# ==============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Configuración por defecto
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-100000}
WARMUP_STEPS=${WARMUP_STEPS:-10000}
PILOT_TYPE=${PILOT_TYPE:-"noisy"}
NOISE_STD=${NOISE_STD:-0.50}
LAG_PROB=${LAG_PROB:-0.80}
TARGET_D=${TARGET_D:-0.0}
INIT_LAMBDA=${INIT_LAMBDA:-1.0}
LR_LAMBDA=${LR_LAMBDA:-0.0003}
RESIDUAL_SCALE=${RESIDUAL_SCALE:-1.0}
CORNER=${CORNER:-"cone_1"}
SAVE_PATH=${SAVE_PATH:-"./models/copilot_constrained_ppo.zip"}
LOG_DIR=${LOG_DIR:-"./logs/copilot_constrained_ppo"}
RESULTS_DIR=${RESULTS_DIR:-"results_TFM"}

# Detección de checkpoint base del piloto óptimo si existe
if [ -z "${OPTIMAL_CHECKPOINT}" ]; then
    if [ -f "${RESULTS_DIR}/models/optimal_pilot_td3_checkpoint.h" ]; then
        OPTIMAL_CHECKPOINT="${RESULTS_DIR}/models/optimal_pilot_td3_checkpoint.h"
        OPTIMAL_BACKEND="rltools_td3"
    elif [ -f "${RESULTS_DIR}/models/optimal_pilot_sac_checkpoint.h" ]; then
        OPTIMAL_CHECKPOINT="${RESULTS_DIR}/models/optimal_pilot_sac_checkpoint.h"
        OPTIMAL_BACKEND="rltools_sac"
    elif [ -f "models/ppo_optimal_pilot.zip" ]; then
        OPTIMAL_CHECKPOINT="models/ppo_optimal_pilot.zip"
        OPTIMAL_BACKEND="sb3_ppo"
    elif [ -f "${RESULTS_DIR}/models/ppo_optimal_pilot.zip" ]; then
        OPTIMAL_CHECKPOINT="${RESULTS_DIR}/models/ppo_optimal_pilot.zip"
        OPTIMAL_BACKEND="sb3_ppo"
    else
        OPTIMAL_CHECKPOINT=""
        OPTIMAL_BACKEND="heuristic"
    fi
fi

mkdir -p "$(dirname "${SAVE_PATH}")"
mkdir -p "${LOG_DIR}"

echo "=============================================================================="
echo " INICIANDO ENTRENAMIENTO: COPILOTO PPO CONSTREÑIDO (SCHAFF & WALTER 2020)"
echo "=============================================================================="
echo " - Total Timesteps   : ${TOTAL_TIMESTEPS}"
echo " - Pasos de Warm-Up  : ${WARMUP_STEPS} (con a_r = 0)"
echo " - Piloto Subrogado  : ${PILOT_TYPE} [${OPTIMAL_BACKEND}]"
echo " - Umbral target_d   : ${TARGET_D}"
echo " - Lambda inicial    : ${INIT_LAMBDA} (lr=${LR_LAMBDA})"
echo " - Factor beta       : ${RESIDUAL_SCALE}"
echo " - Checkpoint destino: ${SAVE_PATH}"
echo " - Logs TensorBoard  : ${LOG_DIR}"
echo "=============================================================================="

OPT_FLAG=""
if [ -n "${OPTIMAL_CHECKPOINT}" ]; then
    OPT_FLAG="--optimal-checkpoint ${OPTIMAL_CHECKPOINT}"
fi

python3 src/training/train_copilot.py \
    --total-timesteps "${TOTAL_TIMESTEPS}" \
    --warmup-steps "${WARMUP_STEPS}" \
    --pilot-type "${PILOT_TYPE}" \
    ${OPT_FLAG} \
    --optimal-backend "${OPTIMAL_BACKEND}" \
    --noise-std "${NOISE_STD}" \
    --lag-prob "${LAG_PROB}" \
    --target-threshold "${TARGET_D}" \
    --init-lambda "${INIT_LAMBDA}" \
    --lr-lambda "${LR_LAMBDA}" \
    --residual-scale "${RESIDUAL_SCALE}" \
    --corner "${CORNER}" \
    --save-path "${SAVE_PATH}" \
    --log-dir "${LOG_DIR}" \
    --verbose 1

echo "=============================================================================="
echo " ENTRENAMIENTO COMPLETADO"
echo " Modelo guardado en: ${SAVE_PATH}"
echo " Para ver curvas en TensorBoard: tensorboard --logdir ${LOG_DIR}"
echo "=============================================================================="
