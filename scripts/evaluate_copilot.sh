#!/usr/bin/env bash
# ==============================================================================
# Script de Evaluación del Copiloto Residual Constreñido (Schaff & Walter, 2020)
# TFM: Autonomía Compartida para UAVs con Webots y RLtools
# Autora: Ana Laura Diedrichs
# ==============================================================================
# Evalúa el copiloto entrenado con PPO Constreñido frente a 4 perfiles degradados
# comparando la condición Placebo (sin copiloto) vs Asistido (con copiloto):
#   1) NoisyPilot con sigma = 0.5   (ruido motriz moderado)
#   2) NoisyPilot con sigma = 1.0   (ruido motriz al 100%)
#   3) LaggyPilot con p_repeat = 0.8 (retardo humano ZOH)
#   4) StandByPilot con p_drop = 0.8 (pérdida de paquetes con hover)
#
# Métricas cuantitativas calculadas:
#   - Tasa de Éxito (%)
#   - Tasa de Colisión (%)
#   - Norma Media de Intervención (||a_r||_2)
#   - Métrica de Suavidad / Jerk (||a_r(t) - a_r(t-1)||_2)
#
# Uso:
#   bash scripts/evaluate_copilot.sh
#
# Parámetros opcionales vía variables de entorno:
#   COPILOT_CHECKPOINT="models/copilot_constrained_ppo.zip"
#   SEEDS="42 123 256 512 789"
#   EPISODES=20
#   OUTPUT_DIR="./eval_results"
# ==============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Configuración por defecto
RESULTS_DIR=${RESULTS_DIR:-"results_TFM"}
COPILOT_CHECKPOINT=${COPILOT_CHECKPOINT:-"${RESULTS_DIR}/models/copilot_constrained_sac_123_noisy_05.zip"}
OUTPUT_DIR=${OUTPUT_DIR:-"${RESULTS_DIR}/logs/copilot_eval"}
EPISODES=${EPISODES:-20}
CORNER=${CORNER:-"cone_1"}
SEEDS=${SEEDS:-"42 123 256 512 789"}

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
        echo "[AVISO] No se detectó checkpoint de piloto óptimo entrenado."
        echo "        Utilizando HeuristicPilot como surrogate base."
        OPTIMAL_CHECKPOINT=""
        OPTIMAL_BACKEND="heuristic"
    fi
fi

mkdir -p "${OUTPUT_DIR}"

echo "=============================================================================="
echo " INICIANDO EVALUACIÓN COMPARATIVA DEL COPILOTO RESIDUAL CONSTREÑIDO"
echo "=============================================================================="
echo " - Checkpoint Copiloto : ${COPILOT_CHECKPOINT}"
echo " - Piloto Óptimo Base  : ${OPTIMAL_CHECKPOINT:-"(Heuristic)"} [${OPTIMAL_BACKEND}]"
echo " - Semillas            : ${SEEDS}"
echo " - Cono Objetivo       : ${CORNER}"
echo " - Salida Resultados   : ${OUTPUT_DIR}"
echo "=============================================================================="

OPT_FLAG=""
if [ -n "${OPTIMAL_CHECKPOINT}" ]; then
    OPT_FLAG="--optimal-checkpoint ${OPTIMAL_CHECKPOINT}"
fi

python3 src/evaluation/evaluate_copilot.py \
    --copilot-checkpoint "${COPILOT_CHECKPOINT}" \
    ${OPT_FLAG} \
    --optimal-backend "${OPTIMAL_BACKEND}" \
    --pilots noisy_0.5 noisy_1.0 laggy_0.8 standby_0.8 \
    --seeds ${SEEDS} \
    --corner "${CORNER}" \
    --output-dir "${OUTPUT_DIR}" \
    --verbose 1

echo "=============================================================================="
echo " EVALUACIÓN COMPLETADA CON ÉXITO"
echo " Resultados y figuras generadas en: ${OUTPUT_DIR}/"
echo "=============================================================================="
