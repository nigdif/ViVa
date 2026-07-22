#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VIVA_CONDA_ROOT="${VIVA_CONDA_ROOT:-/mnt/data/miniconda3}"
VIVA_CONDA_ENV="${VIVA_CONDA_ENV:-viva-lym}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints}"

usage() {
    cat <<'EOF'
Usage:
  ./train_basket_8gpu.sh <job>

Jobs:
  place_outcome      place labeled, outcome_progress
  place_outcome_sep  place labeled, outcome_separated_progress
  place_outcome_late place labeled, outcome_late_failure
  place_outcome_late_balanced
                     place labeled, outcome_late_failure, 50/50 sampling
  place_remaining    place unlabeled, remaining_progress old mode
  place_remaining_success
                     place remaining_progress, success episodes only
  pick_outcome       pick labeled, outcome_progress
  pick_remaining     pick unlabeled, remaining_progress old mode

Optional overrides:
  GPUS=0,1,2,3,4,5,6,7
  CHECKPOINT_DIR=checkpoints
  VIVA_CONDA_ROOT=/mnt/data/miniconda3
  VIVA_CONDA_ENV=viva-lym

Examples:
  ./train_basket_8gpu.sh place_outcome
  GPUS=0,1,2,3 ./train_basket_8gpu.sh pick_remaining
EOF
}

job="${1:-}"
case "$job" in
    place_outcome)
        CONFIG_PATH="config/train_basket_place_outcome_state26.yaml"
        RUN_NAME="basket_place_outcome_state26"
        ;;
    place_outcome_sep)
        CONFIG_PATH="config/train_basket_place_outcome_separated_state26.yaml"
        RUN_NAME="basket_place_outcome_separated_state26"
        ;;
    place_outcome_late)
        CONFIG_PATH="config/train_basket_place_outcome_late_failure_state26.yaml"
        RUN_NAME="basket_place_outcome_late_failure_state26"
        ;;
    place_outcome_late_balanced)
        CONFIG_PATH="config/train_basket_place_outcome_late_failure_balanced_state26.yaml"
        RUN_NAME="basket_place_outcome_late_failure_balanced_state26"
        ;;
    place_remaining)
        CONFIG_PATH="config/train_basket_place_remaining_state26.yaml"
        RUN_NAME="basket_place_remaining_state26"
        ;;
    place_remaining_success)
        CONFIG_PATH="config/train_basket_place_remaining_success_only_state26.yaml"
        RUN_NAME="basket_place_remaining_success_only_state26"
        ;;
    pick_outcome)
        CONFIG_PATH="config/train_basket_pick_outcome_state26.yaml"
        RUN_NAME="basket_pick_outcome_state26"
        ;;
    pick_remaining)
        CONFIG_PATH="config/train_basket_pick_remaining_state26.yaml"
        RUN_NAME="basket_pick_remaining_state26"
        ;;
    -h|--help|help|"")
        usage
        exit 0
        ;;
    *)
        echo "Unknown job: $job" >&2
        usage >&2
        exit 1
        ;;
esac

if [ -f "${VIVA_CONDA_ROOT}/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "${VIVA_CONDA_ROOT}/etc/profile.d/conda.sh"
    conda activate "${VIVA_CONDA_ENV}"
else
    echo "Conda activate script not found: ${VIVA_CONDA_ROOT}/etc/profile.d/conda.sh" >&2
    echo "Set VIVA_CONDA_ROOT=/path/to/miniconda3 if needed." >&2
    exit 1
fi

echo "Starting job: ${job}"
echo "GPUS=${GPUS}"
echo "CONFIG_PATH=${CONFIG_PATH}"
echo "RUN_NAME=${RUN_NAME}"
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}"

GPUS="${GPUS}" \
CONFIG_PATH="${CONFIG_PATH}" \
RUN_NAME="${RUN_NAME}" \
CHECKPOINT_DIR="${CHECKPOINT_DIR}" \
bash train.sh
