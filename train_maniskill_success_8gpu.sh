#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/mnt/nas/linyumin/projects/ViVa_mode"
CONDA_SH="/mnt/data/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="viva-lym"

cd "$PROJECT_DIR"
source "$CONDA_SH"
conda activate "$CONDA_ENV"

export CONFIG_PATH="./config/train_maniskill_remaining_success_state9.yaml"
export RUN_NAME="maniskill_remaining_success_state9"
export CHECKPOINT_DIR="./checkpoints/maniskill_remaining_success_state9"
export LOG_FILE="./logs/maniskill_remaining_success_state9.log"

exec bash ./train.sh
