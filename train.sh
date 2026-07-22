# Training script for Viva - 3 Tasks
# 7-frame: [blank, state, cam_left, cam_right, cam_high, future_state, value]
# future_offset: 50 frames
         

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR"

FUTURE_STATE_WEIGHT="${FUTURE_STATE_WEIGHT:-0.5}"
VALUE_WEIGHT="1.0"

CONFIG_PATH="${CONFIG_PATH:-./config/train_8gpu_3task.yaml}"
LOG_DIR="${LOG_DIR:-./logs}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-./checkpoints}"
RUN_NAME="${RUN_NAME:-train_3task}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_NAME}.log}"

if [ -n "${GPUS:-}" ]; then
    CUDA_DEVICES="$GPUS"
elif [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    CUDA_DEVICES="$CUDA_VISIBLE_DEVICES"
else
    GPU_COUNT="$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ')"
    if [ "${GPU_COUNT:-0}" -le 0 ]; then
        echo "No GPU detected. Set GPUS or CUDA_VISIBLE_DEVICES." >&2
        exit 1
    fi
    CUDA_DEVICES="$(seq -s, 0 "$((GPU_COUNT - 1))")"
fi

GPU_COUNT="$(python - <<PY
devices = "${CUDA_DEVICES}".strip()
print(len([device for device in devices.split(",") if device.strip()]))
PY
)"

echo "=========================================="
echo "VivaModel Training - 3 Tasks"
echo "=========================================="
echo "Config:                  $CONFIG_PATH"
echo "Checkpoint base:         $CHECKPOINT_DIR"
echo "loss_weight_future_state: $FUTURE_STATE_WEIGHT"
echo "loss_weight_value:       $VALUE_WEIGHT (from yaml)"
echo "Run name:                $RUN_NAME"
echo "Log:                     $LOG_FILE"
echo "GPUs:                    $CUDA_DEVICES ($GPU_COUNT process(es))"
echo "=========================================="

if [ "${DRY_RUN:-0}" = "1" ]; then
    exit 0
fi

mkdir -p "$LOG_DIR"
mkdir -p "$CHECKPOINT_DIR"

CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" accelerate launch \
    --num_processes "$GPU_COUNT" \
    --mixed_precision bf16 \
    train.py \
    --config "$CONFIG_PATH" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --run_name "$RUN_NAME" \
    --loss_weight_future_state "$FUTURE_STATE_WEIGHT" \
    2>&1 | tee "$LOG_FILE"
