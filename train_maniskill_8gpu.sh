#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-}"

case "$MODE" in
    remaining_success|remaining|remain|success)
        exec bash "$SCRIPT_DIR/train_maniskill_success_8gpu.sh"
        ;;
    success_failure|outcome_event|outcome)
        exec bash "$SCRIPT_DIR/train_maniskill_success_failure_8gpu.sh"
        ;;
    *)
        echo "Usage: bash train_maniskill_8gpu.sh {remaining_success|success_failure}" >&2
        exit 2
        ;;
esac
