#!/usr/bin/env bash
# Stage 2: DQ-GKD distillation.  Usage: bash scripts/run_distill_gkd.sh configs/gkd_3b_14b.yaml
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-$ROOT_DIR/configs/gkd_3b_14b.yaml}"
export PYTHONPATH="${PYTHONPATH:-$ROOT_DIR}"

accelerate launch \
  --num_processes 1 \
  --num_machines 1 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  -m src.train_dq_gkd \
  --config "$CONFIG"
