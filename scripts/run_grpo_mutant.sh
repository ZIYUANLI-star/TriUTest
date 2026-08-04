#!/usr/bin/env bash
# Stage 3: SM-GRPO.  Usage: bash scripts/run_grpo_mutant.sh configs/grpo_triu3b_sa.yaml [train_file]
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-$ROOT_DIR/configs/grpo_triu3b_sa.yaml}"
TRAIN_FILE="${2:-$ROOT_DIR/data/your_dataset_with_id.json}"
export PYTHONPATH="${PYTHONPATH:-$ROOT_DIR}"

accelerate launch \
  --num_processes 1 \
  --num_machines 1 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  -m src.train_sm_grpo \
  --config "$CONFIG" \
  --train_file "$TRAIN_FILE"
