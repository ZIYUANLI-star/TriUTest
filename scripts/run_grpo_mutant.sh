#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=7
export MUTPY_PYTHON="D:\ANACONDA\envs\MuTAP-master\python.exe"
export MUTPY_SCRIPT="D:\ANACONDA\envs\MuTAP-master\Scripts\mut.py"

set -euo pipefail

# 进入项目根目录（脚本在 scripts/ 下）
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# 保险：把项目根加入 PYTHONPATH
export PYTHONPATH="$PWD"

CONFIG="configs/grpo_mutant_max.yaml"           # ← 你的 mutant 训练配置
TRAIN_FILE="data/your_dataset_with_id.json"     # ← 你的训练集

accelerate launch \
  --num_processes 1 \
  --num_machines 1 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  -m src.train_grpo \
  --config "$CONFIG" \
  --train_file "$TRAIN_FILE"
