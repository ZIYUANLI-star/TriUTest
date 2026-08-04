#!/usr/bin/env bash
# scripts/run_sft.sh
set -euo pipefail

# 计算项目根目录（本脚本所在目录的上一级）
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 默认配置文件；也可以在运行时传入自定义路径：sh scripts/run_sft.sh configs/xxx.yaml
CONFIG="${1:-$ROOT_DIR/configs/sft.yaml}"

# 可选：指定使用哪个 GPU（不需要就删掉）
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# 关键：把项目根加入 Python 搜索路径，避免 "No module named 'src'"
export PYTHONPATH="${PYTHONPATH:-$ROOT_DIR}"

# 如果你还没跑过 `accelerate config`，可以直接用默认值；否则注释掉下一行，使用你的全局配置
ACCEL_FLAGS=(
  --num_processes 1
  --num_machines 1
  --mixed_precision "bf16"
  --dynamo_backend "no"
)

echo "[run_sft] ROOT_DIR=${ROOT_DIR}"
echo "[run_sft] CONFIG=${CONFIG}"
echo "[run_sft] PYTHONPATH=${PYTHONPATH}"
echo "[run_sft] Starting training..."

# 以“模块方式”启动，确保能导入 src.*
accelerate launch "${ACCEL_FLAGS[@]}" -m src.train_sft --config "${CONFIG}"
