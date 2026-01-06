#!/usr/bin/env bash
set -e
accelerate launch --num_processes=1 --mixed_precision=bf16 src/train_grpo.py --config configs/grpo_speed_stable.yaml --train_file data/your_dataset_with_id.json
