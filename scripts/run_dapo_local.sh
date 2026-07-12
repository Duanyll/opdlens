#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

FINETUNE=${1:-"full"}
GPU_ID=${2:-0}

if [[ "$FINETUNE" != "full" && "$FINETUNE" != "lora" ]]; then
    echo "Error: finetune must be 'full' or 'lora'"
    exit 1
fi

ARM="b"

cd "$REPO_ROOT"

mkdir -p configs logs

CONFIG_OUT="configs/dapo_${ARM}_${FINETUNE}_local.jsonc"

echo "==> Generating config for arm=$ARM finetune=$FINETUNE (shallow layers removed) ..."
uv run python scripts/generate_dapo_math_config.py \
    --arm "$ARM" \
    --finetune "$FINETUNE" \
    --out "$CONFIG_OUT"

echo "==> Overriding devices to 1 (single GPU $GPU_ID) ..."
echo "==> Config: $CONFIG_OUT"
echo "==> Starting B-group (Logit-lens) training on cuda:$GPU_ID ..."
echo ""

CUDA_VISIBLE_DEVICES="$GPU_ID" \
LOG_DIR="logs/dapo_${ARM}_${FINETUNE}_$(date +%Y%m%d_%H%M%S)" \
uv run opdlens launch "$CONFIG_OUT" \
    --update 'launch.devices=1' \
    --update 'vllm_gpu_memory_utilization=0.40'
