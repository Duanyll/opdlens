#!/bin/bash
#SBATCH -p a800
#SBATCH -q batch
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 04:00:00
#SBATCH --job-name=fit-2b-jlens
#SBATCH --output=logs/%x_%j.log

set -euo pipefail
REPO=/home/duanyll/opdlens
OUT_DIR=/gdata/users/duanyll/opdlens/artifacts/qwen3p5-2b-jlens-cot-shards
OUT=/gdata/users/duanyll/opdlens/artifacts/qwen3p5-2b-jlens-cot.pt
mkdir -p "$OUT_DIR"

cd "$REPO"
srun --exclusive --ntasks=1 --gpus-per-task=1 --cpus-per-task=8 \
  --container-workdir="$REPO" \
  uv run --project "$REPO" --no-sync -m opdlens.fit.fit_jlens \
  --model Qwen/Qwen3.5-2B \
  --hf-id openai/gsm8k --hf-name main --split train \
  --prompt-key question --answer-key answer --include-completion \
  --source-layers 6,12,18 --prompt-offset 0 --n-prompts 132 \
  --max-seq-len 384 --dim-batch 16 --checkpoint-every 8 \
  --out "$OUT_DIR/shard0.pt" &
shard0_pid=$!

srun --exclusive --ntasks=1 --gpus-per-task=1 --cpus-per-task=8 \
  --container-workdir="$REPO" \
  uv run --project "$REPO" --no-sync -m opdlens.fit.fit_jlens \
  --model Qwen/Qwen3.5-2B \
  --hf-id openai/gsm8k --hf-name main --split train \
  --prompt-key question --answer-key answer --include-completion \
  --source-layers 6,12,18 --prompt-offset 132 --n-prompts 132 \
  --max-seq-len 384 --dim-batch 16 --checkpoint-every 8 \
  --out "$OUT_DIR/shard1.pt" &
shard1_pid=$!

wait "$shard0_pid"
wait "$shard1_pid"
srun --exclusive --ntasks=1 --gpus-per-task=1 --cpus-per-task=1 \
  --container-workdir="$REPO" \
  uv run --project "$REPO" --no-sync -m opdlens.fit.merge_jlens \
  "$OUT_DIR/shard0.pt" "$OUT_DIR/shard1.pt" --out "$OUT"
