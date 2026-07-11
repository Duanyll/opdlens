#!/bin/bash
#SBATCH -p a800
#SBATCH -q batch
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 04:00:00
#SBATCH --job-name=fit-2b-jlens
#SBATCH --output=logs/%x_%j.log

set -euo pipefail
REPO=/home/duanyll/opdlens
OUT=/gdata/users/duanyll/opdlens/artifacts/qwen3p5-2b-jlens.pt
mkdir -p "$(dirname "$OUT")"

cd "$REPO"
srun --container-workdir="$REPO" \
  uv run --project "$REPO" --no-sync -m opdlens.fit.fit_jlens \
  --model Qwen/Qwen3.5-2B \
  --hf-id openai/gsm8k --hf-name main --split train --prompt-key question \
  --source-layers 6,12,18 --n-prompts 264 --max-seq-len 384 \
  --dim-batch 16 --checkpoint-every 8 --out "$OUT"
