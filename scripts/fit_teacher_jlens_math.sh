#!/bin/bash
#SBATCH -p a800
#SBATCH -q batch
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH -t 08:00:00
#SBATCH --job-name=fit-9b-jlens-math
#SBATCH --output=logs/%x_%j.log
#
# Round-3 (DAPO->MATH) teacher Jacobian lens, calibrated on real math CoT.
#
# DAPO-Math's `solution` field is only the bare final answer, so it cannot supply a
# generation-distribution completion the way GSM8K's worked `answer` did. We instead
# calibrate on hendrycks_math TRAIN (the round-3 eval distribution), whose `solution`
# is a full worked derivation ending in \boxed{}. The seven subjects are round-robin
# interleaved (comma-separated --hf-name) so every prompt-offset shard is subject
# balanced. Source layers match the existing GSM8K teacher lens exactly
# (/gdata/users/duanyll/jlens/qwen3p5_9b_v2/lens.pt) so it is a drop-in arm-C/E swap.
#
# 4 shards x 66 = 264 prompts (matches the GSM8K lens n_prompts). The 9B fit is heavy
# (~2 min/prompt for 8 layers at d_model=4096); dim-batch is halved to 8 and
# expandable_segments is on to stay well under the 80 GB a800 (the old 44 GB L40 run
# OOM'd at dim-batch 16).

set -euo pipefail
REPO=/home/duanyll/opdlens
OUT_DIR=/gdata/users/duanyll/opdlens/artifacts/qwen3p5-9b-jlens-math-shards
OUT=/gdata/users/duanyll/opdlens/artifacts/qwen3p5-9b-jlens-math.pt
LAYERS=4,8,12,16,20,24,28,30
SUBJECTS=algebra,counting_and_probability,geometry,intermediate_algebra,number_theory,prealgebra,precalculus
MATH_REV=21a5633873b6a120296cce3e2df9d5550074f4a3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT_DIR"

cd "$REPO"
for shard in 0 1 2 3; do
  offset=$((shard * 66))
  srun --exclusive --exact --ntasks=1 --gpus=1 --gpus-per-task=1 \
    --cpus-per-task=8 --mem=58G \
    --container-workdir="$REPO" \
    uv run --project "$REPO" --no-sync -m opdlens.fit.fit_jlens \
    --model Qwen/Qwen3.5-9B \
    --hf-id EleutherAI/hendrycks_math --hf-name "$SUBJECTS" --hf-revision "$MATH_REV" \
    --split train --prompt-key problem --answer-key solution --include-completion \
    --source-layers "$LAYERS" --prompt-offset "$offset" --n-prompts 66 \
    --max-seq-len 384 --dim-batch 8 --checkpoint-every 8 \
    --out "$OUT_DIR/shard${shard}.pt" &
done
wait

srun --exclusive --exact --ntasks=1 --gpus-per-task=1 --cpus-per-task=1 --mem=8G \
  --container-workdir="$REPO" \
  uv run --project "$REPO" --no-sync -m opdlens.fit.merge_jlens \
  "$OUT_DIR"/shard0.pt "$OUT_DIR"/shard1.pt "$OUT_DIR"/shard2.pt "$OUT_DIR"/shard3.pt \
  --out "$OUT"
