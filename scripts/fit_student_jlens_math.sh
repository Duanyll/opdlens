#!/bin/bash
#SBATCH -p a800
#SBATCH -q batch
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 06:00:00
#SBATCH --job-name=fit-2b-jlens-math
#SBATCH --output=logs/%x_%j.log
#
# Round-3 (DAPO->MATH) student Jacobian lens, calibrated on real math CoT.
#
# Same calibration as fit_teacher_jlens_math.sh (hendrycks_math TRAIN, seven subjects
# interleaved, full worked solutions) but for the 2B student and the arm-E workspace
# source band {6,9,12,15,18,21} = teacher {8,12,16,20,24,28} under map_student_layer
# (superset of the [6,12,18] round-2 lens). Written to a NEW artifact so the GSM8K
# student lenses stay reproducible.

set -euo pipefail
REPO=/home/duanyll/opdlens
OUT_DIR=/gdata/users/duanyll/opdlens/artifacts/qwen3p5-2b-jlens-math-shards
OUT=/gdata/users/duanyll/opdlens/artifacts/qwen3p5-2b-jlens-math.pt
LAYERS=6,9,12,15,18,21
SUBJECTS=algebra,counting_and_probability,geometry,intermediate_algebra,number_theory,prealgebra,precalculus
MATH_REV=21a5633873b6a120296cce3e2df9d5550074f4a3
mkdir -p "$OUT_DIR"

cd "$REPO"
for shard in 0 1; do
  offset=$((shard * 132))
  srun --exclusive --exact --ntasks=1 --gpus=1 --gpus-per-task=1 \
    --cpus-per-task=8 --mem=60G \
    --container-workdir="$REPO" \
    uv run --project "$REPO" --no-sync -m opdlens.fit.fit_jlens \
    --model Qwen/Qwen3.5-2B \
    --hf-id EleutherAI/hendrycks_math --hf-name "$SUBJECTS" --hf-revision "$MATH_REV" \
    --split train --prompt-key problem --answer-key solution --include-completion \
    --source-layers "$LAYERS" --prompt-offset "$offset" --n-prompts 132 \
    --max-seq-len 384 --dim-batch 16 --checkpoint-every 8 \
    --out "$OUT_DIR/shard${shard}.pt" &
done
wait

srun --exclusive --exact --ntasks=1 --gpus-per-task=1 --cpus-per-task=1 --mem=4G \
  --container-workdir="$REPO" \
  uv run --project "$REPO" --no-sync -m opdlens.fit.merge_jlens \
  "$OUT_DIR/shard0.pt" "$OUT_DIR/shard1.pt" --out "$OUT"
