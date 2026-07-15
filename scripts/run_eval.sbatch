#!/bin/bash
#SBATCH -p a800
#SBATCH -q batch
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 04:00:00
#SBATCH --output=logs/%x_%j.log
#
# Eval launcher (no training). Usage:
#   sbatch -J <name> scripts/run_eval.sbatch <config.jsonc> [--checkpoint <dir>]
#
# Runs `opdlens eval`, which loads the config's student (frozen) — or the given
# training checkpoint's student weights — and scores the config's eval_benchmarks
# with the same grader as training. torchrun is sized to the GPUs Slurm allocated.

set -euo pipefail
CONFIG=${1:?usage: sbatch -J <name> scripts/run_eval.sbatch <config.jsonc> [--checkpoint <dir>]}
shift
REPO=/home/duanyll/opdlens
DEVICES="${SLURM_GPUS_ON_NODE:-2}"

cd "$REPO"
srun --container-workdir="$REPO" \
  uv run --project "$REPO" --no-sync \
  python -P -m opdlens.scripts.cli eval "$CONFIG" "$@" \
  --update "launch.devices=$DEVICES"
