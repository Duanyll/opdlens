#!/bin/bash
#SBATCH -p a800
#SBATCH -q batch
#SBATCH --gpus=2
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH -t 1-00:00:00
#SBATCH --output=logs/%x_%j.log
#
# The single training launcher. Usage:
#   sbatch -J <name> scripts/run.sbatch <config.jsonc> [<commit>]
#
# Archives the repo at <commit> (default HEAD) into an immutable snapshot and
# trains from it, exporting OPDLENS_EXPERIMENT_COMMIT so the exact code version is
# recorded in trackio. torchrun is sized to the GPUs Slurm actually allocated, so
# override GPUs/time/mem on the sbatch line and the config follows, e.g.:
#   sbatch -J repro --gpus=4 -t 06:00:00 scripts/run.sbatch experiments/repro_gkd/repro_gkd_2b_full.jsonc

set -euo pipefail
CONFIG=${1:?usage: sbatch -J <name> scripts/run.sbatch <config.jsonc> [<commit>]}
REPO=/home/duanyll/opdlens
COMMIT=$(git -C "$REPO" rev-parse "${2:-HEAD}")
DEVICES="${SLURM_GPUS_ON_NODE:-2}"
SNAPSHOT=/gdata/users/duanyll/opdlens/snapshots/$SLURM_JOB_ID

# vLLM/inductor compile cache on node-local NVMe scratch (/gds), keyed to this job's
# GPU pair. Slurm hands out aligned 2-GPU pairs (IDX 0-1/2-3/4-5/6-7) and only one
# job holds a pair at a time, so each disk (gpu01/23/45/67) has a single writer —
# the cache persists and is reused across jobs (compile once per pair, not per run).
# Falls back to ephemeral per-job /tmp if /gds or the GPU list is unavailable; never
# the shared NFS path, whose non-atomic writes corrupted the cache before.
GPU_LIST="${SLURM_JOB_GPUS:-${GPU_DEVICE_ORDINAL:-}}"
FIRST_GPU="${GPU_LIST%%,*}"
if ! [[ "$FIRST_GPU" =~ ^[0-9]+$ ]]; then  # env vars vary; scontrol IDX is reliable
  FIRST_GPU=$(scontrol show job "$SLURM_JOB_ID" -d 2>/dev/null \
    | grep -oE 'IDX:[0-9]+' | head -1 | cut -d: -f2)
fi
CACHE_ROOT="/tmp/opdlens-cache-$SLURM_JOB_ID"
if [[ "$FIRST_GPU" =~ ^[0-9]+$ ]]; then
  PAIR=$((FIRST_GPU - FIRST_GPU % 2))
  GDS="/gds/gpu${PAIR}$((PAIR + 1))"
  [[ -d "$GDS" ]] && CACHE_ROOT="$GDS/opdlens-cache"
fi
mkdir -p "$CACHE_ROOT/vllm" "$CACHE_ROOT/torchinductor"
echo "[run.sbatch] compile cache: $CACHE_ROOT (SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset})"

mkdir -p "$SNAPSHOT"
git -C "$REPO" archive "$COMMIT" | tar -x -C "$SNAPSHOT"

cd "$REPO"
srun --container-workdir="$REPO" \
  /usr/bin/env PYTHONSAFEPATH=1 PYTHONPATH="$SNAPSHOT" \
  VLLM_CACHE_ROOT="$CACHE_ROOT/vllm" \
  TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/torchinductor" \
  OPDLENS_EXPERIMENT_COMMIT="$COMMIT" \
  uv run --project "$REPO" --no-sync \
  python -P -m opdlens.scripts.cli launch "$SNAPSHOT/$CONFIG" \
  --update "launch.devices=$DEVICES"
