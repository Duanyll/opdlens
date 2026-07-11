#!/bin/bash
#SBATCH -p a800
#SBATCH -q batch
#SBATCH --gpus=2
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH -t 06:00:00
#SBATCH --output=logs/%x_%j.log

set -euo pipefail
CONFIG=${1:?usage: sbatch -J NAME scripts/run_round1.sh CONFIG COMMIT}
COMMIT=${2:?usage: sbatch -J NAME scripts/run_round1.sh CONFIG COMMIT}
REPO=/home/duanyll/opdlens
SNAPSHOT=/gdata/users/duanyll/opdlens/snapshots/$SLURM_JOB_ID

git -C "$REPO" cat-file -e "$COMMIT^{commit}"
mkdir -p "$SNAPSHOT"
git -C "$REPO" archive "$COMMIT" | tar -x -C "$SNAPSHOT"

cd "$REPO"
srun --container-workdir="$REPO" \
  env PYTHONPATH="$SNAPSHOT" OPDLENS_EXPERIMENT_COMMIT="$COMMIT" \
  uv run --project "$REPO" --no-sync \
  opdlens launch "$SNAPSHOT/$CONFIG"
