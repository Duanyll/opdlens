#!/bin/bash
#SBATCH -p a800
#SBATCH -q batch
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH -t 05:00:00
#SBATCH --output=logs/%x_%j.log
# 4-GPU GKD reproduction run. Usage: sbatch -J <name> scripts/run_repro.sh <config.jsonc>
# global_batch_size is held constant, so per-rank prompts = 96/4 = 24 (same dynamics as 1 GPU).
set -euo pipefail
CONFIG=${1:?usage: sbatch -J <name> scripts/run_repro.sh <config.jsonc>}
srun --container-workdir=/home/duanyll/opdlens \
  uv run --project /home/duanyll/opdlens \
  opdlens launch "$CONFIG" --update 'launch.devices=4'
