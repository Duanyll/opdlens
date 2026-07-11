#!/bin/bash
#SBATCH -p a800
#SBATCH -q batch
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH -t 06:00:00
#SBATCH --job-name=repro-full
#SBATCH --output=logs/repro_full_%j.log
# Full fine-tune GKD reproduction: Qwen3.5-9B -> 2B, 300 steps, 1 GPU.
set -euo pipefail
srun --container-workdir=/home/duanyll/opdlens \
  uv run --project /home/duanyll/opdlens \
  opdlens launch examples/repro_gkd_2b_full.jsonc
