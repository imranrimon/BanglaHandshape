#!/bin/bash -l
#SBATCH -J bhc_rsosigner
#SBATCH -a 0-2
#SBATCH -c 8
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH -p gpu_2day,gpu_7day
#SBATCH --exclude=dscog035,dscog036,dsdmg002
#SBATCH -t 12:00:00
#SBATCH --requeue
#SBATCH -o slurm_logs/%x_%A_%a.out
#SBATCH -e slurm_logs/%x_%A_%a.err
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p slurm_logs results
source ~/miniconda3/etc/profile.d/conda.sh; conda activate bdsl_graph
export PYTHONIOENCODING=utf-8 HF_HOME="$HOME/hf_cache" TORCH_HOME="$HOME/torch_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CAMPAIGN_NW=4
python -c "import torch; torch.zeros(1).cuda()" 2>/dev/null || { echo "GPU dead -> requeue"; scontrol requeue "$SLURM_JOB_ID"; exit 1; }
SEED=$SLURM_ARRAY_TASK_ID
export CAMPAIGN_SEEDS="$SEED" CAMPAIGN_CSV="results/bhc_rsbdsl38_osigner_seed${SEED}.csv"
python -u scripts/run_campaign.py rsbdsl38_osigner
