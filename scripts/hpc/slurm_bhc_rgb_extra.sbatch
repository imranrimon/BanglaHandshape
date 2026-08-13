#!/bin/bash -l
# =====================================================================
# Keypoint-FREE pending configs the T1 array didn't cover, fired now in parallel:
#   [0] bdsl47_si_adv     -> tab:mitigate adversary row + tab:probe (E3) adversary
#   [1] rsbdsl38_baseline -> E13 external RGB baseline
# Both are DINOv2-S+LoRA (cached), no MediaPipe dependency.
#     sbatch scripts/hpc/slurm_bhc_rgb_extra.sbatch
# =====================================================================
#SBATCH -J bhc_rgbx
#SBATCH -a 0-1
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
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs results
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bdsl_graph
export PYTHONIOENCODING=utf-8
export HF_HOME="$HOME/hf_cache"
export TORCH_HOME="$HOME/torch_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CAMPAIGN_NW=4

python -c "import torch; torch.zeros(1).cuda()" 2>/dev/null || {
  echo "GPU not usable on $(hostname) -> requeue"; scontrol requeue "$SLURM_JOB_ID"; exit 1; }

CONFIGS=(bdsl47_si_adv rsbdsl38_baseline)
CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
export CAMPAIGN_CSV="results/bhc_${CONFIG}.csv"
echo "=== $CONFIG -> $CAMPAIGN_CSV ==="
python -u scripts/run_campaign.py "$CONFIG"
echo "=== done $CONFIG ==="
