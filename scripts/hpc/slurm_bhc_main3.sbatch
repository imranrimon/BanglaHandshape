#!/bin/bash -l
# =====================================================================
# The 3 tab:main/tab:ablation rows missing from the fixed-data reproduction:
#   [0] cnn_resnet50_imagenet (failed earlier on an offline torchvision download;
#       weight now pre-staged in $HOME/torch_cache), [1] probe_aimv2_l, [2] probe_dinov2reg_b.
# Backbones pre-cached to $HOME/hf_cache. Each = 3 seeds via run_campaign.
#     sbatch scripts/hpc/slurm_bhc_main3.sbatch
# =====================================================================
#SBATCH -J bhc_main3
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

CONFIGS=(cnn_resnet50_imagenet probe_aimv2_l probe_dinov2reg_b)
CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
export CAMPAIGN_CSV="results/bhc_${CONFIG}.csv"
echo "=== $CONFIG -> $CAMPAIGN_CSV ==="
python -u scripts/run_campaign.py "$CONFIG"
echo "=== done $CONFIG ==="
