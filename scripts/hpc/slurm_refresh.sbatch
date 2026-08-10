#!/bin/bash -l
# =====================================================================
# WVU DollySods — backbone refresh (A1/A3): DINOv2-with-registers-B + AIMv2-L
# linear probes. Frozen -> feature-cache path (fast). Weights pre-warmed in the
# HF cache. Writes results/bhc_refresh.csv.
#     sbatch scripts/hpc/slurm_refresh.sbatch
# =====================================================================
#SBATCH -J bhc_refresh
#SBATCH -c 8
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH -p gpu_2day,gpu_7day
#SBATCH -t 08:00:00
#SBATCH -o slurm_logs/%x_%j.out
#SBATCH -e slurm_logs/%x_%j.err

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs results
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bdsl_graph
export PYTHONIOENCODING=utf-8
export CAMPAIGN_NW=8
export HF_HOME=/scratch/$USER/hf_cache
export TORCH_HOME=/scratch/$USER/torch_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CAMPAIGN_CSV="results/bhc_refresh.csv"
python -c "import torch;print('cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0))"

python -u scripts/run_campaign.py probe_dinov2reg_b probe_aimv2_l
echo "=== REFRESH DONE -> results/bhc_refresh.csv ==="
