#!/bin/bash -l
# =====================================================================
# WVU DollySods — Phase-1 journal metrics for the BdSL47 SD vs SI encoders.
# Computes imbalance-aware (macro-F1, balanced accuracy, per-class recall) and
# calibration (ECE/MCE/Brier/NLL) metrics, plus a temperature-scaling
# recalibration, from the already-trained encoders. Writes:
#   results/T_metrics.md          imbalance-aware classification metrics
#   results/T_recalibration.md    ECE/NLL before->after temp scaling + fitted T
#     sbatch scripts/hpc/slurm_metrics.sbatch
# =====================================================================
#SBATCH -J bhc_metrics
#SBATCH -c 8
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH -p gpu_2day,gpu_7day
#SBATCH --exclude=dscog035,dscog036   # dscog035 idle-but-fails; dscog036 down
#SBATCH -t 03:00:00
#SBATCH -o slurm_logs/%x_%j.out
#SBATCH -e slurm_logs/%x_%j.err

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs results
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bdsl_graph
export PYTHONIOENCODING=utf-8
export HF_HOME=/scratch/$USER/hf_cache
export TORCH_HOME=/scratch/$USER/torch_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python -c "import torch;print('cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0))"

echo "===== T_metrics + T_recalibration: BdSL47 SD vs SI ====="
python -u -m benchmark.analysis.eval_metrics --seed 0

echo "===== METRICS DONE ====="
ls -la results/T_metrics.md results/T_recalibration.md 2>/dev/null
