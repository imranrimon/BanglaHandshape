#!/bin/bash -l
# =====================================================================
# WVU DollySods — Phase-2 robustness / OOD table for the SI BdSL47 encoder.
#   T_robustness.md   corruption Top-1 grid (rows=corruption, cols=severities)
#   SF6_robustness.png  Top-1 vs severity, one subplot per source
# Loads work_dir/bhc_bdsl47_si (auto latest epoch, seed 0), fits ONE clean head,
# then re-featurizes the SI-val set through 6 corruptions x 3 severities and
# predicts with the FIXED clean head. Graceful no-op (exit 0) if no checkpoint.
#     sbatch scripts/hpc/slurm_robustness.sbatch
# =====================================================================
#SBATCH -J bhc_robustness
#SBATCH -c 8
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH -p gpu_2day,gpu_7day
#SBATCH --exclude=dscog035,dscog036   # dscog035 idle-but-fails; dscog036 down
#SBATCH -t 04:00:00
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

echo "===== T_robustness + SF6: corruption robustness (SI BdSL47) ====="
python -u -m benchmark.analysis.eval_robustness \
  --si-dir work_dir/bhc_bdsl47_si --seed 0 \
  --sources bdsl47_digits bdsl47_letters || echo "[warn] robustness eval failed"

echo "===== ROBUSTNESS DONE ====="
ls -la results/T_robustness.md results/SF6_robustness.png 2>/dev/null
