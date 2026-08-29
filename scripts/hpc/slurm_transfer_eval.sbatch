#!/bin/bash -l
# =====================================================================
# T4 eval-only: build the N*N cross-dataset matrix from the encoders collected by
# the parallel training array (slurm_transfer_parallel.sbatch). Submit with a
# dependency so it runs after all encoders exist:
#     sbatch --dependency=afterok:<TRAIN_ARRAY_JOBID> scripts/hpc/slurm_transfer_eval.sbatch
# =====================================================================
#SBATCH -J bhc_xfer_eval
#SBATCH -c 8
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH -p gpu_2day,gpu_7day
#SBATCH --exclude=dscog035,dscog036
#SBATCH -t 02:00:00
#SBATCH -o slurm_logs/%x_%j.out
#SBATCH -e slurm_logs/%x_%j.err

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

python -u -m benchmark.analysis.run_transfer_matrix \
    --config benchmark/configs/transfer_matrix.yaml --eval-only
echo "=== T4 matrix built -> results/T4_transfer_matrix.md ==="
