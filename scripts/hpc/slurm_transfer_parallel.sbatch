#!/bin/bash -l
# =====================================================================
# T4 transfer matrix, PARALLELIZED: one array task per (source, seed) single-source
# encoder (4 sources x 3 seeds = 12), instead of run_transfer_matrix's serial
# _train_and_collect. After this array finishes, submit the eval-only step with a
# dependency to build the N*N matrix:
#     TRAIN=$(sbatch --parsable scripts/hpc/slurm_transfer_parallel.sbatch)
#     sbatch --dependency=afterok:$TRAIN scripts/hpc/slurm_transfer_eval.sbatch
# =====================================================================
#SBATCH -J bhc_xfer_train
#SBATCH -a 0-11
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

python -c "import torch; torch.zeros(1).cuda()" 2>/dev/null || {
  echo "GPU not usable on $(hostname) -> requeue"; scontrol requeue "$SLURM_JOB_ID"; exit 1; }
python -u -m path3_handshape_benchmark.run_transfer_train_one --task "$SLURM_ARRAY_TASK_ID"
echo "=== done transfer-train task $SLURM_ARRAY_TASK_ID ==="
