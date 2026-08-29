#!/bin/bash -l
# =====================================================================
# MediaPipe keypoint extraction (A6 + the unblocker for distill/fusion/keypoint
# configs), ONE array task per source, in the isolated mp_kp env. Writes
# work_dir/_kp_cache/<source>.npz. BSLD_45 skipped (~0% hand detection).
#     sbatch scripts/hpc/slurm_keypoints.sbatch
# =====================================================================
#SBATCH -J bhc_kp
#SBATCH -a 0-3
#SBATCH -c 8
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH -p gpu_2day,gpu_7day
#SBATCH --exclude=dscog035,dscog036,dsdmg002
#SBATCH -t 06:00:00
#SBATCH --requeue
#SBATCH -o slurm_logs/%x_%A_%a.out
#SBATCH -e slurm_logs/%x_%A_%a.err

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs work_dir/_kp_cache
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mp_kp
export PYTHONIOENCODING=utf-8

SRCS=(bdsl_mnist bdsl47_digits bdsl47_letters bdsl49_recognition)
S=${SRCS[$SLURM_ARRAY_TASK_ID]}
echo "=== extract keypoints: $S ==="
python -u data_prep/extract_keypoints.py --only "$S"
echo "=== done keypoints: $S ==="
