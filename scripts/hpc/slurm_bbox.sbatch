#!/bin/bash -l
# =====================================================================
# MediaPipe hand-bbox extraction (work_dir/_bbox_cache/<source>.npz) for the
# hand-crop ablation (tab:crop). One array task per BdSL47 source, mp_kp env.
#     sbatch scripts/hpc/slurm_bbox.sbatch
# Then: hand-crop configs (slurm_bhc_handcrop.sbatch) consume the cache.
# =====================================================================
#SBATCH -J bhc_bbox
#SBATCH -a 0-1
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
mkdir -p slurm_logs work_dir/_bbox_cache
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mp_kp
export PYTHONIOENCODING=utf-8

SRCS=(bdsl47_digits bdsl47_letters)
S=${SRCS[$SLURM_ARRAY_TASK_ID]}
echo "=== extract hand bbox: $S ==="
python -u data_prep/extract_hand_bbox.py --only "$S"
echo "=== done bbox: $S ==="
