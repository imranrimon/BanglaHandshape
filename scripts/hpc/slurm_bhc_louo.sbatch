#!/bin/bash -l
# =====================================================================
# Leave-one-user-out per-signer robustness (tab:louo). Two runners as ONE array:
#   [0] louo_appearance  (DINOv2-B frozen features)  -- no keypoint dep
#   [1] louo_keypoint    (cached keypoints + tiny MLP) -- needs work_dir/_kp_cache
# NB: folds are CHEAP (features/keypoints extracted ONCE, then linear/MLP heads),
# so each runner is a single job — a per-fold array would re-extract 10x and be
# SLOWER. Submit appearance now, keypoint gated on the keypoint extraction:
#     sbatch --array=0 scripts/hpc/slurm_bhc_louo.sbatch
#     sbatch --array=1 --dependency=afterok:<KP_JOBID> scripts/hpc/slurm_bhc_louo.sbatch
# =====================================================================
#SBATCH -J bhc_louo
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

RUNNERS=(louo_appearance louo_keypoint)
R=${RUNNERS[$SLURM_ARRAY_TASK_ID]}
echo "=== $R (LOUO, 3 seeds) ==="
python -u -m "benchmark.$R" --seeds 0 1 2 --results-csv "results/bhc_${R}.csv"
echo "=== done $R ==="
