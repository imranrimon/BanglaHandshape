#!/bin/bash -l
# =====================================================================
# Pose-dependent configs (need work_dir/_kp_cache from slurm_keypoints.sbatch),
# one config per array task, each dispatched to its DEDICATED trainer (NOT
# run_campaign/train_baseline). Regenerates tab:mitigate (distill = C3 core + E5
# controls), tab:fusionfull (fusion), and A6 (keypoint MLP). Fire after keypoints:
#     KP=<keypoint array jobid>
#     sbatch --dependency=afterok:$KP scripts/hpc/slurm_bhc_pose.sbatch
#   idx: 0 distill(ours) 1 distill+LS 2 distill+RGBkd 3 fusion-SI 4 fusion-SD 5 keypoint-MLP
# =====================================================================
#SBATCH -J bhc_pose
#SBATCH -a 0-5
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

CONFIGS=(bdsl47_si_distill bdsl47_si_distill_ls bdsl47_si_distill_rgbkd fusion_bdsl47_si fusion_bdsl47_sd keypoint_mlp)
TRAINERS=(train_distill    train_distill       train_distill          train_fusion     train_fusion     train_keypoint)
i=$SLURM_ARRAY_TASK_ID
C=${CONFIGS[$i]}; T=${TRAINERS[$i]}
echo "=== $T on $C (3 seeds) ==="
python -u -m "path3_handshape_benchmark.$T" \
  --config "path3_handshape_benchmark/configs/${C}.yaml" \
  --seeds 0 1 2 --results-csv "results/bhc_${C}.csv"
echo "=== done $C ==="
