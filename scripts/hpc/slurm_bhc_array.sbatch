#!/bin/bash -l
# =====================================================================
# WVU DollySods — BanglaHandshape sister-paper campaign as a JOB ARRAY.
# One config per array task (each runs its 3 seeds via run_campaign.py), all
# in PARALLEL (one GPU each) -> the ~7-day local serial matrix finishes in
# <~1 day (critical path = full_ft's 3 seeds).
#
# Submit from ~/BanglaHandshape:
#     sbatch scripts/hpc/slurm_bhc_array.sbatch
# Pick a partition first:
#     sinfo -s            # partitions + idle counts
# The models are small (batch 64, ViT-S / ResNet / DINOv2-B); a 24GB
# dscog[005-030] node is plenty. Data must be present under data/.
# =====================================================================
#SBATCH -J bhc_array
#SBATCH -a 0-18                     # 19 configs (indices match CONFIGS below)
#SBATCH -c 8
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH -p <PARTITION>             # <-- fill from `sinfo -s`
##SBATCH -a 0-18%10                # <-- alt: cap to 10 concurrent tasks if busy
#SBATCH -t 72:00:00               # generous; full_ft x3 seeds is the long pole
#SBATCH -o slurm_logs/%x_%A_%a.out
#SBATCH -e slurm_logs/%x_%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs results

# conda (miniconda in $HOME; must be sourced in a batch shell)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bdsl_graph
export PYTHONIOENCODING=utf-8          # DollySods locale is often ASCII
export CAMPAIGN_NW=8                    # data on scratch -> workers OK; set 0 if on NFS and it hangs

# Index -> config. run_campaign.py runs each config for 3 seeds (skip-existing).
CONFIGS=( \
  linear_probe probe_imagenet_vit_s probe_dinov2_b probe_dinov2_l probe_siglip_b probe_mae_b \
  lora lora_r4 lora_r16 lora_r32 lora_attn_only full_ft \
  cnn_resnet18_scratch cnn_resnet50_imagenet \
  kshot_k5 kshot_k10 kshot_k20 \
  bdsl47_si bdsl47_sd )

CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
# Per-task CSV: concurrent appends from many array tasks to ONE shared CSV would
# interleave/corrupt rows. Each task writes its own; merge afterwards (see runbook).
export CAMPAIGN_CSV="results/bhc_${CONFIG}.csv"

echo "=== task $SLURM_ARRAY_TASK_ID  config=$CONFIG  csv=$CAMPAIGN_CSV ==="
python -c "import torch;print('cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0))"

# Full-budget (>=50 epochs), 3 seeds, skip-existing, probe feature-caching,
# per-seed retry. Idempotent: re-submitting after a timeout resumes cleanly.
python -u scripts/run_campaign.py "$CONFIG"
echo "=== done: $CONFIG ==="
