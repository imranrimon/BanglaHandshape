#!/bin/bash -l
# =====================================================================
# WVU DollySods — Tier-3 step 1: DRAFT the unified handshape alignment.
# DINOv2-B class prototypes -> cross-source cosine clustering -> proposed JSON +
# human-review CSV. The draft then goes to a signer/linguist to VERIFY before
# run_lodo uses it (alignment/README.md). Tune --sim-threshold (0.90-0.95) for
# finer groups; DINOv2 hand-crop prototypes sit at high cosine.
#     sbatch scripts/hpc/slurm_propose.sbatch
# =====================================================================
#SBATCH -J bhc_propose
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
mkdir -p slurm_logs alignment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bdsl_graph
export PYTHONIOENCODING=utf-8
export HF_HOME=/scratch/$USER/hf_cache
export TORCH_HOME=/scratch/$USER/torch_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python -c "import torch;print('cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0))"

python -u -m path3_handshape_benchmark.propose_alignment \
  --out alignment/handshape_alignment.proposed.json \
  --review-csv alignment/handshape_alignment_review.csv \
  --timm-name vit_base_patch14_dinov2.lvd142m \
  --per-class 60 --sim-threshold 0.92

echo "=== PROPOSE DONE -> alignment/handshape_alignment.proposed.json + review CSV ==="
echo "Next: a signer/linguist verifies -> save as handshape_alignment.json, set verified:true -> sbatch slurm_lodo.sbatch"
