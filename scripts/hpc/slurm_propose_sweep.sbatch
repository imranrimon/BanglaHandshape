#!/bin/bash -l
# =====================================================================
# WVU DollySods — E16 step 1 (threshold SWEEP): draft the unified handshape
# alignment at several cosine thresholds IN PARALLEL, so we can pick the one
# whose cross-source coverage (groups spanning >=2 sources) is best before a
# linguist verifies it. Supersedes the single-threshold slurm_propose.sbatch
# now that 4 sources are on disk (BdSL-MNIST + BdSL47 re-fetched 2026-08-12;
# BSLD_45 still unobtainable). Each array task = one threshold, own output.
#     sbatch scripts/hpc/slurm_propose_sweep.sbatch
# Then: build_taxonomy_workbook off whichever review_<T>.csv has best coverage.
# =====================================================================
#SBATCH -J bhc_propose_sweep
#SBATCH -c 8
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH -p gpu_2day,gpu_7day
#SBATCH --exclude=dscog035,dscog036
#SBATCH -t 04:00:00
#SBATCH -o slurm_logs/%x_%A_%a.out
#SBATCH -e slurm_logs/%x_%A_%a.err
#SBATCH --array=0-2

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs alignment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bdsl_graph
export PYTHONIOENCODING=utf-8
# home-based caches (visible on every node; pre-populated on the login node)
export HF_HOME="$HOME/hf_cache"
export TORCH_HOME="$HOME/torch_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

THRS=(0.90 0.88 0.86)
T=${THRS[$SLURM_ARRAY_TASK_ID]}

python -c "import torch;print('cuda',torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
echo "=== proposing alignment at sim-threshold $T (all on-disk sources) ==="

python -u -m path3_handshape_benchmark.propose_alignment \
  --out "alignment/handshape_alignment.proposed_${T}.json" \
  --review-csv "alignment/handshape_alignment_review_${T}.csv" \
  --timm-name vit_base_patch14_dinov2.lvd142m \
  --per-class 60 --sim-threshold "$T"

echo "=== DONE threshold $T -> alignment/handshape_alignment.proposed_${T}.json + review_${T}.csv ==="
