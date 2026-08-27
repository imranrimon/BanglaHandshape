#!/bin/bash -l
# =====================================================================
# WVU DollySods — Tier-3 Leave-One-Dataset-Out (LODO) cross-corpus benchmark.
# GATED on a VERIFIED alignment at alignment/handshape_alignment.json (draft it
# with propose_alignment.py, then have a signer/linguist verify — see
# alignment/README.md). Trains one unified-space DINOv2+LoRA per held-out source.
#     sbatch scripts/hpc/slurm_lodo.sbatch
# =====================================================================
#SBATCH -J bhc_lodo
#SBATCH -c 8
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH -p gpu_2day,gpu_7day
#SBATCH --exclude=dscog035,dscog036   # dscog035 idle-but-fails; dscog036 down
#SBATCH -t 24:00:00
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

ALIGN=alignment/handshape_alignment.json
test -s "$ALIGN" || { echo "[abort] $ALIGN missing. Draft with propose_alignment.py, then verify (alignment/README.md)."; exit 1; }
python -c "import json,sys; a=json.load(open('$ALIGN')); sys.exit(0 if a.get('verified') else 'ALIGNMENT NOT VERIFIED — set verified:true after expert review before reporting LODO')"

python -u -m path3_handshape_benchmark.run_lodo \
  --alignment-json "$ALIGN" --seeds 0 1 2 \
  --results-csv results/bhc_lodo.csv --work-dir work_dir/bhc_lodo

echo "=== LODO COMPLETE -> results/bhc_lodo.csv + work_dir/bhc_lodo/coverage.json ==="
