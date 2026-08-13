#!/bin/bash -l
# =====================================================================
# WVU DollySods — T4 cross-dataset transfer matrix (RQ3).
# Trains one SINGLE-SOURCE LoRA encoder per source (5 sources × 3 seeds), then
# builds the N×N transfer matrix. Heavier than the main sweep, so it lives in
# its own job. Idempotent/resumable — if it hits the wall-time, just resubmit
# and it resumes from collected checkpoints.
#
# Run scripts/hpc/slurm_sweep.sbatch (the spine) first. Submit from repo root:
#     sbatch scripts/hpc/slurm_transfer.sbatch
# Fill the -p line from `sinfo -s`. Edit seeds/eval_seed in
# path3_handshape_benchmark/configs/transfer_matrix.yaml if you want fewer seeds.
# =====================================================================
#SBATCH -J bhc_transfer
#SBATCH -c 8
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH -p gpu_7day                 # matches the array; 48h walltime fits
#SBATCH --exclude=dscog035,dscog036   # dscog035 idle-but-fails; dscog036 down
#SBATCH -t 48:00:00
#SBATCH -o slurm_logs/%x_%j.out
#SBATCH -e slurm_logs/%x_%j.err

set -eo pipefail    # NB: no -u — conda's MKL activate.d script uses unbound vars
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs results
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bdsl_graph
export PYTHONIOENCODING=utf-8
export HF_HOME="$HOME/hf_cache"                # reuse the warmed weight cache (BH is on home, not scratch)
export TORCH_HOME="$HOME/torch_cache"
export HF_HUB_OFFLINE=1                        # compute nodes have NO outbound HTTPS
export TRANSFORMERS_OFFLINE=1

python - <<'PYEOF'
import sys
from bangla_handshape.class_alignment import discover_default
srcs = discover_default(repo_root=".")
print("sources:", [(s.name, s.num_classes) for s in srcs])
if len(srcs) < 2:
    sys.exit("[abort] need >=2 sources on disk — run scripts/refetch_data.sh first")
PYEOF

python -u -m path3_handshape_benchmark.run_transfer_matrix \
    --config path3_handshape_benchmark/configs/transfer_matrix.yaml

echo "T4 transfer matrix done -> results/T4_transfer_matrix.md"
