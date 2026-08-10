#!/bin/bash -l
# =====================================================================
# WVU DollySods — Tier-2 method: test-time adaptation (Tent) on BdSL47 (SI).
# Adapts a trained SI encoder's LayerNorm affine params to the unseen test
# signer via entropy minimization; reports SI Top-1 before vs after. Needs the
# bdsl47_si checkpoints (work_dir/bhc_bdsl47_si). Writes results/bhc_tta.csv.
#     sbatch scripts/hpc/slurm_tta.sbatch
# =====================================================================
#SBATCH -J bhc_tta
#SBATCH -c 8
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH -p gpu_2day,gpu_7day
#SBATCH -t 04:00:00
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
python -c "import torch;print('cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0))"

for seed in 0 1 2; do
  # idempotent: skip a seed already written, or one whose SI checkpoint isn't ready yet
  if [ -f results/bhc_tta.csv ] && grep -q "bhc_tta_after_bdsl47_letters_seed${seed}" results/bhc_tta.csv; then
    echo "[skip] TTA seed$seed already in CSV"; continue
  fi
  if ! ls work_dir/bhc_bdsl47_si/encoder_seed${seed}_epoch*.pt >/dev/null 2>&1; then
    echo "[skip] no bdsl47_si checkpoint for seed$seed yet"; continue
  fi
  python -u -m path3_handshape_benchmark.eval_tta \
    --si-dir work_dir/bhc_bdsl47_si --seed $seed --epoch 0 \
    --sources bdsl47_digits bdsl47_letters \
    --tta-steps 10 --tta-lr 1e-3 --results-csv results/bhc_tta.csv
done
echo "=== TTA DONE -> results/bhc_tta.csv ==="
