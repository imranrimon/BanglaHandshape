#!/bin/bash -l
# =====================================================================
# WVU DollySods — Phase-2 METHOD: pose->RGB distillation on BdSL47 (SI).
# Trains a keypoint-MLP teacher on the SI TRAIN split (frozen), distills its
# signer-agnostic soft logits into a DINOv2-S+LoRA appearance student. At test
# the student uses IMAGES ONLY. Headline: distilled SI Top-1 > plain LoRA SI
# (digits 94.4 / letters 86.7). Needs work_dir/_kp_cache (A6 extract).
#     sbatch scripts/hpc/slurm_distill.sbatch
# =====================================================================
#SBATCH -J bhc_distill
#SBATCH -c 8
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH -p gpu_2day,gpu_7day
#SBATCH --exclude=dscog035,dscog036   # dscog035 idle-but-fails; dscog036 down
#SBATCH -t 12:00:00
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

for f in bdsl47_digits bdsl47_letters; do
  test -s "work_dir/_kp_cache/${f}.npz" || { echo "[abort] missing keypoint cache work_dir/_kp_cache/${f}.npz — run the A6 extract first"; exit 1; }
done

echo "=== DISTILL: signer-independent (SI), pose -> RGB ==="
python -u -m path3_handshape_benchmark.train_distill \
  --config path3_handshape_benchmark/configs/bdsl47_si_distill.yaml \
  --seeds 0 1 2 --results-csv results/bhc_distill.csv

echo "=== DISTILL COMPLETE -> results/bhc_distill.csv ==="
