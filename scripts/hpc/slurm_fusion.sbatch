#!/bin/bash -l
# =====================================================================
# WVU DollySods — Tier-2 method: pose⊕RGB fusion on BdSL47 (SI and SD).
# Extracts frozen DINOv2-B features once per (source,split) [cached], then
# trains appearance / pose / fusion heads. Reports per-modality SI & SD Top-1;
# headline = gap_fusion < gap_appearance. Needs work_dir/_kp_cache (A6 extract).
#     sbatch scripts/hpc/slurm_fusion.sbatch
# =====================================================================
#SBATCH -J bhc_fusion
#SBATCH -c 8
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH -p gpu_2day,gpu_7day
#SBATCH -t 08:00:00
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

echo "=== FUSION: signer-independent (SI) ==="
python -u -m path3_handshape_benchmark.train_fusion \
  --config path3_handshape_benchmark/configs/fusion_bdsl47_si.yaml \
  --seeds 0 1 2 --results-csv results/bhc_fusion_si.csv

echo "=== FUSION: signer-dependent (SD) ==="
python -u -m path3_handshape_benchmark.train_fusion \
  --config path3_handshape_benchmark/configs/fusion_bdsl47_sd.yaml \
  --seeds 0 1 2 --results-csv results/bhc_fusion_sd.csv

echo "=== FUSION COMPLETE -> results/bhc_fusion_si.csv, bhc_fusion_sd.csv ==="
