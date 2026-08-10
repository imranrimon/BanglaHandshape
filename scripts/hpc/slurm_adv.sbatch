#!/bin/bash -l
# =====================================================================
# WVU DollySods — Tier-2 method: signer-ADVERSARIAL LoRA on BdSL47 (SI).
# Compare bhc_bdsl47_si_adv_* vs bhc_bdsl47_si_* (plain) — does closing the
# signer shortcut raise SI accuracy? Writes results/bhc_adv.csv.
#     sbatch scripts/hpc/slurm_adv.sbatch
# =====================================================================
#SBATCH -J bhc_adv
#SBATCH -c 8
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH -p gpu_2day,gpu_7day
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

python -u -m path3_handshape_benchmark.train_adversarial \
  --config path3_handshape_benchmark/configs/bdsl47_si_adv.yaml \
  --seeds 0 1 2 --results-csv results/bhc_adv.csv
echo "=== ADVERSARIAL DONE -> results/bhc_adv.csv ==="
