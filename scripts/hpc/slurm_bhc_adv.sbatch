#!/bin/bash -l
# =====================================================================
# Signer-adversarial LoRA (E3/tab:probe adversary row + tab:mitigate adversary),
# 3 seeds in PARALLEL. NB: adversarial training has its OWN trainer
# (train_adversarial.py, gradient-reversal) — run_campaign/train_baseline would
# silently train a PLAIN model, so it must NOT be used here.
#     sbatch scripts/hpc/slurm_bhc_adv.sbatch
# =====================================================================
#SBATCH -J bhc_adv
#SBATCH -a 0-2
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

SEED=$SLURM_ARRAY_TASK_ID
echo "=== signer-adversarial seed $SEED ==="
python -u -m path3_handshape_benchmark.train_adversarial \
  --config path3_handshape_benchmark/configs/bdsl47_si_adv.yaml \
  --seeds "$SEED" --results-csv "results/bhc_adv_seed${SEED}.csv"
echo "=== done adv seed $SEED ==="
