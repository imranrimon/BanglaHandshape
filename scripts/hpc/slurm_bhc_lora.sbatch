#!/bin/bash -l
# =====================================================================
# Focused re-generation of the headline bhc_lora encoder (DINOv2-S + LoRA)
# after the 2026-08 data re-fetch. 3 seeds in PARALLEL (array over seed).
# Produces work_dir/bhc_lora/encoder_seed<N>_epoch<E>.pt -> unblocks:
#   - SF1 confusion matrix (plot_confusion), - the E15 similarity<->confusability
#   correlation, - the T4 transfer-matrix base. Uses the login-node-warmed
#   $HOME/hf_cache (DINOv2-S/B), so it runs offline on the compute nodes.
#     sbatch scripts/hpc/slurm_bhc_lora.sbatch
# =====================================================================
#SBATCH -J bhc_lora
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
export CAMPAIGN_NW=4                 # data on NFS home; keep modest

SEED=$SLURM_ARRAY_TASK_ID
export CAMPAIGN_SEEDS="$SEED"
export CAMPAIGN_CSV="results/bhc_lora_seed${SEED}.csv"

# flaky-node guard: a transient "CUDA unknown error" here should REQUEUE to another
# node, not hard-fail the seed (bit us on dsdmg002). --requeue makes this safe.
python -c "import torch; torch.zeros(1).cuda()" 2>/dev/null || {
  echo "GPU not usable on $(hostname) -> requeue"; scontrol requeue "$SLURM_JOB_ID"; exit 1; }
echo "=== bhc_lora seed $SEED -> $CAMPAIGN_CSV ==="
python -u scripts/run_campaign.py lora
echo "=== done bhc_lora seed $SEED ==="
