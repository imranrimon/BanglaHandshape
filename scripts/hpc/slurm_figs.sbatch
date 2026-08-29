#!/bin/bash -l
#SBATCH -J bhc_figs
#SBATCH -a 0-1
#SBATCH -c 8
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH -p gpu_2day,gpu_7day
#SBATCH --exclude=dscog035,dscog036,dsdmg002
#SBATCH -t 03:00:00
#SBATCH --requeue
#SBATCH -o slurm_logs/%x_%A_%a.out
#SBATCH -e slurm_logs/%x_%A_%a.err
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p slurm_logs paper/figs
source ~/miniconda3/etc/profile.d/conda.sh; conda activate bdsl_graph
export PYTHONIOENCODING=utf-8 HF_HOME="$HOME/hf_cache" TORCH_HOME="$HOME/torch_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -c "import torch; torch.zeros(1).cuda()" 2>/dev/null || { echo "GPU dead -> requeue"; scontrol requeue "$SLURM_JOB_ID"; exit 1; }
if [ "$SLURM_ARRAY_TASK_ID" = "0" ]; then
  echo "=== SF2 gradcam ==="
  python -u -m benchmark.figures.plot_gradcam --si-dir work_dir/bhc_bdsl47_si --sd-dir work_dir/bhc_bdsl47_sd \
    --source bdsl47_letters --seed 0 --epoch 50 --lora-rank 8 --lora-targets attn.qkv attn.proj mlp.fc1 mlp.fc2 \
    --output paper/figs/SF2_gradcam.png
else
  echo "=== SF3 per-class gap ==="
  python -u -m benchmark.figures.plot_perclass_gap --output paper/figs/SF3_perclass
fi
