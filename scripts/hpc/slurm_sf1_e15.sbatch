#!/bin/bash -l
# =====================================================================
# SF1 confusion matrix (BdSL47-Letters) from the regenerated bhc_lora encoder,
# then the E15 similarity<->confusability correlation from that matrix. One GPU
# job (both steps fast); named bhc_* so the commit waiter includes it.
#     sbatch scripts/hpc/slurm_sf1_e15.sbatch
# =====================================================================
#SBATCH -J bhc_sf1e15
#SBATCH -c 8
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH -p gpu_2day,gpu_7day
#SBATCH --exclude=dscog035,dscog036,dsdmg002
#SBATCH -t 02:00:00
#SBATCH --requeue
#SBATCH -o slurm_logs/%x_%j.out
#SBATCH -e slurm_logs/%x_%j.err

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs results results/E15
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bdsl_graph
export PYTHONIOENCODING=utf-8
export HF_HOME="$HOME/hf_cache"
export TORCH_HOME="$HOME/torch_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -c "import torch; torch.zeros(1).cuda()" 2>/dev/null || {
  echo "GPU not usable on $(hostname) -> requeue"; scontrol requeue "$SLURM_JOB_ID"; exit 1; }

LT=(attn.qkv attn.proj mlp.fc1 mlp.fc2)

echo "=== SF1: BdSL47-Letters confusion (bhc_lora seed 0, epoch 50) ==="
python -u -m path3_handshape_benchmark.plot_confusion \
  --source bdsl47_letters --encoder-dir work_dir/bhc_lora --seed 0 --epoch 50 \
  --timm-name vit_small_patch14_dinov2.lvd142m --lora-rank 8 --lora-targets "${LT[@]}" \
  --output results/SF1_bdsl47_letters.png
cp -f results/SF1_bdsl47_letters.png paper/figs/SF1_confusion.png

echo "=== E15: similarity <-> confusability correlation ==="
python -u tools/handshape_similarity_analysis.py \
  --source bdsl47_letters \
  --feature-csv docs/E15_handshape_annotation/handshape_feature_coding_sheet.csv \
  --confusion-npz results/SF1_bdsl47_letters_matrix.npy \
  --out results/E15

echo "=== SF1+E15 DONE ==="
