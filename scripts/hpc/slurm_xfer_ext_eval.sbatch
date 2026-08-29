#!/bin/bash -l
#SBATCH -J bhc_xferEv
#SBATCH -c 8 -n 1 --gpus 1
#SBATCH -p gpu_2day,gpu_7day
#SBATCH --exclude=dscog035,dscog036,dsdmg002
#SBATCH -t 02:00:00
#SBATCH -o /users/mh00145/bhc_xferEv_%j.out
#SBATCH -e /users/mh00145/bhc_xferEv_%j.err
set -eo pipefail
R=/scratch/mh00145/BanglaHandshape
H=/users/mh00145/BH_home
cp "$H/benchmark/analysis/eval_cross_dataset.py" "$R/benchmark/"
cp "$H/banglahandshape/class_alignment.py"             "$R/banglahandshape/"
cd "$R"; source ~/miniconda3/etc/profile.d/conda.sh; conda activate bdsl_graph
export HF_HOME=/scratch/$USER/hf_cache TORCH_HOME=/scratch/$USER/torch_cache \
       HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONIOENCODING=utf-8

python -u -m benchmark.analysis.eval_cross_dataset \
  --encoder-dir work_dir/bhc_xferext_matrix --epoch 50 --seed 0 \
  --only-sources bdsl47_digits bdsl47_letters bdsl49_recognition rsbdsl38 \
  --lora-rank 8 --lora-alpha 16.0 \
  --lora-targets attn.qkv attn.proj mlp.fc1 mlp.fc2 \
  --output results/T_xdataset_rsbdsl38.md
mkdir -p /users/mh00145/_revision_results
cp results/T_xdataset_rsbdsl38.md /users/mh00145/_revision_results/ 2>/dev/null || true
echo "== cross-dataset matrix DONE =="
