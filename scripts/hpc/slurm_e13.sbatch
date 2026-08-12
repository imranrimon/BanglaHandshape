#!/bin/bash -l
#SBATCH -J bhc_e13
#SBATCH -a 0-2
#SBATCH -c 8 -n 1 --gpus 1
#SBATCH -p gpu_2day,gpu_7day
#SBATCH --exclude=dscog035,dscog036,dsdmg002
#SBATCH -t 06:00:00
#SBATCH -o /users/mh00145/bhc_e13_%A_%a.out
#SBATCH -e /users/mh00145/bhc_e13_%A_%a.err
set -eo pipefail
R=/scratch/mh00145/BanglaHandshape
H=/users/mh00145/BH_home

# --- stage code (compute node can write scratch; login node cannot) ---
cp "$H/path3_handshape_benchmark/train_baseline.py"          "$R/path3_handshape_benchmark/"
cp "$H/path3_handshape_benchmark/configs/rsbdsl38_baseline.yaml" "$R/path3_handshape_benchmark/configs/"
cp "$H/bangla_handshape/class_alignment.py"                   "$R/bangla_handshape/"

# --- data pre-staged + verified by the bhc_stage job (single writer, no race) ---
test -d "$R/data/RSBdSL38/train/37" || { echo "RSBdSL38 not staged on scratch"; exit 1; }

cd "$R"; source ~/miniconda3/etc/profile.d/conda.sh; conda activate bdsl_graph
export HF_HOME=/scratch/$USER/hf_cache TORCH_HOME=/scratch/$USER/torch_cache \
       HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONIOENCODING=utf-8
s=$SLURM_ARRAY_TASK_ID
python -u -m path3_handshape_benchmark.train_baseline \
  --config path3_handshape_benchmark/configs/rsbdsl38_baseline.yaml \
  --seeds $s --results-csv results/bhc_rsbdsl38_ext_seed${s}.csv
mkdir -p /users/mh00145/_revision_results
cp results/bhc_rsbdsl38_ext_seed${s}.csv /users/mh00145/_revision_results/ 2>/dev/null || true
echo "== E13 rsbdsl38 seed$s DONE =="
