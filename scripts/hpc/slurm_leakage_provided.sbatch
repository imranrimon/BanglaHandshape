#!/bin/bash -l
#SBATCH -J bhc_leakp
#SBATCH -a 0-1
#SBATCH -c 8 -n 1 --gpus 1
#SBATCH -p gpu_2day,gpu_7day
#SBATCH --exclude=dscog035,dscog036,dsdmg002
#SBATCH -t 02:00:00
#SBATCH -o /users/mh00145/bhc_leakp_%A_%a.out
#SBATCH -e /users/mh00145/bhc_leakp_%A_%a.err
set -eo pipefail
R=/scratch/mh00145/BanglaHandshape
H=/users/mh00145/BH_home

SRC=(rsbdsl38 bdsl49_recognition)
src=${SRC[$SLURM_ARRAY_TASK_ID]}

# --- stage code ---
cp "$H/path3_handshape_benchmark/eval_leakage.py"          "$R/path3_handshape_benchmark/"
cp "$H/path3_handshape_benchmark/eval_leakage_provided.py" "$R/path3_handshape_benchmark/"
cp "$H/bangla_handshape/class_alignment.py"                "$R/bangla_handshape/"

# --- stage this task's data into the scratch repo (compute node can write scratch) ---
if [ "$src" = "rsbdsl38" ]; then
  mkdir -p "$R/data/RSBdSL38"
  rsync -a --ignore-existing "$H/data/RSBdSL38/" "$R/data/RSBdSL38/"
else
  mkdir -p "$R/data/bdsl49_extracted"
  rsync -a --ignore-existing "$H/data/bdsl49_extracted/" "$R/data/bdsl49_extracted/"
fi

cd "$R"; source ~/miniconda3/etc/profile.d/conda.sh; conda activate bdsl_graph
export HF_HOME=/scratch/$USER/hf_cache TORCH_HOME=/scratch/$USER/torch_cache \
       HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONIOENCODING=utf-8

python -u -m path3_handshape_benchmark.eval_leakage_provided --sources "$src"

mkdir -p /users/mh00145/_revision_results
cp "results/T_leakage_provided_${src}.md" /users/mh00145/_revision_results/ 2>/dev/null || true
echo "== leakage-provided $src DONE =="
