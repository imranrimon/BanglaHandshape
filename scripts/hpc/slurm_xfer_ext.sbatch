#!/bin/bash -l
#SBATCH -J bhc_xferE
#SBATCH -a 0-3
#SBATCH -c 8 -n 1 --gpus 1
#SBATCH -p gpu_2day,gpu_7day
#SBATCH --exclude=dscog035,dscog036,dsdmg002
#SBATCH -t 08:00:00
#SBATCH -o /users/mh00145/bhc_xferE_%A_%a.out
#SBATCH -e /users/mh00145/bhc_xferE_%A_%a.err
set -eo pipefail
R=/scratch/mh00145/BanglaHandshape
H=/users/mh00145/BH_home
MATRIX="work_dir/bhc_xferext_matrix"

# stage code (data pre-staged: BdSL47/BDSL49 already on scratch, RSBdSL38 verified)
cp "$H/benchmark/baselines/train_baseline.py"    "$R/benchmark/"
cp "$H/banglahandshape/class_alignment.py"            "$R/banglahandshape/"
test -d "$R/data/RSBdSL38/train/37" || { echo "RSBdSL38 not staged"; exit 1; }

SRC=(bdsl47_digits bdsl47_letters bdsl49_recognition rsbdsl38)
declare -A ROOT=(
  [bdsl47_digits]="data/BdSL47/Bangla Sign Language Dataset - Sign Digits"
  [bdsl47_letters]="data/BdSL47/Bangla Sign Language Dataset - Sign Letters"
  [bdsl49_recognition]="data/bdsl49_extracted/Recognition_1/Recognition_1/train"
  [rsbdsl38]="data/RSBdSL38/train"
)
s=${SRC[$SLURM_ARRAY_TASK_ID]}
root="${ROOT[$s]}"

cd "$R"; source ~/miniconda3/etc/profile.d/conda.sh; conda activate bdsl_graph
export HF_HOME=/scratch/$USER/hf_cache TORCH_HOME=/scratch/$USER/torch_cache \
       HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONIOENCODING=utf-8

cfg="benchmark/configs/_xferext_${s}.yaml"
cat > "$cfg" <<YAML
Experiment_name: bhc_xferext_${s}
sources:
  ${s}: "${root}"
split:
  val_users: [4]
  test_users: [5]
  seed: 0
encoder:
  timm_name: vit_small_patch14_dinov2.lvd142m
  pretrained: True
  lora_rank: 8
  lora_alpha: 16.0
  lora_dropout: 0.05
  lora_targets: [attn.qkv, attn.proj, mlp.fc1, mlp.fc2]
image_size: 224
batch_size: 64
num_workers: 4
num_epoch: 50
base_lr: 0.0005
weight_decay: 0.01
grad_clip: 1.0
save_interval: 50
work_dir: ./work_dir/bhc_xferext_${s}
YAML

python -u -m benchmark.baselines.train_baseline \
  --config "$cfg" --seeds 0 --results-csv results/bhc_xferext_${s}_seed0.csv
mkdir -p "$MATRIX"
cp "work_dir/bhc_xferext_${s}/encoder_seed0_epoch50.pt" \
   "$MATRIX/encoder_${s}_seed0_epoch50.pt"
echo "== xfer encoder $s collected =="
