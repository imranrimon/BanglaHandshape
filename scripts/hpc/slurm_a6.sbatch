#!/bin/bash -l
# =====================================================================
# WVU DollySods — A6 (modality: keypoint vs appearance).
# Two-env job: (1) extract MediaPipe hand keypoints in the isolated mp_kp env
# (Tasks API, offline once .kp_models/hand_landmarker.task is local), then
# (2) train the keypoint MLP + LOUO (keypoint & appearance) in bdsl_graph.
# Prereq (login node, has internet): scripts/_a6_prep.sh created the mp_kp env
# and downloaded the landmarker model.
#
#     sbatch scripts/hpc/slurm_a6.sbatch
# =====================================================================
#SBATCH -J bhc_a6
#SBATCH -c 8
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH -p gpu_2day,gpu_7day
#SBATCH -t 12:00:00
#SBATCH -o slurm_logs/%x_%j.out
#SBATCH -e slurm_logs/%x_%j.err

set -eo pipefail    # NB: no -u — conda's MKL activate.d script uses unbound vars
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs results work_dir/_kp_cache
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONIOENCODING=utf-8

echo "############## [1/2] EXTRACT KEYPOINTS (mp_kp env) ##############"
conda activate mp_kp
python -c "import mediapipe as mp; print('mediapipe', mp.__version__)"
test -s .kp_models/hand_landmarker.task || { echo "[abort] .kp_models/hand_landmarker.task missing — run scripts/_a6_prep.sh on the login node first"; exit 1; }
# BdSL47 digits+letters = the SI pose-vs-appearance comparison (identical images
# and user-disjoint split as bdsl47_si). BSLD_45 excluded (drawn-overlay, 0% det).
python preprocessing/extract_keypoints.py --only bdsl47_digits bdsl47_letters
conda deactivate

echo "############## [2/2] TRAIN (bdsl_graph env) ##############"
conda activate bdsl_graph
# LOUO-appearance re-forwards DINOv2-B on frozen features -> needs the warmed
# cache; compute nodes have no outbound HTTPS.
export HF_HOME=/scratch/$USER/hf_cache
export TORCH_HOME=/scratch/$USER/torch_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python -c "import torch;print('cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0))"

echo "=== A6: keypoint MLP (SI, 3 seeds) ==="
python -m path3_handshape_benchmark.train_keypoint \
  --config path3_handshape_benchmark/configs/keypoint_mlp.yaml \
  --seeds 0 1 2 --results-csv results/bhc_keypoint_mlp.csv

echo "=== A6: LOUO keypoint (pose, 10 signers x 3 seeds) ==="
python -m path3_handshape_benchmark.louo_keypoint \
  --seeds 0 1 2 --results-csv results/bhc_louo_keypoint.csv

echo "=== A6: LOUO appearance (frozen DINOv2-B, symmetric) ==="
python -m path3_handshape_benchmark.louo_appearance \
  --seeds 0 1 2 --results-csv results/bhc_louo_appearance.csv

echo "=== A6 COMPLETE -> results/bhc_keypoint_mlp.csv, bhc_louo_keypoint.csv, bhc_louo_appearance.csv ==="
