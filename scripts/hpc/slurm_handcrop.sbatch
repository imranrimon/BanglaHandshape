#!/bin/bash -l
# =====================================================================
# WVU DollySods — Tier-1 hand-crop mechanism ablation.
# (1) extract MediaPipe hand bboxes (mp_kp env) for BdSL47 digits+letters, then
# (2) train SI & SD LoRA on HAND-CROPPED images (bdsl_graph). If the SD-SI gap
# collapses under cropping, the signer shortcut was background-driven.
#     sbatch scripts/hpc/slurm_handcrop.sbatch
# =====================================================================
#SBATCH -J bhc_handcrop
#SBATCH -c 8
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH -p gpu_2day,gpu_7day
#SBATCH -t 12:00:00
#SBATCH -o slurm_logs/%x_%j.out
#SBATCH -e slurm_logs/%x_%j.err

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs results work_dir/_bbox_cache
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONIOENCODING=utf-8

echo "############## [1/2] EXTRACT HAND BBOXES (mp_kp env) ##############"
conda activate mp_kp
test -s .kp_models/hand_landmarker.task || { echo "[abort] .kp_models/hand_landmarker.task missing — run scripts/_a6_prep.sh on the login node first"; exit 1; }
python preprocessing/extract_hand_bbox.py --only bdsl47_digits bdsl47_letters
conda deactivate

echo "############## [2/2] TRAIN hand-cropped SI + SD (bdsl_graph env) ##############"
conda activate bdsl_graph
export HF_HOME=/scratch/$USER/hf_cache
export TORCH_HOME=/scratch/$USER/torch_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CAMPAIGN_NW=8
export CAMPAIGN_CSV="results/bhc_handcrop.csv"
python -c "import torch;print('cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0))"

# run_campaign bumps to 50 epochs (matches bdsl47_si/sd) and is skip-existing/resumable.
python -u scripts/run_campaign.py bdsl47_si_handcrop bdsl47_sd_handcrop
echo "=== HANDCROP DONE -> results/bhc_handcrop.csv ==="
