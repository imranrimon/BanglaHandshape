#!/bin/bash -l
# =====================================================================
# WVU DollySods — generate the qualitative supplementary figures SF1-SF4 from
# the trained encoders (all needed checkpoints already exist).
#   SF1  per-class confusion heatmap(s)         (plot_confusion.py)
#   SF2  Grad-CAM SD-vs-SI on BdSL47            (plot_gradcam.py)
#   SF3  per-class SD-SI delta bar plot         (shortcut_analysis.py)
#   SF4  reliability diagram / ECE              (shortcut_analysis.py)
# Each step is guarded so one failure can't block the others.
#     sbatch scripts/hpc/slurm_figures.sbatch
# =====================================================================
#SBATCH -J bhc_figures
#SBATCH -c 8
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH -p gpu_2day,gpu_7day
#SBATCH -t 03:00:00
#SBATCH -o slurm_logs/%x_%j.out
#SBATCH -e slurm_logs/%x_%j.err

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs results
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bdsl_graph
export PYTHONIOENCODING=utf-8
export HF_HOME=/scratch/$USER/hf_cache
export TORCH_HOME=/scratch/$USER/torch_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python -c "import torch;print('cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0))"

echo "===== SF1: confusion heatmaps ====="
python -u -m path3_handshape_benchmark.plot_confusion \
  --source bdsl47_letters --encoder-dir work_dir/bhc_bdsl47_si --seed 0 --epoch 50 \
  --normalize --output results/SF1_confusion_bdsl47letters.png || echo "[warn] SF1 letters failed"
python -u -m path3_handshape_benchmark.plot_confusion \
  --source bdsl49_recognition --encoder-dir work_dir/bhc_lora --seed 0 --epoch 50 \
  --normalize --output results/SF1_confusion_bdsl49.png || echo "[warn] SF1 bdsl49 failed"

echo "===== SF2: Grad-CAM SD-vs-SI ====="
python -u -m path3_handshape_benchmark.plot_gradcam \
  --source bdsl47_letters --seed 0 --n 6 \
  --output results/SF2_gradcam_sd_vs_si.png || echo "[warn] SF2 failed"

echo "===== SF3 + SF4: per-class SD-SI delta + ECE calibration ====="
python -u -m path3_handshape_benchmark.shortcut_analysis \
  --si-dir work_dir/bhc_bdsl47_si --sd-dir work_dir/bhc_bdsl47_sd \
  --seed 0 --sources bdsl47_digits bdsl47_letters || echo "[warn] SF3/SF4 failed"

echo "===== FIGURES DONE ====="
ls -la results/SF*.png results/T_shortcut_perclass.md results/T_calibration.md 2>/dev/null
