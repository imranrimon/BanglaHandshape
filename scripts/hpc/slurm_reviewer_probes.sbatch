#!/bin/bash -l
# =====================================================================
# WVU DollySods — reviewer major-revision experiments as a JOB ARRAY.
# The two NEW analyses the reviewer's critique demands, on FROZEN backbones
# (no adapted checkpoint needed — a first cut on the pretrained features):
#
#   §E  signer-decodability battery  (linear vs MLP vs class-conditional)
#       -> fixes the "I(Phi)" overclaim; run across 5 pretraining paradigms
#          so we can show decodability does NOT track SI accuracy.
#   §A  factorial signer / session-proxy + mixed-effects
#       -> replaces Eq. 1's non-identified decomposition with beta1/beta2 + sigma^2.
#
# Submit from ~/BanglaHandshape (after data/BdSL47 is normalised):
#     sbatch -p gpu_2day -t 04:00:00 scripts/hpc/slurm_reviewer_probes.sbatch
# Frozen-feature jobs are fast; dinov2_l extraction over ~47k imgs is the long pole.
# =====================================================================
#SBATCH -J bhc_reviewer
#SBATCH -a 0-5
#SBATCH -c 8
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH -p <PARTITION>              # <-- fill from `sinfo -s` or override: sbatch -p gpu_2day
#SBATCH -t 04:00:00
#SBATCH -o slurm_logs/%x_%A_%a.out
#SBATCH -e slurm_logs/%x_%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs results/reviewer

# `set +u` around activation: conda's MKL activate.d scripts reference unbound
# vars (e.g. MKL_INTERFACE_LAYER), which `set -u` turns into a fatal error.
set +u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bdsl_graph
set -u
export PYTHONIOENCODING=utf-8
# compute nodes are offline: load backbone weights from the login-node-populated
# cache (data_prep/predownload_weights.py). Skips timm's network HEAD check.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# index -> "mode timm_name tag". §E on 5 paradigms; §A on the headline dinov2_b.
SPECS=( \
  "E vit_base_patch14_dinov2.lvd142m dinov2_b" \
  "E vit_large_patch14_dinov2.lvd142m dinov2_l" \
  "E vit_small_patch16_224.augreg_in1k imagenet_vit_s" \
  "E vit_base_patch16_siglip_224.webli siglip_b" \
  "E vit_base_patch16_224.mae mae_b" \
  "A vit_base_patch14_dinov2.lvd142m dinov2_b" )

read -r MODE TIMM TAG <<< "${SPECS[$SLURM_ARRAY_TASK_ID]}"
echo "=== task $SLURM_ARRAY_TASK_ID  mode=$MODE  timm=$TIMM  tag=$TAG ==="
python -c "import torch;print('cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0))"

if [ "$MODE" = "E" ]; then
  python -u -m benchmark.analysis.signer_probe \
      --timm-name "$TIMM" \
      --output "results/reviewer/E_decodability_${TAG}.md"
else
  python -u -m benchmark.analysis.factorial_signer_session \
      --timm-name "$TIMM" --seeds 0 1 2 \
      --output "results/reviewer/A_factorial_${TAG}.csv"
  python -u tools/mixed_effects.py \
      --csv "results/reviewer/A_factorial_${TAG}.csv" \
      --output "results/reviewer/A_mixed_effects_${TAG}.md"
fi
echo "=== done: $MODE $TAG ==="
