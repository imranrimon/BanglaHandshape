#!/bin/bash -l
# =====================================================================
# Eval-based analyses re-run on the FIXED-data encoders, as a parallel dispatch
# array (one command per array task; each writes its OWN output to avoid
# concurrent-write corruption). Completes the fixed-data reproduction of the
# remaining paper tables:
#   0 E1  near-dup leakage (tab:leakage)      1 E1  leakage on provided splits
#   2 E3  signer-decodability probe (tab:probe)   3 E8 background probe (tab:bgprobe)
#   4 E9/E11 TTA (Tent/IM)                    5 E10 calibration + recalibration (tab:ece)
#   6-15 E2/E11 paired LOUO, Letters users 01-10   16-18 E14 signer-aware k-shot k=1/2/4
#     sbatch scripts/hpc/slurm_bhc_eval.sbatch
# =====================================================================
#SBATCH -J bhc_eval
#SBATCH -a 0-18
#SBATCH -c 8
#SBATCH -n 1
#SBATCH --gpus 1
#SBATCH -p gpu_2day,gpu_7day
#SBATCH --exclude=dscog035,dscog036,dsdmg002
#SBATCH -t 08:00:00
#SBATCH --requeue
#SBATCH -o slurm_logs/%x_%A_%a.out
#SBATCH -e slurm_logs/%x_%A_%a.err

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs results
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bdsl_graph
export PYTHONIOENCODING=utf-8
export HF_HOME="$HOME/hf_cache"
export TORCH_HOME="$HOME/torch_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -c "import torch; torch.zeros(1).cuda()" 2>/dev/null || {
  echo "GPU not usable on $(hostname) -> requeue"; scontrol requeue "$SLURM_JOB_ID"; exit 1; }

CMDS=(
  "eval_leakage --out results/T_leakage.md"
  "eval_leakage_provided --out-dir results"
  "eval_signer_probe --seed 0 --out results/T_signer_probe.md"
  "eval_background_probe --out results/T_background_probe.md"
  "eval_tta --si-dir work_dir/bhc_lora --seed 0 --epoch 50 --results-csv results/bhc_tta.csv"
  "eval_metrics --si-dir work_dir/bhc_lora --sd-dir work_dir/bhc_bdsl47_sd --seed 0 --epoch 50 --metrics-md results/T_ece.md --recalibration-md results/T_recalib.md"
  "run_louo_paired --source bdsl47_letters --test-user 1  --seed 0 --results-csv results/bhc_paired_u01.csv"
  "run_louo_paired --source bdsl47_letters --test-user 2  --seed 0 --results-csv results/bhc_paired_u02.csv"
  "run_louo_paired --source bdsl47_letters --test-user 3  --seed 0 --results-csv results/bhc_paired_u03.csv"
  "run_louo_paired --source bdsl47_letters --test-user 4  --seed 0 --results-csv results/bhc_paired_u04.csv"
  "run_louo_paired --source bdsl47_letters --test-user 5  --seed 0 --results-csv results/bhc_paired_u05.csv"
  "run_louo_paired --source bdsl47_letters --test-user 6  --seed 0 --results-csv results/bhc_paired_u06.csv"
  "run_louo_paired --source bdsl47_letters --test-user 7  --seed 0 --results-csv results/bhc_paired_u07.csv"
  "run_louo_paired --source bdsl47_letters --test-user 8  --seed 0 --results-csv results/bhc_paired_u08.csv"
  "run_louo_paired --source bdsl47_letters --test-user 9  --seed 0 --results-csv results/bhc_paired_u09.csv"
  "run_louo_paired --source bdsl47_letters --test-user 10 --seed 0 --results-csv results/bhc_paired_u10.csv"
  "run_kshot_signer --source bdsl47_letters --k 1 --seed 0 --results-csv results/bhc_kshot_signer_k1.csv"
  "run_kshot_signer --source bdsl47_letters --k 2 --seed 0 --results-csv results/bhc_kshot_signer_k2.csv"
  "run_kshot_signer --source bdsl47_letters --k 4 --seed 0 --results-csv results/bhc_kshot_signer_k4.csv"
)
ARGS=${CMDS[$SLURM_ARRAY_TASK_ID]}
echo "=== task $SLURM_ARRAY_TASK_ID: $ARGS ==="
python -u -m "benchmark.${ARGS%% *}" ${ARGS#* }
echo "=== done task $SLURM_ARRAY_TASK_ID ==="
