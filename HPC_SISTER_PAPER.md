# HPC Runbook — BanglaHandshape sister paper (WVU DollySods)

Run the full sister-paper benchmark on DollySods as a **SLURM job array**: one
config per GPU, all in parallel. The ~7-day local serial matrix finishes in
**<~1 day** (critical path = `full_ft`'s 3 seeds). Mirrors the conventions in
`../SLGTformer/scripts/hpc/*.sbatch`.

The local RTX 8000 is GPU-compute-bound (~1085 s/epoch for 2 concurrent ViT-S
LoRA streams); HPC wins purely by running configs on separate GPUs at once.

---

## 0. What runs

`scripts/hpc/slurm_bhc_array.sbatch` fans out **19 configs**, each as its own
array task running that config × 3 seeds via `run_campaign.py`:

| Group | Configs |
|---|---|
| A1/A3 probes | linear_probe, probe_imagenet_vit_s, probe_dinov2_b, probe_dinov2_l, probe_siglip_b, probe_mae_b |
| A2 adaptation | lora, lora_r4, lora_r16, lora_r32, full_ft |
| A5 placement | lora_attn_only |
| T1 CNN | cnn_resnet18_scratch, cnn_resnet50_imagenet |
| A4 k-shot | kshot_k5, kshot_k10, kshot_k20 |
| T3 gap | bdsl47_si, bdsl47_sd |

A6 (keypoint modality) is a separate optional step (§6) — it was already
produced locally.

---

## 1. Code

```bash
cd ~                       # or /scratch/$USER
git clone https://github.com/imranrimon/BanglaHandshape.git   # or: git pull
cd BanglaHandshape
```

## 2. Data (git-ignored — must be placed under `data/`)

Only the image sets are needed. Easiest: rsync the arranged local copy up:

```bash
# from the local Windows box (git-bash / WSL):
rsync -avz F:/BanglaHandshape/data/BdSL-MNIST         <hpc>:~/BanglaHandshape/data/
rsync -avz "F:/BanglaHandshape/data/BdSL47"           <hpc>:~/BanglaHandshape/data/
rsync -avz F:/BanglaHandshape/data/BSLD_45            <hpc>:~/BanglaHandshape/data/
rsync -avz F:/BanglaHandshape/data/bdsl49_extracted   <hpc>:~/BanglaHandshape/data/
```

Or re-fetch on HPC per `docs/SISTER_PAPER_DATA_REFETCH.md`.

## 3. Environment (once)

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda env create -f environment.yml        # name: bdsl_graph
conda activate bdsl_graph
python -c "import torch;print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

> **timm pretrained weights** (DINOv2 / ViT / SigLIP / MAE) download from
> HuggingFace on first use. If compute nodes have no outbound HTTPS, pre-fetch on
> the **login node** first (shared HF cache), e.g. run the smoke step in §5 there
> once, or `export HF_HOME=~/.cache/huggingface` and pull the models.

## 4. Smoke test (login node, ~3 min)

```bash
python scripts/run_campaign.py linear_probe    # probes are cheap; writes results/bhc_linear_probe.csv
```

If it writes rows, the pipeline works.

## 5. Pick a partition, then submit the array

```bash
sinfo -s                                       # partitions + idle counts
# edit <PARTITION> in scripts/hpc/slurm_bhc_array.sbatch (24GB dscog[005-030] is plenty)
sbatch scripts/hpc/slurm_bhc_array.sbatch
squeue -u $USER
```

Each task writes its **own** `results/bhc_<config>.csv` (concurrent appends to a
single shared CSV across nodes would corrupt it). If the cluster is busy, cap
concurrency by submitting `--array=0-18%10`.

## 6. Merge results + aggregate

```bash
# combine per-task CSVs, keeping a single header row:
awk 'FNR==1 && NR!=1 {next} {print}' results/bhc_*.csv > results_final.csv
python tools/summarize_seeds.py --csv results_final.csv --markdown > results/master_table.md
```

## 7. (optional) A6 keypoint modality — reproduce on HPC

Already produced locally; only needed for an HPC-only reproduction. Extraction
uses an isolated mediapipe env (Tasks API); the MLP + LOUO run in `bdsl_graph`.

```bash
conda create -n mp_kp python=3.11 -y
conda activate mp_kp && pip install mediapipe opencv-python-headless   # Tasks API needs mediapipe >=0.10
mkdir -p .kp_models
curl -sL -o .kp_models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
python preprocessing/extract_keypoints.py --only bdsl47_digits bdsl47_letters

conda activate bdsl_graph
python -m path3_handshape_benchmark.train_keypoint  --config path3_handshape_benchmark/configs/keypoint_mlp.yaml --seeds 0 1 2
python -m path3_handshape_benchmark.louo_keypoint   --seeds 0 1 2      # pose LOUO across 10 signers
python -m path3_handshape_benchmark.louo_appearance --seeds 0 1 2      # frozen DINOv2-B LOUO (symmetric)
```

## 8. Pitfalls (from the main HPC guide)

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: mediapipe` | `pip install mediapipe` (Linux: pip-only; Tasks API needs ≥0.10.x) |
| `num_worker` hangs on NFS | set `CAMPAIGN_NW=0` in the sbatch (data on local scratch avoids this) |
| `CUDA out of memory` on a smaller GPU | lower `batch_size` in the config (64 → 32) |
| `UnicodeEncodeError` in stdout | `export PYTHONIOENCODING=utf-8` (already in the sbatch) |
| timm can't download on compute node | pre-fetch backbones on the login node (shared HF cache) |

## 9. Notes

- `run_campaign.py` is **idempotent** (skip-existing per seed + per-epoch resume
  checkpoints), so re-submitting after a timeout continues where it stopped.
- Env overrides: `CAMPAIGN_CSV` (per-task output), `CAMPAIGN_SEEDS` (default
  `"0 1 2"`), `CAMPAIGN_NW` (DataLoader workers).
- **BSLD_45 caveat** carries into the paper: its images ship with the hand
  skeleton drawn on (0% MediaPipe re-detection) — report with that caveat; it is
  excluded from A6. See the project notes / `docs/`.
