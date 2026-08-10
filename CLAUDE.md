# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**BanglaHandshape** — a signer-independent (SI) benchmark for **still-image**
Bangla Sign Language *handshape* classification. It aggregates four/five public
Bangla handshape image datasets under an SI protocol, provides the first
foundation-model-era (DINOv2 / LoRA) baseline table across them, and quantifies
the **signer-dependent → signer-independent accuracy drop** (the identity
shortcut) on the one source with real user metadata (BdSL47).

This is a standalone git repo, extracted as the **"sister paper"** to a separate
word-level video SLR project that lives in the parent directory
(`../` = `SLGTformer/`, which has its own `CLAUDE.md`). This repo is *still-image
only*; it shares the *identity-shortcut* framing and the SI-reporting rigor with
the video project but answers a different question and targets different venues
(ICPR / ICCVW MSLR / IEEE Access / MDPI Sensors). It corresponds to Path 1 + Path 3
of the parent program.

The contribution is a **protocol + benchmark**, not a SOTA chase: published Bangla
handshape papers report 98–99.8% Top-1, almost always under *signer-dependent*
splits. Read `RUNBOOK_SISTER_PAPER.md` (end-to-end reproduction) and
`docs/SISTER_PAPER_EXPERIMENTAL_DESIGN.md` (RQ → tables → baselines → ablations,
the master plan) first.

## Environment

Uses the shared research conda env `bdsl_graph` (Python 3.8, PyTorch + CUDA 11.8,
timm, torchvision; `environment.yml` installs more than this image-only repo
strictly needs). Training/eval **never import mediapipe**.

Per the design doc this benchmark runs on a **local RTX 8000** (~1–2 GPU-days
core); the HPC-only rule applies to the parent video program, not here. Data can
still be placed on HPC scratch if you run there.

**Datasets are not in the repo and are currently not on disk** (deleted in a
storage cleanup, never uploaded to HPC). `data/`, `work_dir/`, `results_final.csv`,
and all `*.npy/*.npz/*.pt/*.pkl` are gitignored. Re-fetch the four sources per
`docs/SISTER_PAPER_DATA_REFETCH.md` before any GPU work. Path source-of-truth is
`bangla_handshape/class_alignment.py` (`DEFAULT_SOURCES`).

### MediaPipe lives in a *separate* env here (differs from the parent repo)

Keypoint extraction (`preprocessing/extract_keypoints.py`, the A6 modality
control) runs in an **isolated `mp_kp` conda env using mediapipe 1.0.0's Tasks
API** (`mediapipe.tasks` `HandLandmarker`), deliberately kept out of `bdsl_graph`
so it can't perturb a running training campaign. This is *not* the legacy
`mp.solutions.holistic` 0.10.9 path the parent `SLGTformer/CLAUDE.md` describes.
It writes a per-source `.npz` cache keyed by absolute image path; the trainer
(`train_keypoint.py`, in `bdsl_graph`) reads the cache and never touches mediapipe.

## Core commands

Run everything from the repo root.

| Task | Command |
|---|---|
| Smoke test (7 tests, synthetic data, no data/GPU needed) | `python -m pytest tests/test_bangla_handshape_smoke.py -v` |
| Single test | `python -m pytest tests/test_bangla_handshape_smoke.py -k lora -v` |
| Confirm datasets on disk | `python -c "from bangla_handshape.class_alignment import discover_default; [print(s.name, s.num_classes) for s in discover_default('.')]"` |
| Train a baseline (image) | `python -m path3_handshape_benchmark.train_baseline --config path3_handshape_benchmark/configs/<cfg>.yaml --seeds 0 1 2` |
| Keypoint (pose) baseline — A6 | `python -m path3_handshape_benchmark.train_keypoint --config path3_handshape_benchmark/configs/keypoint_mlp.yaml --seeds 0 1 2` |
| Leave-one-user-out (signer variance) | `python -m path3_handshape_benchmark.louo_keypoint --seeds 0 1 2` · `python -m path3_handshape_benchmark.louo_appearance --seeds 0 1 2` |
| Cross-dataset transfer matrix — T4 (trains per-source encoders + builds matrix) | `python -m path3_handshape_benchmark.run_transfer_matrix --config path3_handshape_benchmark/configs/transfer_matrix.yaml` |
| T4 low-level eval only (needs per-source `encoder_<A>_*.pt` already collected) | `python -m path3_handshape_benchmark.eval_cross_dataset --encoder-dir work_dir/bhc_xfer_matrix --epoch 50 --seed 0 --lora-rank 8 --lora-targets attn.qkv attn.proj mlp.fc1 mlp.fc2 --output results/T4_transfer_matrix.md` |
| Resumable single-process campaign (many configs × 3 seeds) | `python -u scripts/run_campaign.py lora probe_imagenet_vit_s ...` |
| Aggregate seeds (mean±std + 95% bootstrap CI) | `python tools/summarize_seeds.py --csv results_final.csv --markdown` |
| Paired significance test | `python tools/paired_bootstrap.py --csv results_final.csv --a <expA> --b <expB>` |
| SF1 per-class confusion heatmap | `python -m path3_handshape_benchmark.plot_confusion --source bdsl49_recognition --encoder-dir work_dir/bhc_lora --seed 0 --epoch 50 --output results/SF1.png` |
| Path-1 LoRA-adapt DINOv2 on the corpus | `python -m path1_bangla_dinov2.train --config path1_bangla_dinov2/configs/train_lora.yaml --seed 0` |
| Build the pre-resize image cache (2–3× speedup) | `python preprocessing/build_resized_cache.py --workers 4` |
| Extract MediaPipe keypoints (in `mp_kp` env, needs `.kp_models/hand_landmarker.task`) | `<mp_kp python> preprocessing/extract_keypoints.py --only bdsl47_digits bdsl47_letters` |
| Re-fetch the four datasets (idempotent; stage archives in `$STAGE`) | `bash scripts/refetch_data.sh` |
| Submit the HPC campaign array (fill `<PARTITION>` first) | `sbatch scripts/hpc/slurm_bhc_array.sbatch` |

The lora/full-FT configs default to `num_epoch: 5`; `run_campaign.py` bumps every
config to ≥50 epochs (best-over-epochs selection makes this harmless).

## Running on DollySods (HPC)

**`HPC_SISTER_PAPER.md` is the authoritative HPC runbook** — clone under
`/scratch/$USER` (or `~`), upload `data/` via the tar-pipe-over-ssh recipe there
(or `scripts/refetch_data.sh`), create the `bdsl_graph` env, then submit the job
array. `data/` and `work_dir/` are gitignored; scratch is purge-prone, so back up
`results_final.csv` and any wanted checkpoints.

- `scripts/hpc/slurm_bhc_array.sbatch` — **primary launcher.** A SLURM job array,
  one config per task (19 configs × 3 seeds via `run_campaign.py`), all in
  parallel. Each task writes its **own** `results/bhc_<config>.csv` (`CAMPAIGN_CSV`
  env) because concurrent appends from many nodes to one shared CSV would corrupt
  it; merge afterward with `awk 'FNR==1 && NR!=1{next}1' results/bhc_*.csv >
  results_final.csv`, then `summarize_seeds.py`. Fill `#SBATCH -p <PARTITION>`;
  cap concurrency with `--array=0-18%10` if the cluster is busy.
- `scripts/hpc/slurm_transfer.sbatch` — T4 cross-dataset matrix (not in the
  array; trains 5 single-source encoders × 3 seeds, then builds the matrix).
  A6 (keypoint) reproduction is `HPC_SISTER_PAPER.md §7`.

`scripts/run_campaign.py` is the resumable per-config driver the array calls;
env knobs `CAMPAIGN_CSV` / `CAMPAIGN_SEEDS` / `CAMPAIGN_NW`, and its
process-contention check is cross-platform (`pgrep` on Linux). `scripts/
watchdog_stop.py` and `scripts/run_t1_core.sh` remain Windows-authored (hardcoded
`F:\` / PowerShell) — translate paths if you use them on Linux.

## Architecture — big picture

### The multi-head contract is the central abstraction

Every model — `dinov2_lora.MultiHeadLoRADinov2`, `cnn_baseline.MultiHeadCNN`,
`keypoint_baseline.MultiHeadKeypointMLP` — exposes the **same** interface so the
shared training loop and CSV plumbing work unchanged across appearance and pose:

- `forward(x, src_idx) -> [(source_idx, mask, logits), ...]` for **only the
  sources present in the batch** (one classification head per source).
- `features(x) -> (N, feat_dim)` (used by the transfer matrix and cached probe).
- `.backbone` (its `state_dict()` is what gets checkpointed) and
  `.num_lora_replacements` (for the shared log line).

**Correctness invariant (an audit fix — do not "simplify" it away):** the first
element of each tuple is the *true* source index, not the `enumerate` position.
`forward` skips absent sources, so list position ≠ head index; `train_utils.
{evaluate, multihead_topk}` key on the emitted index. Keying on position
mis-attributes every per-source number on a source-ordered val batch.

Label spaces across sources are kept **disjoint** (multi-head), never silently
merged; a unified space is opt-in only via a verified `--alignment-json`.

### Shared library `bangla_handshape/`

Used by both `path1_bangla_dinov2/` and `path3_handshape_benchmark/`:

- `class_alignment.py` — `discover_source`/`discover_default` enumerate class
  folders from `DEFAULT_SOURCES`. BdSL47 uses a `User XX (...)/Sign N/` layout
  (user-aware enumeration); the other sources are flat `<class>/<image>`.
- `handshape_dataset.py` — `HandshapeDataset` returns `(image, src_idx,
  label_within_source)`; enumerate/split helpers; optional pre-resize image cache.
- `dinov2_lora.py` — `LoRALinear` (frozen base + low-rank A/B delta),
  `apply_lora_to_linears` (substring-matched wrapping), `build_dinov2_lora`.
- `cnn_baseline.py` / `keypoint_baseline.py` — the ResNet and MLP counterparts.
- `train_utils.py` — `multihead_loss`, `multihead_topk`, `train_one_epoch` (AMP),
  `evaluate`.

### Split selection (the heart of the SI protocol)

`train_baseline._train_one_seed` picks the split per source, in this order:

1. **BdSL47** (`bdsl47_digits`/`bdsl47_letters`, the only sources with user IDs)
   → **user-disjoint** SI split via `split.val_users` / `split.test_users`.
   Overridden to a random 80/10/10 (the SD split) **only** when
   `split.force_random: True` — this is the mechanism that makes the T3
   identity-shortcut measurement possible.
2. Sources shipping an author-provided sibling `test/` next to `train/`
   (**BSLD_45**, **BDSL49**) → train on `train/`, evaluate on the held-out
   `test/`. Avoids the augmentation / random-split leak. (Still signer-dependent
   — these sources have no user IDs.)
3. Otherwise → deterministic random 80/10/10 seeded by `split.seed`.

`split.max_train_per_class` caps train items per class (the A4 k-shot ablation;
0 = all). `train_keypoint`/`louo_*` mirror the same selection logic.

### Config-driven adaptation regime

`encoder.arch` chooses `dinov2` (any timm ViT) vs `resnet*`. For ViTs the regime
is selected by flags, not by name: **linear probe** = `lora_rank: 0` + empty
`lora_targets` (backbone frozen); **LoRA** = `lora_rank>0` + `lora_targets`;
**full-FT** = `full_finetune: True`. `timm_name` may be any timm ViT
(DINOv2 / ImageNet / SigLIP / MAE) because LoRA matches on submodule-name
substrings (`attn.qkv`, `attn.proj`, `mlp.fc1`, `mlp.fc2`) shared across ViT
blocks. DINOv2 is created with `dynamic_img_size=True` so its native 518px
positional embeddings interpolate to our 224px inputs.

### Results logging

One row per **(config, source, seed)** appended to `results_final.csv`, with
`Experiment = <Experiment_name>_<source>_seed<N>` and (for the image/keypoint
baselines) `Top5_Policy = best_over_epochs`. `summarize_seeds.py` collapses the
`_seed<N>` and `_loso_test<XX>` suffixes into one group and reports both Top-5
policies. LOUO runners instead use **final-epoch** accuracy (no epoch-peeking)
and report per-source Top-1 **mean ± std across the 10 signers** — that spread is
the honest signer-noise variance.

### Two caches make GPU cheap (both bit-identical to the naive path)

- **Feature-cache probe path** (`train_probe_cached.py`): a frozen backbone
  re-forwards the same images every epoch — pure waste. This extracts each
  image's feature once per (source, split), caches to `.npz`, and trains tiny
  linear heads on the cached vectors (features are identical across seeds).
  `run_campaign.py` auto-routes frozen (probe) configs here via
  `is_probe_config`; ~10×+ faster, same result.
- **Pre-resize image cache** (`build_resized_cache.py` → `work_dir/_img_cache`):
  the transform has no random augmentation, so Resize+CenterCrop→224 uint8 is
  precomputed into memmap shards; the Dataset then only does ToTensor+Normalize.
  A cache miss safely falls back to decoding the original file.
- **Keypoint cache** `work_dir/_kp_cache/<source>.npz` from `extract_keypoints.py`.

### Experiment / table ID scheme (from the design doc)

T1 = main SI benchmark table; T2 = SI-vs-published-SD delta; T3 = SD-vs-SI gap on
BdSL47; T4 = cross-dataset transfer matrix. Ablations: A1 pretraining paradigm,
A2 adaptation method, A3 backbone scale, A4 data efficiency (k-shot), A5 LoRA
placement, A6 modality (RGB vs keypoints). Config filenames and their header
comments are tagged with these IDs.

## Gotchas

- **SI is the ground truth; SD exists only to measure the shortcut.** `bdsl47_sd.yaml`
  (`force_random: True`) is not a result to report on its own — it pairs with
  `bdsl47_si.yaml` as `gap = Top1(SD) − Top1(SI)` (predicted ≥10 pp). Without
  `force_random`, BdSL47 is *always* user-disjoint, so the runbook's older "set
  `val_users:[]`/`test_users:[]` for SD" instruction was a silent no-op (fixed).
- **BSLD_45 has ~0% MediaPipe hand detection** — its images ship with a hand
  skeleton already drawn on — so it's excluded from keypoint (A6) experiments.
  `train_keypoint`/`louo_keypoint` skip any source with fewer than `MIN_TRAIN`
  detected keypoints; the per-source detection rate is itself a benchmark stat.
- **Re-fetch pitfalls** (`docs/SISTER_PAPER_DATA_REFETCH.md`): rename BdSL47's
  `Sign Alphabets` → `Sign Letters`; keep BDSL49's doubled
  `Recognition_1/Recognition_1/{train,test}` nesting; verify BSLD_45 is the
  45-class set with `Train/Val/Test/`. `discover_default` skips missing sources
  silently, so a wrong path just drops a source from the table.
- **Windows-authored scaffolding.** `scripts/run_campaign.py`,
  `scripts/watchdog_stop.py`, and `scripts/run_t1_core.sh` hardcode Windows
  paths (`F:\`, `C:/Users/rimon/anaconda3/...`) and use PowerShell/`Win32_Process`
  to detect running jobs — translate paths / process checks on Linux. The
  `num_workers>0` retry loop in `run_campaign.py` works around a Windows
  "shared file mapping 1455" DataLoader crash; drop `num_workers` to 0 if flaky.
- **Resume is automatic.** `train_baseline` writes a full-state `resume_seed<N>.pt`
  each epoch and resumes from it; `run_campaign.py` is idempotent (skips any
  `(config, seed)` already present in `results_final.csv`).
- **T4 transfer matrix needs per-source encoders.** The headline sweep trains one
  *joint* multi-head encoder and saves a single `encoder_seed<N>_epoch<E>.pt`;
  `eval_cross_dataset.py` needs one encoder *per source* (`encoder_<A>_seed<N>_
  epoch<E>.pt`) or its rows would all reuse the joint encoder. `run_transfer_
  matrix.py` handles this: it trains a single-source LoRA encoder per source
  (in-memory single-source configs), collects them into `work_dir/bhc_xfer_matrix`
  under the per-source names, and calls `eval_cross_dataset`. Use it (or
  `slurm_transfer.sbatch`) rather than pointing `eval_cross_dataset` at the joint
  `bhc_lora` dir. `_per_source_splits` in `eval_cross_dataset` mirrors
  `train_baseline`'s split selection (incl. author-provided test dirs) so T4 and
  T1 use identical eval sets.
- **Only the backbone is checkpointed** (`model.backbone.state_dict()`), not the
  classification heads. Any post-hoc eval from a checkpoint (`eval_cross_dataset`,
  `plot_confusion`) therefore re-fits a fresh head on frozen features — faithful
  for the encoder's confusion structure, but not a literal replay of the trained
  head's predictions.
- **`path1_bangla_dinov2/extract_features.py` is referenced in its README but not
  yet scaffolded** — only `train.py` + `configs/` exist. Path 1's downstream
  BdSLW60 feature extraction depends on the parent `SLGTformer/` repo's data, so
  it is outside the sister paper's T1–T4 / A1–A6 critical path.
