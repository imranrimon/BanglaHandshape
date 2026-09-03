# BanglaHandshape

[![CI](https://github.com/imranrimon/BanglaHandshape/actions/workflows/ci.yml/badge.svg)](https://github.com/imranrimon/BanglaHandshape/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8](https://img.shields.io/badge/Python-3.8-blue.svg)](https://www.python.org/)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg)](https://pytorch.org/)

Code, splits, and figures for **“What Breaks on an Unseen Signer? A Controlled
Analysis of Leakage and Generalization in Bangla Sign Language Handshape
Recognition.”**

Published still-image Bangla Sign Language (BdSL) handshape classifiers report
98–99.8% Top-1, almost always under **signer-dependent (SD)** splits in which the
same signer's frames appear in both training and test. This repository contains the
controlled study of what actually breaks when the signer is unseen: a
near-duplicate/session **leakage audit**, a **signer-decodability probe**, a paired
**RGB-vs-geometry** comparison, and **pose-guided distillation** — an RGB-only
student trained against a frozen keypoint teacher that needs no keypoints at
inference.

> **Scope — this is an analysis/method study, not a benchmark.** The five sources
> use heterogeneous, non-comparable protocols (only BdSL47 supports a truly
> user-disjoint split), so the multi-source table is a *multi-dataset comparison*,
> never a ranked leaderboard. Please do not cite it as one.

## Key findings

**1. Changing only the split costs up to 13.7 pp.** DINOv2-S + LoRA, same encoder,
BdSL47:

| Split | Digits | Letters |
|---|---|---|
| Signer-dependent (published-style) | 97.7 ± 0.0 | 99.3 ± 0.0 |
| Signer-independent (user-disjoint) | 94.4 ± 0.8 | 85.6 ± 1.2 |
| **Gap (upper bound)** | **3.3** | **13.7** |

The SD and SI test sets differ, so this gap is an **upper bound** on the
signer-overlap effect. The correctly paired estimator — same held-out test signer,
trained with vs. without that signer — puts it at **30.6 ± 20.4 pp** across the 10
Letters signers (SD 99.2 vs. SI 68.6), with one signer collapsing to 17.0% when unseen.

**2. Most of the inflated SD score is leakage, not recognition.** Share of eval
images with a training neighbour at DINOv2 cosine ≥ 0.98:

| Source | Split | ≥0.95 | ≥0.98 | ≥0.99 |
|---|---|---|---|---|
| Digits | SD (random) | 97.8 | **88.4** | 70.5 |
| Digits | SI (user-disjoint) | 12.4 | **0.0** | 0.0 |
| Letters | SD (random) | 97.1 | **82.3** | 56.4 |
| Letters | SI (user-disjoint) | 9.8 | **0.0** | 0.0 |
| RSBdSL38 | provided (external) | 44.9 | **19.9** | 2.8 |
| BDSL49 | provided (external) | 2.9 | **0.3** | 0.0 |

82–88% of random-split test images have a near-identical training frame, against
**0%** once the split is signer-disjoint. Honest evaluation must be *both*
signer-disjoint *and* de-duplicated. The audit needs no signer labels, so it
transfers to external datasets and reveals a leakage spectrum
(BdSL47-SD ≫ RSBdSL38 ≫ BDSL49).

**3. Signer-invariance does not predict signer-independent accuracy.** Balanced
accuracy of a signer classifier on frozen features (chance = 12.5%):

| Representation | Digits | Letters | SI Top-1 (D/L) |
|---|---|---|---|
| Pose (keypoints) | 54.1 | 44.7 | 99.9 / 85.5 |
| DINOv2-B (frozen) | 100.0 | 99.4 | 81.3 / 68.2 |
| LoRA (plain) | 99.9 | 99.2 | 94.4 / 85.6 |
| LoRA + pose-distill (ours) | 99.8 | 98.7 | 95.2 / 88.2 |
| LoRA + signer-adversary | **42.8** | **48.3** | 93.9 / 85.7 |

Adversarial training genuinely scrubs signer identity yet does **not** raise SI
accuracy; our most accurate model leaves identity almost perfectly decodable. Across
the representations tested, linear decodability is decoupled from generalization —
consistent with domain-generalization results that invariance is neither necessary
nor sufficient for out-of-domain accuracy.

**4. Pose-guided distillation helps, but the gain is mostly generic.** Distilling a
frozen keypoint teacher into the RGB student reaches **95.2 / 88.2** (+0.8/+2.6 pp
over plain LoRA) with no keypoints at inference, and the student (88.2) exceeds its
pose teacher (85.5). But controls show the effect is largely *distillation*, not
geometry transfer: label smoothing alone recovers much of it (95.3/87.6), and an
**RGB** teacher matches it (95.2/88.5). We therefore report C3 as a cheap,
keypoint-free recipe — not as evidence that geometry transfers invariance.

## Datasets

Five public BdSL still-image sources, plus RSBdSL38 as an external replication.
**Only BdSL47 has recoverable per-signer folders**, so it is the sole source
supporting a user-disjoint SI split and the focus of all signer-shift analysis.

| Source | #Classes | Signer IDs? | Protocol |
|---|---|---|---|
| BdSL47 Sign Digits | 10 | yes | user-disjoint (SI) |
| BdSL47 Sign Letters | 37 | yes | user-disjoint (SI) |
| BDSL49 (Recognition) | 49 | — | provided test (SD) |
| BSLD_45 | 45 | — | provided test (SD) |
| BdSL-MNIST | 37 | — | random 80/10/10 (SD) |

Datasets are **not** redistributed here. See
[`docs/DATA_REFETCH.md`](docs/DATA_REFETCH.md) for sources, download commands, the
on-disk layout, and known gotchas (notably BdSL47's case-inconsistent folder names).

**SI split (BdSL47):** train on all users except {4, 5}; validate on user 4; test on
user 5. The paired SD control replaces this with a random 80/10/10 over all users.

## Repository layout

```
banglahandshape/     shared library — datasets, DINOv2+LoRA, CNN/keypoint models, training utils
benchmark/
├── baselines/       train_baseline, train_keypoint, train_distill, train_fusion, train_adversarial, train_probe_cached
├── analysis/        leakage, signer probe, background probe, factorial, LOUO, transfer, k-shot, TTA, attribution
├── figures/         plot_confusion, plot_gradcam, plot_perclass_gap
└── configs/         35 YAMLs — probe/LoRA/full-FT/CNN, si|sd variants, k-shot
domain_adapt/        LoRA fine-tuning of a DINOv2 encoder on the aggregated corpus
data_prep/           download, normalization, resize/keypoint caches
alignment/           cross-corpus handshape taxonomy (future work — needs expert verification)
tools/               summarize_seeds, paired_bootstrap, mixed_effects, figstyle, plotting
tests/               synthetic smoke tests (no data or GPU required)
scripts/hpc/         36 SLURM launchers (DollySods)
docs/                EXPERIMENTAL_DESIGN, RUNBOOK, HPC, DATA_REFETCH, DATASHEET
```

## Installation

```bash
conda env create -f environment.yml && conda activate bdsl_graph
pip install -e . --no-deps          # register the packages (deps come from conda)
```

Python 3.8, PyTorch (CUDA 11.8), timm, torchvision. Training and evaluation never
import mediapipe — keypoint extraction is an optional, isolated step.

## Quickstart

```bash
# smoke test — synthetic data, no GPU or datasets required
python -m pytest tests/test_banglahandshape_smoke.py -q

# main RGB baseline (DINOv2-S + LoRA, 3 seeds)
python -m benchmark.baselines.train_baseline \
    --config benchmark/configs/lora.yaml --seeds 0 1 2

# aggregate (mean±std + 95% bootstrap CI)
python tools/summarize_seeds.py --csv results_final.csv --markdown
```

## Reproducing the paper

Each analysis maps to a module. All headline numbers are mean ± std over 3 seeds.

| Claim | Module |
|---|---|
| Near-duplicate leakage audit (E1) | `benchmark/analysis/eval_leakage.py` |
| Same audit on provided external splits (E13) | `benchmark/analysis/eval_leakage_provided.py` |
| Signer-decodability probe (E3) | `benchmark/analysis/eval_signer_probe.py`, `signer_probe.py` |
| Paired leave-one-user-out gap (E2) | `benchmark/analysis/run_louo_paired.py`, `louo_appearance.py`, `louo_keypoint.py` |
| Background-only probe (E8) | `benchmark/analysis/eval_background_probe.py` |
| Size-matched signer/session factorial (E15) | `benchmark/analysis/factorial_signer_session.py` |
| Pose-guided distillation + KD controls (E5) | `benchmark/baselines/train_distill.py` |
| Signer-adversarial control | `benchmark/baselines/train_adversarial.py` |
| Cross-corpus transfer matrix | `benchmark/analysis/run_transfer_matrix.py` |
| Test-time adaptation (E11) | `benchmark/analysis/eval_tta.py` |
| Calibration / robustness / imbalance metrics | `benchmark/analysis/eval_metrics.py`, `eval_robustness.py` |
| Per-signer variance modelling | `tools/mixed_effects.py` |
| Significance testing | `tools/paired_bootstrap.py` |

Full protocol and experiment definitions:
[`docs/EXPERIMENTAL_DESIGN.md`](docs/EXPERIMENTAL_DESIGN.md); end-to-end commands:
[`docs/RUNBOOK.md`](docs/RUNBOOK.md).

## Running on HPC (SLURM)

[`docs/HPC.md`](docs/HPC.md) is the authoritative cluster runbook. Campaigns launch
as **parallel SLURM job arrays** (one config per task) under `scripts/hpc/`. `data/`
and `work_dir/` are gitignored and never committed.

## Limitations

- Only BdSL47 supports a user-disjoint split; BDSL49, BSLD_45, and BdSL-MNIST have
  no recoverable signer metadata, so their numbers are SD and not comparable to the
  BdSL47 SI columns.
- The SD−SI difference conflates signer overlap with held-out-signer difficulty and
  acquisition differences; the paired LOUO estimate is the sound one.
- The keypoint stream keeps a single hand, so two-handed signs lose information.
- Signer decodability is measured **linearly**, a lower bound on recoverable identity.
- BSLD_45 could not be re-fetched from any verified public source; those cells are
  carried from the original run.
- The cross-corpus taxonomy in `alignment/` is unverified and must not back reported
  numbers until an expert pass is complete.

## Citation

See [`CITATION.cff`](CITATION.cff). A paper citation will be added on publication.

## License

Code is released under the [MIT License](LICENSE). The datasets are **not**
redistributed here and retain their own upstream licenses — see
[`docs/DATA_REFETCH.md`](docs/DATA_REFETCH.md).
