# BanglaHandshape

[![CI](https://github.com/imranrimon/BanglaHandshape/actions/workflows/ci.yml/badge.svg)](https://github.com/imranrimon/BanglaHandshape/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8](https://img.shields.io/badge/Python-3.8-blue.svg)](https://www.python.org/)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg)](https://pytorch.org/)

A rigorous **signer-independent (SI) benchmark for still-image Bangla Sign
Language handshape classification**.

Published Bangla handshape papers routinely report 98–99.8% Top-1 — almost always
under *signer-dependent* splits, where the same signer's frames appear in both
train and test. This repository aggregates four public Bangla handshape image
datasets under a **signer-independent protocol**, provides the first
foundation-model-era (DINOv2 / LoRA) baseline table across all of them, and
quantifies the **signer-dependent → signer-independent accuracy drop** — the
identity shortcut — on the one source with real user metadata.

## Key finding

Changing only the split, DINOv2-S + LoRA drops from a published-style **97–99%**
(signer-dependent) to **94.4% / 85.6%** Top-1 on BdSL47 Digits / Letters
(signer-independent). A near-duplicate audit shows **82–88%** of random-split test
images have a near-identical training neighbour, against **0%** once the split is
signer-disjoint — so most of the inflated signer-dependent score is
session/near-duplicate leakage rather than recognition. Full protocol, tables, and
ablations: [`docs/EXPERIMENTAL_DESIGN.md`](docs/EXPERIMENTAL_DESIGN.md).

## Repository layout

```
banglahandshape/     shared library — dataset, DINOv2+LoRA, CNN/keypoint baselines, train utils
benchmark/           the benchmark
├── baselines/       train_baseline, train_keypoint, train_distill, train_fusion, train_adversarial, …
├── analysis/        leakage & signer-decodability probes, factorial, LOUO, transfer, k-shot
├── figures/         plot_confusion, plot_gradcam, plot_perclass_gap
└── configs/         linear-probe / LoRA / full-FT / CNN / probe-{imagenet,dinov2,siglip,mae} / si|sd / k-shot
domain_adapt/        LoRA fine-tune a DINOv2 encoder on the aggregated corpus
data_prep/           dataset download, normalization, resize/keypoint caches
tools/               summarize_seeds, paired_bootstrap, figstyle, mixed_effects
tests/               test_banglahandshape_smoke.py (synthetic; no data or GPU needed)
scripts/hpc/         SLURM launchers (DollySods)
docs/                EXPERIMENTAL_DESIGN, DATA_REFETCH, RUNBOOK, HPC, DATASHEET
```

## Installation

```bash
conda env create -f environment.yml && conda activate bdsl_graph
pip install -e . --no-deps          # register the packages (deps come from conda)
```

Python 3.8, PyTorch (CUDA 11.8), timm, torchvision. Training and evaluation never
import mediapipe — keypoint extraction is an optional, isolated step.

## Datasets

Four public datasets (~195k images, 10–49 classes): **BdSL-MNIST**, **BdSL47**
(Sign Digits + Sign Letters), **BSLD_45**, and **BDSL 49**. They are **not**
bundled here — see [`docs/DATA_REFETCH.md`](docs/DATA_REFETCH.md) for the exact
sources, download commands, and the on-disk layout the code expects under `data/`.

```bash
# confirm the datasets are placed correctly (after re-fetch)
python -c "from banglahandshape.class_alignment import discover_default; \
[print(s.name, s.num_classes) for s in discover_default('.')]"
```

## Quickstart

```bash
# smoke test — synthetic data, no GPU or datasets required
python -m pytest tests/test_banglahandshape_smoke.py -q

# a baseline (DINOv2-S LoRA, 3 seeds)
python -m benchmark.baselines.train_baseline \
    --config benchmark/configs/lora.yaml --seeds 0 1 2

# aggregate (mean±std + 95% bootstrap CI)
python tools/summarize_seeds.py --csv results_final.csv --markdown
```

## Reproducing the experiments

Every table and figure in the accompanying paper maps to a documented command:

- **T1** (main SI benchmark), **T2 / T3** (SD-vs-SI gap), **T4** (cross-dataset
  transfer), and ablations **A1–A6** — see
  [`docs/EXPERIMENTAL_DESIGN.md`](docs/EXPERIMENTAL_DESIGN.md) (master plan) and
  [`docs/RUNBOOK.md`](docs/RUNBOOK.md) (end-to-end runbook).
- Reviewer analyses (near-duplicate leakage audit, signer-decodability probe,
  size-matched signer/session factorial, per-signer leave-one-user-out variance) —
  `benchmark/analysis/{signer_probe,factorial_signer_session,eval_leakage}.py`,
  `tools/mixed_effects.py`, launched by `scripts/hpc/slurm_reviewer_probes.sbatch`.
- Figures — `tools/figstyle.py` (shared publication style) plus
  `benchmark/figures/plot_*.py` and `tools/plot_*.py`.
- Aggregation — `tools/summarize_seeds.py` (mean±std + 95% bootstrap CI) and
  `tools/paired_bootstrap.py` (paired significance test).

The canonical signer-independent split is fixed and released; all headline numbers
use it. Results are reported as mean±std over three seeds, and per-signer variance
across the 10 leave-one-user-out folds.

## Running on HPC (SLURM)

[`docs/HPC.md`](docs/HPC.md) is the authoritative cluster runbook; the primary
launcher is a SLURM job array (`scripts/hpc/slurm_bhc_array.sbatch`, one config per
task). `data/` and `work_dir/` are gitignored and never committed.

## Citation

If you use this benchmark or code, please cite the repository — see
[`CITATION.cff`](CITATION.cff) (a paper citation will be added on publication).

## License

Code is released under the [MIT License](LICENSE). The four handshape datasets are
**not** redistributed here and retain their own upstream licenses — see
[`docs/DATA_REFETCH.md`](docs/DATA_REFETCH.md) for sources and terms.
