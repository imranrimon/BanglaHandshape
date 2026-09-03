# BanglaHandshape

[![CI](https://github.com/imranrimon/BanglaHandshape/actions/workflows/ci.yml/badge.svg)](https://github.com/imranrimon/BanglaHandshape/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8](https://img.shields.io/badge/Python-3.8-blue.svg)](https://www.python.org/)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg)](https://pytorch.org/)

Code and splits for **“What Breaks on an Unseen Signer? A Controlled Analysis of
Leakage and Generalization in Bangla Sign Language Handshape Recognition.”**

## What this is

Still-image Bangla Sign Language (BdSL) handshape classifiers are usually evaluated
under **signer-dependent** splits, where the same signer's frames appear in both
training and test. This repository contains a controlled study of what happens when
the signer is genuinely unseen.

It provides three things:

- a **leakage audit** that measures how much of the usual accuracy comes from
  near-duplicate and same-session frames rather than recognition,
- a **signer-decodability probe** testing whether removing signer identity from a
  representation actually improves generalization to new signers, and
- **pose-guided distillation**: an RGB-only model trained against a frozen hand-keypoint
  teacher, which needs no keypoints at inference.

> **Scope.** This is an analysis and method study, not a benchmark. The datasets use
> heterogeneous, non-comparable protocols — only BdSL47 has recoverable signer IDs and
> supports a truly user-disjoint split — so the multi-source results are a comparison,
> not a leaderboard.

## Datasets

Five public BdSL still-image sources, plus RSBdSL38 for external replication. They are
**not** redistributed here; see [`docs/DATA_REFETCH.md`](docs/DATA_REFETCH.md) for
sources, download commands, and the expected layout under `data/`.

## Installation

```bash
conda env create -f environment.yml && conda activate bdsl_graph
pip install -e . --no-deps          # deps come from conda
```

## Quickstart

```bash
# smoke test — synthetic data, no GPU or datasets needed
python -m pytest tests/test_banglahandshape_smoke.py -q

# train a baseline (DINOv2-S + LoRA, 3 seeds)
python -m benchmark.baselines.train_baseline \
    --config benchmark/configs/lora.yaml --seeds 0 1 2

# aggregate results
python tools/summarize_seeds.py --csv results_final.csv --markdown
```

## Layout

```
banglahandshape/   shared library — datasets, models, training utils
benchmark/         baselines/, analysis/, figures/, configs/
domain_adapt/      LoRA fine-tuning of a DINOv2 encoder
data_prep/         download, normalization, keypoint caches
alignment/         cross-corpus taxonomy (future work, unverified)
tools/             aggregation, statistics, plotting
scripts/hpc/       SLURM launchers
docs/              protocol, runbook, HPC guide, datasheet
```

## Documentation

- [`docs/EXPERIMENTAL_DESIGN.md`](docs/EXPERIMENTAL_DESIGN.md) — protocol and experiment definitions
- [`docs/RUNBOOK.md`](docs/RUNBOOK.md) — end-to-end commands
- [`docs/HPC.md`](docs/HPC.md) — running on SLURM
- [`docs/DATA_REFETCH.md`](docs/DATA_REFETCH.md) — obtaining the datasets

## Citation

See [`CITATION.cff`](CITATION.cff). A paper citation will be added on publication.

## License

MIT — see [LICENSE](LICENSE). The datasets retain their own upstream licenses.
