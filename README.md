# BanglaHandshape

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
ablations: `docs/SISTER_PAPER_EXPERIMENTAL_DESIGN.md`.

## Repository layout

```
bangla_handshape/            shared library (dataset, DINOv2+LoRA, CNN baselines, train utils)
path1_bangla_dinov2/         LoRA fine-tune a DINOv2 encoder on the handshape corpus
path3_handshape_benchmark/   the benchmark: per-source baselines, transfer matrix, SD/SI gap
  configs/                   linear-probe / LoRA / full-FT / CNN / probe-{imagenet,dinov2-b,siglip,mae} / bdsl47-si|sd / k-shot
tools/                       summarize_seeds.py, paired_bootstrap.py, figstyle.py, mixed_effects.py
tests/                       test_bangla_handshape_smoke.py (synthetic; no data or GPU needed)
docs/                        experimental design, data re-fetch guide, datasheet
scripts/hpc/                 SLURM launchers (DollySods)
RUNBOOK_SISTER_PAPER.md      end-to-end reproduction runbook
HPC_SISTER_PAPER.md          cluster runbook
```

## Installation

```bash
conda env create -f environment.yml && conda activate bdsl_graph
```

Python 3.8, PyTorch (CUDA 11.8), timm, torchvision. Training and evaluation never
import mediapipe — keypoint extraction is an optional, isolated step.

## Datasets

Four public datasets (~195k images, 10–49 classes): **BdSL-MNIST**, **BdSL47**
(Sign Digits + Sign Letters), **BSLD_45**, and **BDSL 49**. They are **not**
bundled here — see `docs/SISTER_PAPER_DATA_REFETCH.md` for the exact sources,
download commands, and the on-disk layout the code expects under `data/`.

```bash
# confirm the datasets are placed correctly (after re-fetch)
python -c "from bangla_handshape.class_alignment import discover_default; \
[print(s.name, s.num_classes) for s in discover_default('.')]"
```

## Quickstart

```bash
# smoke test — synthetic data, no GPU or datasets required
python -m pytest tests/test_bangla_handshape_smoke.py -q

# a baseline (DINOv2-S LoRA, 3 seeds)
python -m path3_handshape_benchmark.train_baseline \
    --config path3_handshape_benchmark/configs/lora.yaml --seeds 0 1 2

# aggregate (mean±std + 95% bootstrap CI)
python tools/summarize_seeds.py --csv results_final.csv --markdown
```

## Reproducing the experiments

Every table and figure in the accompanying paper maps to a documented command:

- **T1** (main SI benchmark), **T2 / T3** (SD-vs-SI gap), **T4** (cross-dataset
  transfer), and ablations **A1–A6** — see
  `docs/SISTER_PAPER_EXPERIMENTAL_DESIGN.md` (master plan) and
  `RUNBOOK_SISTER_PAPER.md` (end-to-end runbook).
- Reviewer analyses (near-duplicate leakage audit, signer-decodability probe,
  size-matched signer/session factorial, per-signer leave-one-user-out variance) —
  `path3_handshape_benchmark/{signer_probe,factorial_signer_session}.py`,
  `tools/mixed_effects.py`, launched by `scripts/hpc/slurm_reviewer_probes.sbatch`.
- Figures — `tools/figstyle.py` (shared publication style) plus
  `tools/plot_*.py` and `path3_handshape_benchmark/plot_*.py`.
- Aggregation — `tools/summarize_seeds.py` (mean±std + 95% bootstrap CI) and
  `tools/paired_bootstrap.py` (paired significance test).

The canonical signer-independent split is fixed and released; all headline numbers
use it. Results are reported as mean±std over three seeds, and per-signer variance
across the 10 leave-one-user-out folds.

## Running on HPC (SLURM)

`HPC_SISTER_PAPER.md` is the authoritative cluster runbook; the primary launcher is
a SLURM job array (`scripts/hpc/slurm_bhc_array.sbatch`, one config per task).
`data/` and `work_dir/` are gitignored and never committed.

## Citation

If you use this benchmark or code, please cite the repository — see `CITATION.cff`
(a paper citation will be added on publication).

## License

Code is released under the [MIT License](LICENSE). The four handshape datasets are
**not** redistributed here and retain their own upstream licenses — see
`docs/SISTER_PAPER_DATA_REFETCH.md` for sources and terms.

## Notes

`environment.yml` is the shared research environment and installs more than this
image-only repository strictly needs.
