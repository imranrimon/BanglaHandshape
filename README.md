# BanglaHandshape

A rigorous **signer-independent benchmark for still-image Bangla Sign Language
handshape classification**.

Published Bangla handshape papers routinely report 98–99.8% Top-1 — almost
always under *signer-dependent* splits, where the same signer's frames appear in
both train and test. This repo aggregates four public Bangla handshape image
datasets under a **signer-independent protocol**, provides the first
foundation-model-era (DINOv2 / LoRA) baseline table across all of them, and
quantifies the **signer-dependent → signer-independent accuracy drop** (the
identity shortcut) on the one source with real user metadata.

## What's here

```
bangla_handshape/            shared library (dataset, DINOv2+LoRA, CNN baselines, train utils)
path1_bangla_dinov2/         LoRA fine-tune a DINOv2 encoder on the handshape corpus
path3_handshape_benchmark/   the benchmark: per-source baselines, transfer matrix, SD/SI gap
  configs/                   linear-probe / LoRA / full-FT / CNN / probe-{imagenet,dinov2-b,siglip,mae} / bdsl47-si|sd / k-shot
tools/                       summarize_seeds.py (mean±std + bootstrap CI), paired_bootstrap.py
tests/                       test_bangla_handshape_smoke.py
docs/
  SISTER_PAPER_EXPERIMENTAL_DESIGN.md   RQ → tables → baselines → ablations → protocol
  SISTER_PAPER_DATA_REFETCH.md          how to download the four datasets + exact layout
RUNBOOK_SISTER_PAPER.md      end-to-end reproduction runbook
```

## Datasets

Four public datasets (~195k images, 10–49 classes): **BdSL-MNIST**, **BdSL47**
(Sign Digits + Sign Letters), **BSLD_45**, **BDSL 49**. They are not bundled
here — see `docs/SISTER_PAPER_DATA_REFETCH.md` for the exact sources, download
commands, and the on-disk layout the code expects under `data/`.

## Quickstart

```bash
conda env create -f environment.yml && conda activate bdsl_graph

# 1. confirm the datasets are placed correctly (after re-fetch)
python -c "from bangla_handshape.class_alignment import discover_default; \
[print(s.name, s.num_classes) for s in discover_default('.')]"
python -m pytest tests/test_bangla_handshape_smoke.py -q

# 2. a baseline (DINOv2-S LoRA, 3 seeds)
python -m path3_handshape_benchmark.train_baseline \
    --config path3_handshape_benchmark/configs/lora.yaml --seeds 0 1 2

# 3. aggregate
python tools/summarize_seeds.py --csv results_final.csv --markdown
```

The full baseline slate, ablation matrix, and the identity-shortcut (SD vs SI)
experiment are laid out in `docs/SISTER_PAPER_EXPERIMENTAL_DESIGN.md`.

## Notes

This is the companion ("sister paper") to a separate word-level Bangla sign
language recognition project, extracted here as a standalone benchmark. The
`environment.yml` is the shared research environment and installs more than this
image-only repo strictly needs.
