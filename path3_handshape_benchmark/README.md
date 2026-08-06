# Path 3 — Bangla Handshape Benchmark (sister paper scaffold)

**Goal**: rigorous, signer-independent benchmark for **still-image Bangla sign
language handshape recognition** with foundation-model baselines, across the
four image datasets on disk. Independent publishable contribution from the
word-level project; reuses the `bangla_handshape/` shared library.

The motivation: published Bangla handshape papers routinely report 99%+ on
BdSL-MNIST / Ishara-Lipi / BDSL 49, almost always under signer-dependent
splits. A clean three-way comparison (linear probe vs LoRA vs from-scratch
CNN baseline) on **signer-disjoint splits across all four datasets** has
not been done.

## Layout

```
path3_handshape_benchmark/
├── README.md                  (this file)
├── configs/
│   ├── linear_probe.yaml      (frozen DINOv2 + per-source linear head)
│   ├── lora.yaml              (LoRA-tuned DINOv2 + per-source head)
│   └── cross_dataset.yaml     (train on one source, eval on the others)
├── train_baseline.py          entry point: per-source training and reporting
└── eval_cross_dataset.py      entry point: zero-shot cross-source eval
```

The shared library `bangla_handshape/` provides the dataset, LoRA wrapper,
multi-head model, and training utilities. This directory only adds:

* benchmark configs that pin hyperparameters per-baseline
* an entry point that produces a per-source results table

## Designed contributions

1. **Headline table A**: Top-1 per source under signer-disjoint splits, for
   four model families:
   * Linear probe on frozen ImageNet-21k ViT
   * Linear probe on frozen DINOv2 ViT-S/14 (the foundation-model baseline)
   * LoRA-tuned DINOv2 ViT-S/14
   * From-scratch CNN baseline (small ResNet) for context
2. **Headline table B**: cross-dataset zero-shot transfer matrix
   (train on BdSL-MNIST → test on BSLD_45, etc.). Quantifies how much each
   source's "handshape model" generalises across taxonomies.
3. **Headline table C**: identity-shortcut measurement on BdSL47
   (signer-disjoint vs signer-dependent), the only source with explicit user
   metadata. Same protocol as Path 1's identity-shortcut experiment.

## Run commands

### Step P3.1 — train + evaluate the four baselines on each source

```bash
python -m path3_handshape_benchmark.train_baseline \
    --config path3_handshape_benchmark/configs/linear_probe.yaml --seeds 0 1 2

python -m path3_handshape_benchmark.train_baseline \
    --config path3_handshape_benchmark/configs/lora.yaml --seeds 0 1 2
```

Each run writes per-source Top-1 to `results_final.csv` with
`Experiment` = `<config_stem>_<source>_seed<N>`.

### Step P3.2 — aggregate

```bash
python tools/summarize_seeds.py --csv results_final.csv --markdown > results/path3_table_A.md
```

### Step P3.3 — cross-dataset transfer matrix

```bash
python -m path3_handshape_benchmark.eval_cross_dataset \
    --encoder-checkpoint work_dir/bhc_lora/encoder_epoch5.pt \
    --output results/path3_table_B.md
```

## Why it's a sister paper, not a chapter

The word-level project (Options A/B/C) is a *video* recognition paper. This is
a *still-image* paper. They share infrastructure (the `bangla_handshape/`
library, the rigor protocol of signer-independent reporting), but they answer
different questions and would target different venues. Likely venues:
ICPR, ICCVW MSLR, IEEE Access, MDPI Sensors.

If both papers land, they cross-cite each other on the methodological
identity-shortcut argument.
