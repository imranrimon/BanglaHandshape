# Path 1 — Bangla-handshape-adapted DINOv2

**Goal**: produce a Bangla-domain-adapted DINOv2 image encoder by LoRA-tuning
on the ~195k aggregated still-image handshape corpus, then plug it into
Option B's feature-isolation experiment as a third encoder
(MediaPipe vs **generic DINOv2** vs **Bangla-DINOv2**).

If the Bangla-adapted encoder gives *higher* downstream Top-1 on BdSLW60-SI
*and* a *smaller* signer-dependent → signer-independent gap, that is the
sharpest methodological finding the entire project produces.

## Layout

```
path1_bangla_dinov2/
├── README.md            (this file)
├── configs/
│   └── train_lora.yaml  LoRA tuning hyperparameters
├── train.py             entry point: LoRA fine-tune DINOv2 on aggregated corpus
└── extract_features.py  apply the LoRA-tuned encoder to BdSLW60 hand crops
                         (produces NPYs in the same shape as Option B's DINOv2)
```

The shared library `bangla_handshape/` (in the repo root) provides:
* `class_alignment.py` — per-source class enumeration; supports an optional
  user-supplied alignment table for a unified label space.
* `handshape_dataset.py` — multi-source PyTorch Dataset returning
  `(image, source_idx, label_within_source)`.
* `dinov2_lora.py` — `LoRALinear`, `apply_lora_to_linears`, and a
  `MultiHeadLoRADinov2` model with one classification head per source.
* `train_utils.py` — multi-head loss, multi-head topk, train/eval loops.

## Pipeline

1. Inventory the four image sources on disk.
2. Build the multi-source dataset with one head per source (no class alignment
   needed for Path 1's transfer use case).
3. LoRA-fine-tune DINOv2 ViT-S/14 with AdamW + cosine LR.
4. Save *backbone-only* state dict (not the heads — heads aren't reused).
5. Apply the encoder to BdSLW60 hand crops via
   `extract_features.py`. Output is `(N, 384, T_max, V=3, M=1)` — drop-in
   replacement for Option B's `data/bdsl_si_dino/` artefacts.
6. Train `model.flat_temporal.Model` on the new features — call this config
   `bdsl_bdinov2_temporal_si.yaml` (build it from
   `config/bdsl_dino_temporal_si.yaml` by changing only the data path).

## Run commands (offline-runnable; smoke-tested)

### Step P1.1 — train the LoRA-adapted encoder

```bash
python -m path1_bangla_dinov2.train --config path1_bangla_dinov2/configs/train_lora.yaml --seed 0
```

Wall-clock estimate on Quadro RTX 8000: **~3-6 h** for ~10 epochs over
~150k images at batch 64, DINOv2 ViT-S/14 with LoRA rank 8.

Output: `work_dir/bdino_lora/encoder_epoch<N>.pt` (backbone state dict only).

### Step P1.2 — apply encoder to BdSLW60 hand crops

```bash
python -m path1_bangla_dinov2.extract_features ^
    --dataset-root "Word_level_Bangla_Sign_Language_Dataset/BdSLW30" ^
    --output-dir data/bdsl_si_bdino ^
    --cache-dir data/bdsl_bdino_cache ^
    --encoder-checkpoint work_dir/bdino_lora/encoder_epoch10.pt ^
    --splits train val test ^
    --device cuda --batch-size 64
```

Wall-clock estimate: ~3-5 h GPU (same order as Option B's generic DINOv2 pass).

### Step P1.3 — train FlatTemporal on the Bangla-adapted features

Copy `config/bdsl_dino_temporal_si.yaml` to
`config/bdsl_bdino_temporal_si.yaml` and change only the data paths from
`data/bdsl_si_dino/` to `data/bdsl_si_bdino/`. Then:

```bash
python tools/run_multiseed.py --single config/bdsl_bdino_temporal_si.yaml --seeds 0 1 2
```

### Step P1.4 — three-way comparison

After Stages B.1+B.2 and Path 1 above are done, the three rows in
`results_final.csv` to compare are:

* `bdsl_pose_temporal_si_seed*`  (MediaPipe pose baseline)
* `bdsl_dino_temporal_si_seed*`  (generic DINOv2)
* `bdsl_bdino_temporal_si_seed*` (Bangla-domain-adapted DINOv2)

For the identity-shortcut measurement, repeat each on the legacy random split
(`data/bdsl/` paths) and report `Top1_SD - Top1_SI` per representation.

## What "novelty" means concretely

1. **First Bangla-specific handshape foundation encoder** (LoRA-tuned DINOv2
   on a ~150k-image aggregated Bangla corpus).
2. **First three-way representation comparison** (pose vs generic vs
   domain-adapted) on a Bangla word-level SLR target.
3. **First quantification of the signer-identity-shortcut narrowing** under
   domain adaptation, on Bangla.
