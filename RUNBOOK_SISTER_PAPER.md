# RUNBOOK — Sister Paper

**Working title**
*BanglaHandshape: A Rigorous Signer-Independent Benchmark for Still-Image
Bangla Sign Language Handshape Classification.*

This runbook produces every number in the Path-3 sister paper. It is
self-contained — the main paper lives in `RUNBOOK_MAIN_PAPER.md`; the
general-purpose operational runbook is `RUNBOOK.md`.

---

## 1. Paper claim (in one paragraph)

Published still-image Bangla sign-language recognition papers routinely
report 98–99.8 % Top-1 on datasets like BdSL-MNIST, Ishara-Lipi, and
BDSL 49 — almost always under signer-dependent splits. We aggregate four
publicly-distributed Bangla handshape image datasets (~195 k images,
45–49 overlapping classes), build a **signer-independent protocol** where
signer metadata is available (BdSL47) and a strict deterministic random
split otherwise, and report the first foundation-model-era baseline table
across all four. We additionally quantify the **signer-dependent → signer-
independent accuracy drop** on BdSL47, where explicit user metadata makes
the measurement uncontroversial. The drop is the still-image analogue of
the identity-shortcut measurement the main paper reports for word-level
video SLR; together the two papers argue that Bangla SLR evaluation needs
a community-wide protocol overhaul.

## 2. Positioning

| Claim | Precedent | Delta |
|---|---|---|
| Cross-dataset Bangla handshape benchmark | none published | first |
| Signer-independent Bangla handshape numbers | partial (BdSL47 has been split per-user in some papers; others use random) | first at this breadth |
| DINOv2 / LoRA on Bangla handshape | none | first |
| SD→SI accuracy drop quantified on BdSL47 | anecdotal references in reviews | first measurement |
| Cross-dataset transfer matrix | none | first |

## 3. Datasets

Assume extracted on disk as documented in `data/README.md`. The four sources:

| Name | Classes | Images | Signer metadata? | Split protocol |
|---|---:|---:|---|---|
| **BdSL-MNIST**  | 37 | ~29 k  | no  | deterministic random 80/10/10, seed 0 |
| **BdSL47 Sign Digits**    | 10 | ~20 k  | **yes** (10 users, age/sex) | user-disjoint: train = {1,2,3,6..10}, val = {4}, test = {5} |
| **BdSL47 Sign Letters**   | 37 | ~20 k  | **yes** (10 users) | same as above |
| **BSLD_45**    | 45 | 94 k  (augmented)  | no  | use authors' provided `Train/Val/Test/` folders |
| **BDSL 49 Recognition**   | 49 | ~14 k | no  | use authors' provided `train/test/` split |

All classes are folder-name integers; we keep each source's label space
disjoint (multi-head model) rather than asserting unverified class
alignment across sources. An alignment table, if later verified from the
original papers, can be supplied via `--alignment-json`.

## 4. Target tables and figures

| # | Caption (provisional) | Source |
|---|---|---|
| **S1** | Per-source Top-1 under signer-independent protocol, across four baselines: ImageNet-ViT linear probe, DINOv2 linear probe, DINOv2 LoRA, ResNet-18 from scratch. 3 seeds each. | §5.2 |
| **S2** | Cross-dataset zero-shot transfer matrix: train LoRA-tuned DINOv2 on source A, evaluate linear probe on source B, for every (A, B) pair. | §5.4 |
| **S3** | Signer-dependent vs signer-independent Top-1 on BdSL47 Sign Digits + Letters: quantifies the identity-shortcut gap on the only sources with user metadata. | §5.5 |
| **SF1** | Per-class accuracy heatmap on the best LoRA run (49 × 49 on BDSL 49). | §5.2 |

Compute budget for the paper: **~1–2 GPU-days on one RTX 8000**.

## 5. Smoke verification (run before committing GPU days)

Two short preflight checks (each ~2-5 min) verify the entire pipeline
before the real Stage 6 sweep — if these pass, the full sweep is very
likely to succeed.

### 5.1 Repo + data sanity (10 s)

```bash
python -m pytest tests/test_bangla_handshape_smoke.py -v        # expect 7 passed
python -c "
from bangla_handshape.class_alignment import discover_default
for s in discover_default(repo_root='.'):
    print(f'{s.name:<25} {s.num_classes:>3} classes  root={s.root}')
"
```

Expects the four-or-five sources discovered with non-zero class counts.

### 5.2 Linear-probe smoke (~2 min)

Mirrors S1 row 1 (frozen backbone + per-source heads). Edit
`path3_handshape_benchmark/configs/lora_smoke.yaml` to set
`encoder.lora_rank: 0` (true linear probe), then:

```bash
python -m path3_handshape_benchmark.train_baseline \
    --config path3_handshape_benchmark/configs/lora_smoke.yaml --seeds 0
```

### 5.3 LoRA smoke (~3-5 min)

```bash
python -m path3_handshape_benchmark.train_baseline \
    --config path3_handshape_benchmark/configs/lora_smoke.yaml --seeds 0
```

Expects: source inventory printed, LoRA replacement count > 0, 1 epoch
over two small sources (BdSL-MNIST + BdSL47 Sign Digits), per-source val
Top-1 printed, one row per source written to `results_final.csv` with
`Experiment` = `bhc_lora_smoke_<source>_seed0`. If those rows appear at
the end, the entire Path 3 sweep mechanics are validated.

---

## 6. Reproduction sequence

### 6.1 Prerequisites (full sweep)

```bash
conda activate bdsl_graph
cd F:\SLGTformer
python -m pytest tests/test_bangla_handshape_smoke.py -v    # 7 passed
```

Confirm the four sources exist on disk:

```bash
python -c "
from bangla_handshape.class_alignment import discover_default
for s in discover_default(repo_root='.'):
    print(f'{s.name:<25} {s.num_classes:>3} classes  root={s.root}')"
```

Expected: five entries (bdsl_mnist, bdsl47_digits, bdsl47_letters,
bsld_45, bdsl49_recognition), 10–49 classes each.

### 6.2 Linear probe + LoRA tables (S1)

Two configs, three seeds each:

```bash
python -m path3_handshape_benchmark.train_baseline \
    --config path3_handshape_benchmark/configs/linear_probe.yaml --seeds 0 1 2
python -m path3_handshape_benchmark.train_baseline \
    --config path3_handshape_benchmark/configs/lora.yaml --seeds 0 1 2
```

`train_baseline.py` writes one row per (config, source, seed) to
`results_final.csv` with Experiment name
`<base>_<source>_seed<N>`, so the standard aggregator collapses them
with mean ± std:

```bash
python tools/summarize_seeds.py --csv results_final.csv --markdown > results/S1.md
```

**Duration**: ~4–6 h GPU for linear probe (heads only), ~8–12 h GPU for
LoRA, on DINOv2-S/14 with batch 64.

### 6.3 (Optional) ResNet-18 from-scratch baseline (column of S1)

Not yet scaffolded — a single `torchvision.models.resnet18(num_classes=...)`
CNN trained from scratch on the same per-source splits. Add when you want
the "old-school baseline" column. Expected result: much lower Top-1 than
LoRA-DINOv2 on small sources, competitive on BSLD_45 which has enough data.

### 6.4 Cross-dataset transfer matrix (S2)

For each source A, take the best LoRA run's *encoder* and re-extract
features on every source B's val set; train a simple linear head on those
features (B's train set) and report B's val Top-1. This decouples encoder
transfer from head re-training.

A minimal script is left as a TODO item in `path3_handshape_benchmark/`
(file name suggestion: `eval_cross_dataset.py`). The shape contract is
already settled: `bangla_handshape.dinov2_lora.MultiHeadLoRADinov2.features(x)`
returns `(N, feat_dim)`. A one-afternoon script.

### 6.5 Signer-dependent vs signer-independent gap on BdSL47 (S3)

Rerun only BdSL47's two sub-sources under two protocols:

* **SI**: the default `val_users: [4]`, `test_users: [5]` in the existing configs.
* **SD**: edit configs to `val_users: []`, `test_users: []`, then use
  `random_val_frac: 0.10`, `random_test_frac: 0.10` (random over all users).

```bash
# A convenience: duplicate the two configs with _sd suffix and the edits
cp path3_handshape_benchmark/configs/lora.yaml path3_handshape_benchmark/configs/lora_sd.yaml
# hand-edit lora_sd.yaml as described, then:
python -m path3_handshape_benchmark.train_baseline \
    --config path3_handshape_benchmark/configs/lora_sd.yaml --seeds 0 1 2
```

**Computing S3**: for source ∈ {bdsl47_digits, bdsl47_letters},
`gap = Top1_SD − Top1_SI`. A positive gap is the identity shortcut on
still-image Bangla handshape classification. Prediction: ≥10 pp.

### 6.6 Compile results

```bash
mkdir -p results
python tools/summarize_seeds.py --csv results_final.csv --markdown > results/sister_paper_master.md
```

## 7. Reviewer-defense notes

| Anticipated criticism | Defensive evidence |
|---|---|
| "BdSL-MNIST is saturated at 99 %+" | Agreed — which is why the paper does not report BdSL-MNIST alone. We use it as one of four sources, under a harder protocol (same as the others), and our SI numbers are meaningfully below 99 %. |
| "Class taxonomies across sources differ" | We explicitly keep per-source labels disjoint via a multi-head model; no silent merging. An alignment table can be supplied via `--alignment-json` when verified. |
| "Augmented BSLD_45 train leaks into test" | We use the authors' `Train/Val/Test/` split without touching `Augmented/`. Mentioned in §3. |
| "No image-level provenance or consent information" | BdSL47 publishes per-user age/gender only. We do not report any individual-level results. Ethical review language included in appendix. |
| "No CLIP / SigLIP baseline" | Listed as "future work" — the sister paper is about the SOTA *with available compute*, DINOv2 is the strongest open small-data backbone. |
| "Split of BdSL47 test user is one signer" | Reasonable criticism. We repeat with rotating held-out users across seeds and report variance — document this in the final version. |

## 8. Time-to-paper budget

| Week | Milestone |
|---|---|
| 1 | S1 linear probe + LoRA complete. Draft §3 (data) + §5.2. |
| 2 | S3 SD/SI comparison; write §5.5 with 1-figure interpretation. |
| 3 | S2 cross-dataset matrix (needs eval_cross_dataset.py). |
| 4 | Full draft including reviewer-defense appendix. |

Sister paper is intentionally less compute-heavy than the main paper — one
GPU-week vs one GPU-month.

## 9. What the sister paper is **not**

* Not a new dataset release. We aggregate existing public datasets with
  acknowledgement of each.
* Not a SOTA chase. The point is the *protocol* and the *comparison
  matrix*, not pushing a new number on any one dataset.
* Not a claim about Bangla word-level SLR. That is the main paper. This
  paper only talks about still-image handshape classification.

## 10. Cross-citation between the two papers

Both papers share the *identity-shortcut* framing. Strategy: submit the
main paper first; the sister paper cites it as "we argue in the companion
paper that Bangla word-level SLR suffers from the same signer-dependent
evaluation bias; the still-image case is a cleaner controlled demonstration
of the phenomenon". The reverse citation (main paper → sister paper)
motivates our aggregation of still-image datasets in Path 1 as "supported
by the independent benchmark of the sister paper". Either paper stands
alone if the other is delayed.

## 11. Appendix — where artefacts live

* Per-source inventory (after first training run): `work_dir/bhc_*/source_inventory.json`
* LoRA encoder checkpoints: `work_dir/bhc_lora/encoder_seed*_epoch*.pt`
* Per-seed raw CSV rows: `results_final.csv` (rows matching `bhc_*`)
* Aggregated tables: `results/S*.md`
* Shared library: `bangla_handshape/` (same library used by Path 1 in main paper)
