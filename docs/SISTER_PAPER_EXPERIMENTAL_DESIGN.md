# Sister-Paper Experimental Design

Full experiment / baseline / ablation plan for the sister paper —
*BanglaHandshape: A Rigorous Signer-Independent Benchmark for Still-Image Bangla
SL Handshape Classification* (`RUNBOOK_SISTER_PAPER.md`,
`path3_handshape_benchmark/`, shared lib `bangla_handshape/`). This expands the
runbook's S1–S3 sketch into the table/baseline/ablation slate needed for a
competitive venue.

- **Contribution type:** a *protocol + benchmark*, not a SOTA chase. The claim is
  that Bangla handshape recognition is evaluated under leaky signer-dependent
  splits, and a fair signer-independent (SI) protocol changes the picture.
- **Compute:** ~1–2 GPU-days core, +2–3 for the full ablation set. **Runs on the
  local RTX 8000** (image classification; the HPC-only rule is for the word-level
  program). VRAM is shared with the `resp_env` project — check `nvidia-smi` and
  drop `batch_size` if needed.
- **Datasets:** re-fetch first per `docs/SISTER_PAPER_DATA_REFETCH.md` (they were
  deleted in the storage cleanup and are not on HPC).

---

## 1. Research questions → experiments

| RQ | Question | Delivered by |
|---|---|---|
| RQ1 | Under a *fair* SI protocol, how good are modern backbones vs the inflated SD numbers in the literature? | T1 + T2 |
| RQ2 | How large is the identity shortcut (SD→SI gap)? Is it a one-signer artifact? | T3 + leave-one-user-out |
| RQ3 | Do handshape representations transfer across datasets / label spaces? | T4 |
| RQ4 | What *drives* accuracy — pretraining paradigm, adaptation method, scale, modality? | A1–A6 |

RQ1+RQ2 are the paper's spine (the protocol point); RQ3+RQ4 lift it from a data
note to a benchmark paper.

## 2. Result tables

| Tbl | Content | Metric |
|---|---|---|
| **T1** | Main SI benchmark: encoder rows × dataset columns | Top-1 mean±std (3 seeds) + 95% bootstrap CI |
| **T2** | Ours (SI) vs best **published** number per dataset (all SD) | Δ = the identity-shortcut headline |
| **T3** | SD vs SI on BdSL47 (only source with real user metadata) | gap = Top1(SD) − Top1(SI), predicted ≥10 pp |
| **T4** | Cross-dataset transfer matrix (train A, linear-probe B) | N×N Top-1, `eval_cross_dataset.py` |
| **SF1** | Per-class confusion on best run (49×49, BDSL49) | heatmap |

## 3. Baseline slate (two tiers)

"SOTA" here is mostly **inflated CNN-transfer numbers under SD splits**, so we
compete on two fronts.

### Tier 1 — legacy / prior-art (re-run honestly under SI)
| Baseline | Config | Status |
|---|---|---|
| ResNet-18 from scratch | `configs/cnn_resnet18_scratch.yaml` | ✅ scaffolded |
| ResNet-50 ImageNet fine-tune (what the dataset papers did) | `configs/cnn_resnet50_imagenet.yaml` | ✅ scaffolded |
| Dataset paper's own architecture | — | cite reported SD number in T2 |

### Tier 2 — modern foundation models (the real contribution)
| Baseline | Config | Status |
|---|---|---|
| ImageNet ViT-S linear probe | `configs/probe_imagenet_vit_s.yaml` | ✅ |
| DINOv2-S linear probe | `configs/linear_probe.yaml` | ✅ (existing) |
| DINOv2-S LoRA | `configs/lora.yaml` | ✅ (existing) |
| DINOv2-S full fine-tune | `configs/full_ft.yaml` | ✅ |
| SigLIP-B linear probe (VLM) | `configs/probe_siglip_b.yaml` | ✅ preempts "no VLM" reviewer note |
| MAE-B linear probe (other SSL) | `configs/probe_mae_b.yaml` | ✅ |
| MediaPipe keypoints → MLP (pose vs appearance) | — | ⚠️ TODO (see §6) |

> **Skip CLIP/SigLIP *zero-shot* as a classifier** — the folder labels are
> unverified integers with no reliable Bangla class names, so text-prompt
> zero-shot isn't sound. VLMs enter only as frozen feature extractors (probe).

## 4. Ablation study (one factor at a time, all SI, 3 seeds)

| # | Ablation | Sweep | Configs |
|---|---|---|---|
| A1 | Pretraining paradigm | SSL (DINOv2/MAE) vs supervised (ImageNet) vs VLM (SigLIP), **same scale, linear probe** | `probe_*` + `linear_probe.yaml` |
| A2 | Adaptation method | probe → LoRA(rank 2/4/8/16/32) → full-FT | `linear_probe`→`lora`(edit `lora_rank`)→`full_ft` |
| A3 | Backbone scale | DINOv2 S → B → L | `linear_probe.yaml`, `probe_dinov2_b.yaml`, (+ an L copy) |
| A4 | Data efficiency | k-shot, k ∈ {1,5,10,20,all} | `kshot_template.yaml` (edit `max_train_per_class`) |
| A5 | LoRA placement | attn-only vs mlp-only vs both; rank | copy `lora.yaml`, edit `lora_targets` |
| A6 | Modality | RGB (DINOv2) vs keypoints (MLP) vs fusion | ⚠️ TODO (see §6) |

A1, A2, A4 are the ones reviewers care about. A2/A3/A5 are just config edits on
already-scaffolded code.

**A1 scale caveat:** DINOv2 and ImageNet ViT are available at S; SigLIP/MAE
start at B. For a clean paradigm comparison, read A1 primarily at **ViT-B**
(`probe_dinov2_b` vs `probe_siglip_b` vs `probe_mae_b` vs an ImageNet-B probe),
and use the S row for the scale ablation A3 — don't compare S-DINOv2 directly to
B-SigLIP.

## 5. Protocol & statistics (reviewer armor)

- **3 seeds** everywhere → `results_final.csv` → `tools/summarize_seeds.py --markdown`
  (mean±std + 95% bootstrap CI, already wired; rows are `<Experiment>_<source>_seed<N>`).
- **T3 leave-one-user-out:** don't report the gap at one held-out signer — rerun
  `bdsl47_si.yaml` sweeping `test_users:[k]` for each user id (with a distinct
  held-out val user), then report the gap's **mean±std across users**. This kills
  the "you picked an easy test signer" criticism (runbook §7).
- **Paired significance:** `tools/paired_bootstrap.py` for the key deltas
  (DINOv2-LoRA vs ImageNet-CNN-FT; SD vs SI).
- **No silent class merging:** per-source heads stay disjoint (the design);
  alignment only via a verified `--alignment-json`.
- Fixed preprocessing: Resize 1.15× → CenterCrop → ImageNet norm
  (`train_baseline._build_transforms`).

## 6. Build status — what's runnable now vs TODO

**Scaffolded this pass (validated on CPU with synthetic data):**
- Encoder configs for A1/A3 + the two CNN baselines + full-FT + k-shot template
  + T3 SD/SI pair (11 new/updated configs under `configs/`).
- `bangla_handshape/cnn_baseline.py` — multi-head ResNet-18/34/50 (scratch or
  ImageNet), same `(x,src_idx)→[(src_idx,mask,logits)]` / `features()` contract.
- `train_baseline.py` — `encoder.arch: resnet*` branch; `encoder.full_finetune`;
  `split.force_random` (enables T3-SD); `split.max_train_per_class` (A4 k-shot).

**Two correctness fixes made (both would have invalidated results):**
1. **Per-source attribution.** `MultiHeadLoRADinov2.forward` (and the new CNN)
   now emit the **true source index** in each `(src_idx, mask, logits)` tuple;
   `train_utils.evaluate/multihead_topk` key on it instead of the `enumerate`
   position. Before, an unshuffled/source-ordered val batch for source 3 was
   attributed to source 0 — scrambling every per-source number and CSV row.
2. **T3 was impossible.** BdSL47 was *always* user-disjoint regardless of config,
   so the runbook's "set val_users:[]/test_users:[] for SD" instruction did
   nothing. Added `split.force_random` to actually produce the SD split.

**Still TODO (bigger new modules):**
- **A6 keypoint baseline.** BdSL47 ships per-sample MediaPipe 21-keypoint CSVs
  next to the jpg. A `bangla_handshape/keypoint_baseline.py` should parse those
  CSVs → `(63,)` feature → small MLP, exposing the same multi-head contract, and
  a `configs/keypoint_mlp.yaml`. This is the appearance-vs-pose control (ties to
  the main paper's Option B). Left as a stub because it needs the exact CSV
  column format, which isn't on disk yet (data deleted).
- **LOUO driver.** A small loop over `test_users` for T3 (no new code strictly
  needed — sweep the config field — but a convenience runner + aggregator helps).
- **ResNet-18 L-scale / DINOv2-L config.** Copy `probe_dinov2_b.yaml`, set
  `vit_large_patch14_dinov2.lvd142m`, `batch_size: 32`.

## 7. Run sequence (local)

```bash
conda activate bdsl_graph
cd F:\SLGTformer

# 0. data present? (after re-fetch)
python -c "from bangla_handshape.class_alignment import discover_default; \
[print(s.name, s.num_classes) for s in discover_default('.')]"
python -m pytest tests/test_bangla_handshape_smoke.py -q

# 1. T1 main table — foundation-model rows (3 seeds each)
python -m path3_handshape_benchmark.train_baseline --config path3_handshape_benchmark/configs/linear_probe.yaml       --seeds 0 1 2
python -m path3_handshape_benchmark.train_baseline --config path3_handshape_benchmark/configs/lora.yaml               --seeds 0 1 2
python -m path3_handshape_benchmark.train_baseline --config path3_handshape_benchmark/configs/probe_imagenet_vit_s.yaml --seeds 0 1 2
# ... probe_dinov2_b, probe_siglip_b, probe_mae_b, full_ft

# 1b. Tier-1 CNN baselines
python -m path3_handshape_benchmark.train_baseline --config path3_handshape_benchmark/configs/cnn_resnet18_scratch.yaml   --seeds 0 1 2
python -m path3_handshape_benchmark.train_baseline --config path3_handshape_benchmark/configs/cnn_resnet50_imagenet.yaml  --seeds 0 1 2

# 2. T3 identity-shortcut gap (run BOTH, same encoder)
python -m path3_handshape_benchmark.train_baseline --config path3_handshape_benchmark/configs/bdsl47_si.yaml --seeds 0 1 2
python -m path3_handshape_benchmark.train_baseline --config path3_handshape_benchmark/configs/bdsl47_sd.yaml --seeds 0 1 2

# 3. T4 transfer matrix (after an encoder is trained)
python -m path3_handshape_benchmark.eval_cross_dataset --encoder-dir work_dir/bhc_lora --epoch 5 --seed 0 \
    --lora-rank 8 --lora-targets attn.qkv attn.proj mlp.fc1 mlp.fc2 --output results/S2_transfer_matrix.md

# 4. aggregate everything
python tools/summarize_seeds.py --csv results_final.csv --markdown > results/sister_paper_master.md
```

> Windows: keep `num_workers: 0` in configs if the DataLoader is flaky (CLAUDE.md).
> A single run streams to console; if you want me to launch/monitor a long sweep
> detached, we can, but since you run these yourself in your own terminal the
> earlier console-kill issue doesn't apply.

## 8. Suggested paper order (per runbook §8)

Week 1: T1 (foundation rows) + Tier-1 CNNs → §5.2. Week 2: T3 + LOUO → §5.5.
Week 3: T4 + A1/A2/A4 ablations. Week 4: A6 keypoint control + reviewer-defense
appendix. Sister paper stays ~1 GPU-week vs the main paper's GPU-month.


## 9. Future work — Tier-3: unified handshape taxonomy & LODO

**Decision: deferred to future work.** The unified cross-corpus protocol (unified
label space + leave-one-dataset-out, LODO) is the one contribution that needs a
human in the loop — a native-signer / sign-linguist to verify the canonical
handshape mapping — so it is a deliberate future-work boundary, not a compute
blocker. Cross-corpus generalization is still reported in the present work via the
pairwise transfer matrix (T4, label-agnostic); LODO is its deeper, single-label-space
successor.

**Paper Future-Work paragraph (draft):**

> A natural extension of this benchmark is a *unified* cross-corpus protocol. The
> sources annotate overlapping hand *handshapes* under disjoint, dataset-specific
> label indices; mapping them into a single canonical handshape vocabulary — e.g.
> HamNoSys handshape symbols or the phonological handshape inventory of Bangla Sign
> Language — would enable (i) a single-label-space classifier trained across corpora
> and (ii) a *leave-one-dataset-out* (LODO) evaluation, in which a model trained on
> the other corpora is tested on a fully held-out corpus in the same label space —
> the strongest test of generalization to an unseen recording environment and signer
> population. We provide the complete infrastructure for this: a canonical-alignment
> schema, a DINOv2-prototype cross-source clustering tool that drafts a candidate
> mapping with per-class similarity evidence, and a LODO training/evaluation runner
> gated on a verified alignment. We release a compute-proposed draft alignment
> (55 candidate handshape groups over 178 source classes) as a bootstrap. The
> remaining step — expert verification of the canonical mapping — is left to future
> work, as a credible unified benchmark requires linguistic ground truth rather than
> an automatically-clustered proxy.

**How to resume (repo state, all committed):**
- Tooling: `bangla_handshape/handshape_taxonomy.py` (schema + coverage API),
  `path3_handshape_benchmark/propose_alignment.py` (draft), `run_lodo.py` (LODO),
  `scripts/hpc/slurm_{propose,lodo}.sbatch`, `alignment/README.md`.
- Bootstrap artifacts: `alignment/handshape_alignment.proposed.json`,
  `alignment/handshape_alignment_review.csv`.
- Steps: (1) re-run the proposer sweeping `--sim-threshold` up to 0.90–0.95 — the
  DINOv2 hand-crop prototypes are high-cosine, so the 0.92 draft under-merges across
  sources (only 1 group shared by ≥2 sources ⇒ near-zero LODO coverage today);
  (2) a signer/linguist edits the review CSV (confirm/split/merge, fill handshape
  names); (3) save as `alignment/handshape_alignment.json` with `"verified": true`;
  (4) `sbatch scripts/hpc/slurm_lodo.sbatch`.
