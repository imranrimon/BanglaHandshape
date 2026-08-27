# Datasheet for the BanglaHandshape Aggregated Benchmark

This datasheet follows the "Datasheets for Datasets" template of Gebru et al.
(*Communications of the ACM*, 2021). It documents the **aggregated
signer-independent (SI) still-image Bangla Sign Language (BdSL) handshape
benchmark** assembled in this repository. The benchmark is a *protocol +
aggregation* over five pre-existing public datasets: we do **not** collect,
own, or redistribute the underlying images. We release code, fixed
train/val/test splits, and a cross-corpus class-alignment schema; users
re-fetch the third-party images themselves.

Source-of-truth for the source list and on-disk paths is
`bangla_handshape/class_alignment.py` (`DEFAULT_SOURCES`); the re-fetch recipe
is `docs/SISTER_PAPER_DATA_REFETCH.md`; the experimental design is
`docs/SISTER_PAPER_EXPERIMENTAL_DESIGN.md`.

---

## Motivation

**For what purpose was the dataset created?**
To provide the first *signer-independent* (SI), foundation-model-era benchmark
for still-image Bangla handshape classification, and to quantify the
**signer-dependent -> signer-independent accuracy drop** (the "identity
shortcut"). Published Bangla handshape papers report 98-99.8% Top-1, almost
always under *signer-dependent* splits in which the same signer's frames appear
in both train and test. The aggregated benchmark re-evaluates modern backbones
(DINOv2 / LoRA and others) under a fair SI protocol and makes the shortcut
explicit.

**Who created it and on whose behalf?**
The benchmark aggregation, protocol, splits, and code were created by the
authors of this repository (the "sister paper" to a separate word-level video
SLR project in the parent directory). The **underlying image datasets were
created by their respective third-party authors** (see Composition and
Collection process). Exact author/institution attribution for the aggregation
is TODO/VERIFY before submission.

**Who funded the creation?**
TODO/VERIFY (funding for the aggregation work). The underlying datasets were
funded by their respective original authors/institutions.

---

## Composition

**What do the instances represent?**
Each instance is a still RGB image of a single hand (or hands) forming a Bangla
Sign Language *handshape* (a digit or an alphabet/letter sign), labeled with a
per-source class index.

**How many instances are there?**
The benchmark aggregates **five sources** (four distinct public datasets;
BdSL47 contributes two label spaces). Per-source class counts and signer
metadata:

| Source (code key) | # Classes | Signer / user IDs? | SI split available |
|---|---:|---|---|
| BdSL47 Sign Digits (`bdsl47_digits`) | 10 | Yes (per-user folders) | Yes — user-disjoint |
| BdSL47 Sign Letters (`bdsl47_letters`) | 37 | Yes (per-user folders) | Yes — user-disjoint |
| BDSL49 Recognition (`bdsl49_recognition`) | 49 | No | No — author-provided train/test |
| BSLD_45 (`bsld_45`) | 45 | No | No — provided Train/Val/Test |
| BdSL-MNIST (`bdsl_mnist`) | 37 | No | No — fixed random 80/10/10 |

Total image count: approximately 195k images across the four datasets (see the
re-fetch checklist); exact per-source counts are not fixed in the repo — see
repo / re-fetched data. `discover_default` skips any missing source silently, so
the effective count depends on what the user has re-fetched.

**Only BdSL47 supports a true user-disjoint signer-independent split.** The
other three datasets ship no user/signer metadata, so their "SI" is best-effort
(author-provided held-out test for BDSL49 and BSLD_45; a fixed random split for
BdSL-MNIST) and is **not strictly signer-independent**. These sources are
reported for completeness only; all identity-shortcut analysis is done on
BdSL47.

**What data does each instance consist of?**
Raw RGB images ("as is"). BdSL47 also ships per-sample MediaPipe keypoint CSVs
next to each JPG; the image loader uses only the JPG. For the pose (A6) modality
control, keypoints are (re-)extracted with MediaPipe into a separate `.npz`
cache (see Preprocessing).

**Is there a label for each instance?**
Yes — an integer class folder name per source. Label spaces are kept **disjoint
per source** (multi-head), never silently merged. A unified cross-corpus label
space is opt-in only via a verified `--alignment-json` (the `alignment/` schema
is currently a compute-proposed draft, unverified; see `alignment/README.md`).

**Is any information missing?**
Yes. Folder names are integer-stringified with no reliable Bangla character
dictionary, so cross-source semantic identity of like-named folders is
uncertain (hence the disjoint-by-default policy). **Signer demographics
(age, gender, handedness, region, disability status) are unknown / not
provided** for all sources. Only BdSL47 has any user identifiers at all, and
even there no demographic attributes are given.

**Are relationships between instances made explicit?**
Within BdSL47, images are grouped by user folder (`User XX/Sign N/...`); this
grouping is what the SI split exploits. No cross-source instance relationships
are asserted.

**Recommended data splits?**
Yes — the SI protocol splits are fixed and released (see Preprocessing). These
are the ground-truth splits for the benchmark; do not substitute random splits
for headline numbers.

**Errors, noise, redundancies?**
Not systematically audited. BSLD_45 images ship with a **hand-skeleton overlay
already drawn on the image** (see Preprocessing / Uses caveats). Some sources
are known to be augmentation-heavy; author-provided test folders are used where
available to avoid augmentation/random-split leakage.

**Self-contained or relying on external resources?**
**Relies on external resources.** The images are third-party public datasets and
are **not** included in this repository (gitignored; deleted in a storage
cleanup and re-fetchable). This repo contains only code, splits, and the
alignment schema.

**Confidential / sensitive content?**
The images depict human hands and, in some datasets, faces/body/background of
signers. See Biases and the ethics discussion; treat as potentially
identity-revealing.

---

## Collection process

**How was the data acquired?**
We did not collect any images. Each source is a pre-existing public dataset
downloaded from its original distribution channel. Download locations and exact
extract paths are in `docs/SISTER_PAPER_DATA_REFETCH.md`. In summary
(exact citations marked TODO/VERIFY):

- **BdSL47 (Sign Digits + Sign Letters)** — public release (Mendeley
  `pbb3w3f92y/3`, Dryad mirror `10.5061/dryad.1vhhmgqwk`). Citation: TODO/VERIFY.
- **BDSL49 Recognition** — public release (Mendeley `k5yk4j8z8s/6`,
  arXiv:2208.06827). Citation: TODO/VERIFY.
- **BSLD_45** — 45-class set with a `Train/Val/Test/` layout; canonical source
  is not the `rayeed045` Kaggle slug (that slug is BdSL47). Correct source
  TBD from the original BSLD_45 paper/authors. Citation: TODO/VERIFY.
- **BdSL-MNIST** — public release (Mendeley `6f2wm5p3vf/1`). Citation: TODO/VERIFY.

**Who collected the underlying images and how were they compensated?**
The original dataset authors; unknown to us. TODO/VERIFY per source.

**Over what timeframe were the data collected?**
Unknown to us; refers to each original dataset's collection period. TODO/VERIFY.

**Were the individuals depicted notified / did they consent?**
Unknown to us. Consent and IRB/ethics status are governed by each original
dataset's terms. We re-use only publicly released datasets under their stated
terms; see Distribution and the ethics discussion.

**Ethical review?**
No separate ethical review was conducted for the aggregation. Any review
attaches to the original datasets (unknown to us). TODO/VERIFY.

---

## Preprocessing / cleaning / labeling

**Was any preprocessing done?**
Yes, at training/eval time (the raw images are left untouched on disk):

- **Image transform.** Resize (1.15x) then centre-crop to 224x224, then ToTensor
  + ImageNet normalization. There is no random augmentation in the eval path, so
  a pre-resized uint8 image cache (`work_dir/_img_cache`) is precomputed for
  speed; it is bit-identical to the naive path and falls back to decoding the
  original file on a cache miss.
- **Signer-independent split logic** (`train_baseline._train_one_seed`):
  1. **BdSL47** -> user-disjoint SI split via `split.val_users` /
     `split.test_users` (the only sources with user IDs). A random 80/10/10
     signer-dependent (SD) split is produced **only** when
     `split.force_random: True` — this is the mechanism that makes the SD-vs-SI
     shortcut measurement possible; SD is never a headline result on its own.
  2. **BDSL49, BSLD_45** -> train on the author-provided `train/`, evaluate on
     the author-provided `test/` (avoids augmentation / random-split leakage);
     still signer-dependent (no user IDs).
  3. **BdSL-MNIST** -> deterministic random 80/10/10 seeded by `split.seed`.
  `split.max_train_per_class` optionally caps train items per class (the k-shot
  data-efficiency ablation).
- **Keypoint (pose) extraction.** For the pose modality control, MediaPipe
  Hands landmarks are extracted in an **isolated `mp_kp` conda env** (mediapipe
  1.0.0 Tasks API `HandLandmarker`) deliberately kept out of the training env so
  it cannot perturb a running campaign. It writes a per-source `.npz` cache keyed
  by absolute image path; the trainer reads the cache and never imports
  mediapipe.

**BSLD_45 drawn-skeleton caveat.** BSLD_45 images carry a **hand-skeleton
overlay already drawn on the image**, so MediaPipe hand re-detection is
approximately 0%. BSLD_45 is therefore **excluded from all keypoint/pose (A6)
experiments**; sources below a minimum detected-keypoint threshold are skipped,
and the per-source detection rate is itself a reported benchmark statistic.

**Was the raw data saved?**
The raw images are the original third-party files; we do not modify them.
Preprocessing outputs (caches) are gitignored and regenerable.

**Is the preprocessing software available?**
Yes — all preprocessing (`bangla_handshape/`, `preprocessing/`,
`path3_handshape_benchmark/`) is released in this repository.

---

## Uses

**What (has / can) the dataset be used for?**
Signer-independent still-image Bangla handshape classification benchmarking:
comparing pretraining paradigms, adaptation methods (linear probe / LoRA /
full fine-tune), backbone scale, data efficiency, LoRA placement, and RGB-vs-pose
modality; and quantifying/mitigating the identity shortcut.

**What should it *not* be used for?**
- **Not for signer / person identification, re-identification, or any
  biometric / surveillance use.** The very existence of a signer-identity
  shortcut means signer-identifiable features are present; we explicitly
  discourage building identity classifiers from these images or from models
  trained on them.
- Not as evidence that Bangla handshape recognition is "solved" — the SD numbers
  in the literature are inflated by the shortcut.
- Not for deployment claims without reporting per-signer variance (see Biases):
  accuracy varies substantially across individual signers.

**Is there anything about composition or collection that might affect future
uses?**
Yes — unknown signer demographics, few signers, capture-condition confounds, and
Bangla-only scope (see Biases). Users should report SI numbers and per-signer
variance, and should not merge label spaces without a verified alignment.

---

## Distribution

**Will the dataset be distributed to third parties?**
The **images are not redistributed by us.** They remain third-party public
datasets; users re-fetch them from the original sources per
`docs/SISTER_PAPER_DATA_REFETCH.md`. We release only: the benchmark **code**, the
fixed **SI splits / protocol**, and the cross-corpus **class-alignment schema**
(compute-proposed draft, unverified).

**How is our released material distributed, and under what license?**
Via this public code repository. Repository license: TODO/VERIFY. The underlying
image datasets retain their **original licenses / terms of use**, which vary by
source and which users must comply with when re-fetching; license heterogeneity
across sources is a known limitation.

**Any IP, ToU, export controls, or fees on the underlying data?**
Governed by each original dataset's terms (unknown/heterogeneous to us).
TODO/VERIFY per source before redistribution of any derived artifact.

---

## Maintenance

**Who is supporting / maintaining the benchmark?**
The repository authors. Contact: TODO/VERIFY.

**How can it be updated / who contributes errata?**
Via the public repository (issues / pull requests). The re-fetch checklist and
`class_alignment.py` `DEFAULT_SOURCES` are the source of truth for source paths;
corrections to citations, the BSLD_45 source, or the alignment schema flow
through the repo.

**Will older versions be supported?**
Splits are fixed and version-controlled in the repo; the SI protocol is stable.
The cross-corpus alignment is explicitly future work and will only back reported
numbers once expert-verified (`alignment/README.md`).

**If others extend/build on it?**
Encouraged via the repo, under the SI-reporting norms above (report SI + per-signer
variance; keep label spaces disjoint absent a verified alignment).

---

## Biases (explicit)

- **Signer demographics largely unknown.** No source provides age, gender,
  handedness, skin tone, regional, or disability attributes; even BdSL47 gives
  only opaque user IDs. Demographic representativeness cannot be assessed and is
  almost certainly skewed.
- **Very limited number of signers.** Only BdSL47 has user identifiers at all
  (a small user pool), so per-signer variance is large and estimated from few
  people; the other datasets have unknown, likely small, signer counts.
- **Capture-condition confounds (the identity shortcut).** Background, lighting,
  camera, and per-signer hand morphology correlate with signer identity, so a
  classifier can exploit these instead of the handshape. This confound is the
  central object of study in this benchmark, not an incidental flaw — but it also
  means models trained here can encode signer-identifiable, non-handshape
  information.
- **Per-signer accuracy variance is large** (a fairness/equity concern): under
  leave-one-user-out, appearance models collapse on specific signers while pose
  models are far more stable. Deployment on an under-represented signer may be
  much worse than the mean suggests.
- **Bangla-only scope.** The benchmark covers Bangla Sign Language handshapes
  only; generality to other sign languages / handshape inventories is untested
  and is future work.
- **Label-space uncertainty.** Integer folder labels without a verified Bangla
  character dictionary; cross-source class identity is not assumed.
