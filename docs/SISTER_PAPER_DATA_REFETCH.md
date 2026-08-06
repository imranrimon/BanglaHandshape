# Sister-Paper Dataset Re-Fetch Checklist

The four still-image Bangla handshape datasets that feed the sister paper
(`RUNBOOK_SISTER_PAPER.md`, Path 3 / `path3_handshape_benchmark/`) were removed
from local disk during the 2026-08 storage cleanup and were **never uploaded to
HPC** (the DollySods upload was word-level video/pose data only). All four are
public and re-downloadable. The **code is unaffected** — only the raw images are
gone. This file is the recipe to get them back.

> The Path-3 benchmark reads exactly these four sources; Ishara-Lipi is **not**
> a code source (the empty `ishara_lipi*` folders were unused) so it is optional.
> Source-of-truth for the paths below: `bangla_handshape/class_alignment.py`
> (`DEFAULT_SOURCES`).

## The four sources

| Source (code key) | Classes | Download | Extract to (exact path the code expects) |
|---|---:|---|---|
| **BdSL-MNIST** (`bdsl_mnist`) | 37 | [Mendeley `6f2wm5p3vf/1`](https://data.mendeley.com/datasets/6f2wm5p3vf/1) | `data/BdSL-MNIST/<class>/*.png` |
| **BdSL47 Sign Digits** (`bdsl47_digits`) | 10 | [Mendeley `pbb3w3f92y/3`](https://data.mendeley.com/datasets/pbb3w3f92y/3) · mirror [Dryad](https://doi.org/10.5061/dryad.1vhhmgqwk) | `data/BdSL47/Bangla Sign Language Dataset - Sign Digits/<User>/<Sign>/…` |
| **BdSL47 Sign Letters** (`bdsl47_letters`) | 37 | same download as Digits | `data/BdSL47/Bangla Sign Language Dataset - Sign Letters/<User>/<Sign>/…` |
| **BSLD_45** (`bsld_45`) | 45 | [Kaggle `rayeed045/bangla-sign-language-dataset`](https://www.kaggle.com/datasets/rayeed045/bangla-sign-language-dataset) *(verify — see note)* | `data/BSLD_45/Train/<class>/*.jpg` (+ `Val/`, `Test/`) |
| **BDSL 49 Recognition** (`bdsl49_recognition`) | 49 | [Mendeley `k5yk4j8z8s/6`](https://data.mendeley.com/datasets/k5yk4j8z8s/6) · [arXiv 2208.06827](https://arxiv.org/abs/2208.06827) | `data/bdsl49_extracted/Recognition_1/Recognition_1/train/<class>/…` (+ `test/`) |

Total ≈ 195 k images across the four. Compute for the whole paper is ~1–2 GPU-days.

## Gotchas (these will bite if skipped)

1. **BdSL47 folder rename.** The current Mendeley release names the alphabet
   folder `Bangla Sign Language Dataset - Sign Alphabets`, but the code expects
   `… - Sign Letters`. After extracting, rename it:
   `mv "Bangla Sign Language Dataset - Sign Alphabets" "Bangla Sign Language Dataset - Sign Letters"`.
   The Digits folder already matches. (BdSL47 also ships per-sample CSVs of
   MediaPipe keypoints alongside the raw jpg — the image loader uses the jpg;
   the CSVs are harmless.)
2. **BDSL 49 double-nesting.** The Mendeley archive contains several task zips;
   the sister paper uses only the **Recognition** task. Extracting it yields the
   doubled `Recognition_1/Recognition_1/{train,test}` path — that exact nesting is
   what the code reads, so don't flatten it.
3. **BSLD_45 — verify identity.** The Kaggle link above (by `rayeed045`, the
   BdSL47 author) is the most likely source, but confirm on download that it is
   the **45-class** set with author-provided `Train/Val/Test/` folders and ~94.5 k
   augmented jpg. If it differs, search Kaggle for the 45-class Bangla handshape
   set and adjust `DEFAULT_SOURCES` if the folder name changes.

## Where to put it

- **If running Path 3 on HPC** (recommended, consistent with the cut-over):
  download straight onto DollySods — e.g. `kaggle datasets download` on a compute
  node for the Kaggle one, `wget` the Mendeley "Download All" zips — and extract
  under `/scratch/mh00145/SLGTformer/data/` (which is symlinked to `~/SLGTformer/data`).
- **If running locally:** extract under `F:\SLGTformer\data\` at the paths above.

## Verify before spending GPU

```bash
conda activate bdsl_graph
python -c "
from bangla_handshape.class_alignment import discover_default
for s in discover_default(repo_root='.'):
    print(f'{s.name:<20} {s.num_classes:>3} classes  root={s.root}')"
# expect: bdsl_mnist 37, bdsl47_digits 10, bdsl47_letters 37, bsld_45 45, bdsl49_recognition 49
python -m pytest tests/test_bangla_handshape_smoke.py -v      # expect 7 passed
```

If `discover_default` lists all five entries with the class counts above, the
data is correctly placed and the Path-3 smoke sweep (`RUNBOOK_SISTER_PAPER.md`
§5) will run.
