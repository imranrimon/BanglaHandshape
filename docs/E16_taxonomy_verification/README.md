# E16 — Unified handshape taxonomy: expert-verification package

**Reviewer point (§29).** *"A unified cross-corpus label space needs a real handshape
taxonomy, verified by a native signer / SL-linguist, before leave-one-dataset-out
(LODO) numbers mean anything."*

The compute half is done (`propose_alignment.py` → a DINOv2-prototype cross-source
clustering of all 178 source classes into candidate canonical handshape groups). This
package is the **human half made cheap**: one workbook the expert edits, seeded with
the E15 Bengali labels, plus the exact protocol to turn edits into a verified
alignment that unlocks `run_lodo.py`.

> **Status: future work, gated on this verification.** The paper reports cross-corpus
> generalization via the *label-agnostic* transfer matrix (T4). LODO is its deeper,
> single-label-space successor and is **not** reported until the alignment below is
> marked `"verified": true` by a qualified annotator. Do not report LODO off an
> unverified map.

---

## The one artifact to edit

`alignment/verification_workbook.csv` (built by `tools/build_taxonomy_workbook.py`).
One row per source class, candidate groups contiguous, cross-source groups first
(those are the ones that make LODO non-trivial). BdSL47 rows already carry their
`proposed_bengali` grapheme from the E15 draft.

**Columns you fill:** `decision`, `canonical_name`, `hamnosys`, `notes`.
**Columns for context (read-only):** `canonical_id`, `group_size`, `group_sources`,
`auto_name`, `source`, `class_folder`, `proposed_bengali`, `n_images`,
`top/second_cross_source_match` + cosine, `needs_review`.

### Decision codes

| `decision` | meaning |
|---|---|
| `confirm` | this class belongs in its current `canonical_id` group |
| `split` | this class does **not** belong — give it a fresh handshape (leave `canonical_id`, we renumber) |
| `merge_into:<id>` | this class's whole group should fold into canonical group `<id>` |
| `reassign:<id>` | move just this class into existing group `<id>` |
| `drop` | exclude this class from the unified space (ambiguous / non-canonical) |
| *(blank)* | undecided — treated as `drop` until resolved |

Set `canonical_name` (e.g. a HamNoSys handshape name or a Bangla phonological label)
and, ideally, `hamnosys` once per confirmed group.

---

## Protocol (≈ the same effort as reading a 178-row sheet once)

1. Work top-down. The first block is the single group that already spans ≥2 sources —
   confirm/split it first; cross-source groups are what LODO trains and tests on.
2. For each group, judge the members by the images (`n_images` per class on disk) and
   the cross-source evidence columns. Use the decision codes above.
3. Fill `canonical_name` / `hamnosys` per confirmed group.
4. Save as `alignment/handshape_alignment.verified.csv`, then compile it:
   ```bash
   python -m tools.compile_verified_alignment \
       --workbook alignment/handshape_alignment.verified.csv \
       --out alignment/handshape_alignment.json      # sets "verified": true
   ```
   *(`compile_verified_alignment.py` is the small deterministic reducer that turns the
   decision column into the contiguous-id `handshape_alignment.json` schema; add it
   when verification begins — it has no data/GPU dependency.)*
5. Run LODO:
   ```bash
   python -m path3_handshape_benchmark.run_lodo \
       --alignment-json alignment/handshape_alignment.json --seeds 0 1 2 \
       --results-csv results/bhc_lodo.csv          # or: sbatch scripts/hpc/slurm_lodo.sbatch
   ```

---

## ⚠️ Compute prerequisite before verification is worthwhile

The current workbook is built from the **0.92-threshold** proposal, which
**under-merges**: only **1** candidate group spans ≥2 sources, so LODO coverage would
be near-zero. Two things must happen first, both needing all five sources on disk
(BdSL47, BdSL-MNIST, BSLD_45, BDSL49, RSBdSL38 — only the last two are currently
present):

1. **Re-fetch** the missing sources (`docs/SISTER_PAPER_DATA_REFETCH.md`).
2. **Re-run the proposer sweeping the threshold down** so groups actually merge across
   sources (the DINOv2 hand-crop prototypes sit at high cosine):
   ```bash
   for T in 0.90 0.88 0.86; do
     python -m path3_handshape_benchmark.propose_alignment \
        --sim-threshold $T \
        --out alignment/handshape_alignment.proposed_$T.json \
        --review-csv alignment/handshape_alignment_review_$T.csv
   done
   # then rebuild the workbook off the threshold whose cross-source coverage is best:
   python -m tools.build_taxonomy_workbook \
        --review-csv alignment/handshape_alignment_review_0.90.csv \
        --proposed-json alignment/handshape_alignment.proposed_0.90.json
   ```
   `scripts/hpc/slurm_propose.sbatch` runs this on the cluster (fill `<PARTITION>`).
   Pick the threshold that maximises `group_sources ≥ 2` without collapsing distinct
   handshapes — the `build_taxonomy_workbook` summary line reports that count.

Only after a threshold with real cross-source coverage is the human verification a
good use of the expert's time.

---

## How this lands in the paper

- **Future-Work → Results** if verification completes: a `T5` LODO table (train on 4
  corpora, test on the 5th, unified handshape space) — the strongest external-validity
  claim, with a per-source coverage report from `run_lodo.py`.
- If it stays future work: the released, seeded workbook + proposer + LODO runner are
  the reproducible bootstrap the Future-Work paragraph already promises.
