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

1. Work top-down. The first blocks are the 2 groups that already span ≥2 sources —
   confirm/split them first; cross-source groups are what LODO trains and tests on.
   Then extend coverage: most same-handshape/different-dataset pairs sit just below
   the cosine threshold (domain shift), so the expert's `merge_into`/`reassign`
   decisions are where the real cross-corpus taxonomy is built.
2. For each group, judge the members by the images (`n_images` per class on disk) and
   the cross-source evidence columns. Use the decision codes above.
3. Fill `canonical_name` / `hamnosys` per confirmed group.
4. Save as `alignment/handshape_alignment.verified.csv`, then compile it:
   ```bash
   python -m tools.compile_verified_alignment \
       --workbook alignment/handshape_alignment.verified.csv \
       --out alignment/handshape_alignment.json      # sets "verified": true
   ```
   *(`tools/compile_verified_alignment.py` is the small deterministic reducer that
   turns the decision column into the contiguous-id `handshape_alignment.json`
   schema. It is scaffolded and self-tested. Pass `--mark-verified` only after a
   qualified annotator has reviewed; without it the JSON stays `verified:false` and
   `run_lodo` warns. It has no data/GPU dependency.)*
5. Run LODO:
   ```bash
   python -m path3_handshape_benchmark.run_lodo \
       --alignment-json alignment/handshape_alignment.json --seeds 0 1 2 \
       --results-csv results/bhc_lodo.csv          # or: sbatch scripts/hpc/slurm_lodo.sbatch
   ```

---

## Compute prerequisite: DONE (4-source threshold sweep, 2026-08-12)

Three of the five sources were re-fetched (BdSL-MNIST from Mendeley; BdSL47
Digits+Letters from the Dryad mirror), joining the existing RSBdSL38 + BDSL49 →
**4 datasets / 5 source entries** on disk (`bdsl_mnist`, `bdsl47_digits`,
`bdsl47_letters`, `bdsl49_recognition`, `rsbdsl38`). **BSLD\_45 remains
unobtainable** (no verified public source; the `rayeed045` slug is BdSL47), so the
sweep runs on 4 datasets, not 5.

The proposer was re-run as a parallel array (`scripts/hpc/slurm_propose_sweep.sbatch`)
over thresholds 0.90 / 0.88 / 0.86 (145 DINOv2-B prototypes across the 5 source
entries). Cross-source coverage:

| threshold | canonical groups K | groups spanning ≥2 sources |
|---|---|---|
| **0.90** | 21 | **2** |
| 0.88 | 14 | 2 |
| 0.86 | 11 | 1 |

`alignment/verification_workbook.csv` is now rebuilt off the **0.90** proposal
(finest grouping with peak cross-source coverage): 145 members, 21 candidate groups.

**Honest finding — visual clustering under-merges across corpora.** Even with 4
datasets and aggressive thresholds, at most **2** handshape groups span ≥2 sources,
and 137/145 classes are flagged `needs_review`. Frozen DINOv2 hand-crop prototypes
are dominated by per-dataset domain shift (capture conditions, background, image
style) rather than handshape identity, so same-handshape/different-dataset pairs sit
below cosine 0.86. **The bottleneck is therefore expert linguistic verification, not
compute or threshold** — a native-signer/SL-linguist must supply the cross-corpus
`merge_into`/`reassign` decisions the visual proposer cannot. The workbook's 2
cross-source seeds are the starting point; the expert extends them. Until that
verification exists, LODO stays future work (do not report it off the unverified,
visually-under-merged map).

---

## How this lands in the paper

- **Future-Work → Results** if verification completes: a `T5` LODO table (train on 4
  corpora, test on the 5th, unified handshape space) — the strongest external-validity
  claim, with a per-source coverage report from `run_lodo.py`.
- If it stays future work: the released, seeded workbook + proposer + LODO runner are
  the reproducible bootstrap the Future-Work paragraph already promises.
