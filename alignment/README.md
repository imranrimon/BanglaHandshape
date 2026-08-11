# Tier-3 — Unified handshape taxonomy & LODO

> **STATUS: FUTURE WORK (deferred).** The tooling and a compute-proposed draft
> alignment are complete and committed, but LODO is *not run* in the paper — it is
> gated on expert (native-signer / sign-linguist) verification of the canonical
> mapping. See `docs/SISTER_PAPER_EXPERIMENTAL_DESIGN.md` §9 for the rationale and
> the paper Future-Work paragraph. Cross-corpus generalization is reported in the
> paper via the pairwise transfer matrix (T4); LODO is its deeper successor. Do NOT
> report LODO numbers off an unverified alignment. Current draft (sim-threshold 0.92)
> under-merges across sources, so re-run the proposer at 0.90–0.95 before verifying.

This directory holds the **cross-corpus handshape alignment** that unifies the
five sources' disjoint local label spaces into ONE canonical handshape
vocabulary. It is the opt-in unification the benchmark anticipates
(`bangla_handshape/class_alignment.py`): baselines stay multi-head/disjoint
until a *verified* alignment exists here.

## Why

BdSL47-digits, BdSL-MNIST, BSLD_45, BDSL49 photograph overlapping **handshapes**
under different folder names/indices. A verified mapping to a canonical set lets
us (a) train a single unified-space classifier, (b) report a handshape-coverage
table across datasets, and (c) run **leave-one-dataset-out (LODO)**: train on
four corpora, test on the fifth in the same label space — true cross-corpus
generalization (new camera/background/signer pool).

## Workflow

1. **Propose** (compute-assisted, minutes of GPU):
   ```
   python -m path3_handshape_benchmark.propose_alignment \
       --out alignment/handshape_alignment.proposed.json \
       --review-csv alignment/handshape_alignment_review.csv
   ```
   Computes a DINOv2 feature prototype per (source, class), groups prototypes
   across sources by cosine similarity, and drafts a canonical id per group.
   Emits the proposed JSON + a human-review CSV with nearest-cross-source
   evidence for each class.

2. **Verify** (human — a native signer / SL linguist; this is the contribution):
   Open the review CSV, confirm/split/merge groups, fill `name`/`hamnosys`,
   then save the corrected mapping as `alignment/handshape_alignment.json` and
   set `"verified": true`. Only verified alignments may back reported numbers.

3. **Use**:
   ```
   python -m path3_handshape_benchmark.run_lodo \
       --alignment-json alignment/handshape_alignment.json --seeds 0 1 2 \
       --results-csv results/bhc_lodo.csv
   # or: sbatch scripts/hpc/slurm_lodo.sbatch
   ```

## Schema (`handshape_alignment.json`)

```json
{
  "version": "0.1",
  "verified": false,
  "canonical": { "0": {"name": "A-hand", "hamnosys": "", "notes": ""} },
  "map": { "bdsl47_digits": {"Sign 0": 0}, "bdsl_mnist": {"0": 12} }
}
```

- Canonical ids are contiguous `"0".."K-1"`.
- `map` keys are the on-disk class-folder names (== `SourceSpec.class_to_idx`).
- A local class absent from `map` is **unmapped** — excluded from the unified /
  LODO space and counted in the coverage report.

See `handshape_alignment.example.json` for a concrete (illustrative, unverified)
skeleton, and `bangla_handshape/handshape_taxonomy.py` for the loader/coverage API.
