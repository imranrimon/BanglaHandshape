#!/bin/bash -l
# =====================================================================
# Re-fetch the four still-image Bangla handshape datasets for the sister
# paper (docs/SISTER_PAPER_DATA_REFETCH.md). Idempotent: each source is
# skipped if its target folder already exists, so re-running only fills gaps.
#
# All four sources use a STAGING model: you drop the downloaded archives into
# $STAGE and the script extracts + normalises them (handles the "Sign Alphabets"
# -> "Sign Letters" rename and the BDSL49 double-nesting). The three Mendeley
# sources have no stable direct URL (click-through). BSLD_45 is NOT auto-pulled
# either: its usual Kaggle slug (rayeed045/bangla-sign-language-dataset) is
# actually BdSL47, not the 45-class set — set BSLD45_KAGGLE=<owner/slug> only if
# you have verified a correct Kaggle source.
#
# Usage (from repo root, on a DollySods compute node or locally):
#     bash scripts/refetch_data.sh              # extract from $STAGE + verify
#     DATA_DIR=data STAGE=data/_downloads bash scripts/refetch_data.sh
#
# Env:
#   DATA_DIR       where the code reads datasets from (default: ./data)
#   STAGE          where you place the downloaded archives (default: $DATA_DIR/_downloads)
#   PY             python with the bdsl_graph env (default: use `python` on PATH)
#   BSLD45_KAGGLE  optional owner/slug for a VERIFIED 45-class BSLD_45 Kaggle source
#                  (unset by default — the common rayeed045 slug is BdSL47, not this)
# =====================================================================
set -uo pipefail

DATA_DIR="${DATA_DIR:-data}"
STAGE="${STAGE:-$DATA_DIR/_downloads}"
PY="${PY:-python}"
mkdir -p "$DATA_DIR" "$STAGE"

say()  { printf '\033[1;36m[refetch]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[refetch]\033[0m %s\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }

for tool in unzip; do
  have "$tool" || warn "missing '$tool' — extraction of some archives will fail (module load / conda install it)"
done

# find the first archive in $STAGE matching any of the given globs
find_archive() {
  local g
  for g in "$@"; do
    local hit; hit=$(ls -1 "$STAGE"/$g 2>/dev/null | head -n1 || true)
    [ -n "$hit" ] && { echo "$hit"; return 0; }
  done
  return 1
}

extract_into() {  # $1=archive  $2=dest_dir
  local arc="$1" dest="$2"
  mkdir -p "$dest"
  case "$arc" in
    *.zip)            unzip -q -o "$arc" -d "$dest" ;;
    *.tar.gz|*.tgz)   tar -xzf "$arc" -C "$dest" ;;
    *.tar)            tar -xf  "$arc" -C "$dest" ;;
    *) warn "don't know how to extract $arc"; return 1 ;;
  esac
}

# ---------------------------------------------------------------------
# 1. BdSL-MNIST  (Mendeley 6f2wm5p3vf/1)  -> data/BdSL-MNIST/<class>/*.png
# ---------------------------------------------------------------------
fetch_bdsl_mnist() {
  local dest="$DATA_DIR/BdSL-MNIST"
  [ -d "$dest" ] && { say "BdSL-MNIST present, skip"; return; }
  local arc; arc=$(find_archive 'BdSL-MNIST*.zip' '6f2wm5p3vf*.zip' '*MNIST*.zip') || {
    warn "BdSL-MNIST archive not in $STAGE."
    warn "  Download 'Download All' from https://data.mendeley.com/datasets/6f2wm5p3vf/1"
    warn "  and place the .zip in $STAGE, then re-run."; return; }
  say "extracting BdSL-MNIST from $(basename "$arc")"
  extract_into "$arc" "$dest.tmp"
  # flatten a single wrapping folder if the zip nests one
  local inner; inner=$(find "$dest.tmp" -maxdepth 2 -type d -name '*MNIST*' | head -n1)
  mv "${inner:-$dest.tmp}" "$dest" 2>/dev/null || mv "$dest.tmp" "$dest"
  rm -rf "$dest.tmp"
}

# ---------------------------------------------------------------------
# 2/3. BdSL47 Sign Digits + Sign Letters  (Mendeley pbb3w3f92y/3)
#   -> data/BdSL47/Bangla Sign Language Dataset - Sign {Digits,Letters}/<User>/<Sign>/...
#   GOTCHA: current release names the alphabet folder "... - Sign Alphabets";
#           the code expects "... - Sign Letters". We rename it.
# ---------------------------------------------------------------------
fetch_bdsl47() {
  local base="$DATA_DIR/BdSL47"
  local digits="$base/Bangla Sign Language Dataset - Sign Digits"
  local letters="$base/Bangla Sign Language Dataset - Sign Letters"
  if [ -d "$digits" ] && [ -d "$letters" ]; then say "BdSL47 present, skip"; return; fi
  local arc; arc=$(find_archive 'BdSL47*.zip' 'pbb3w3f92y*.zip' '*Sign*Language*.zip') || {
    warn "BdSL47 archive not in $STAGE."
    warn "  Download from https://data.mendeley.com/datasets/pbb3w3f92y/3 (mirror: doi.org/10.5061/dryad.1vhhmgqwk)"
    warn "  and place the .zip in $STAGE, then re-run."; return; }
  say "extracting BdSL47 from $(basename "$arc")"
  mkdir -p "$base"; extract_into "$arc" "$base"
  # rename Alphabets -> Letters if that's how it extracted
  local alpha="$base/Bangla Sign Language Dataset - Sign Alphabets"
  [ -d "$alpha" ] && [ ! -d "$letters" ] && { say "renaming 'Sign Alphabets' -> 'Sign Letters'"; mv "$alpha" "$letters"; }
  # some releases nest one wrapper dir; try to locate the two sub-datasets
  if [ ! -d "$digits" ]; then
    local found; found=$(find "$base" -maxdepth 3 -type d -name '*Sign Digits*' | head -n1)
    [ -n "$found" ] && say "found Digits at $found (leave in place; adjust DEFAULT_SOURCES if path differs)"
  fi
}

# ---------------------------------------------------------------------
# 4. BSLD_45  (45-class handshape set with author-provided Train/Val/Test)
#   -> data/BSLD_45/{Train,Val,Test}/<class>/*.jpg
#   The source is NOT the Kaggle set `rayeed045/bangla-sign-language-dataset`:
#   verified 2026-08 that slug is actually BdSL47 (user-organized Sign Digits/
#   Letters + per-sample MediaPipe CSVs), NOT the 45-class BSLD_45. So we do not
#   auto-pull any default slug. Supply the correct archive by staging it in
#   $STAGE, or set BSLD45_KAGGLE=<owner/slug> to a Kaggle source you've verified.
# ---------------------------------------------------------------------
fetch_bsld45() {
  local dest="$DATA_DIR/BSLD_45"
  [ -d "$dest/Train" ] && { say "BSLD_45 present, skip"; return; }
  local arc
  arc=$(find_archive 'BSLD*45*.zip' 'BSLD_45*.zip' 'bsld*45*.zip') || true
  if [ -z "${arc:-}" ] && [ -n "${BSLD45_KAGGLE:-}" ] && have kaggle; then
    say "downloading BSLD_45 via kaggle CLI ($BSLD45_KAGGLE)"
    kaggle datasets download -d "$BSLD45_KAGGLE" -p "$STAGE" || \
      warn "kaggle download failed (check ~/.kaggle/kaggle.json credentials)"
    arc=$(find_archive "$(basename "$BSLD45_KAGGLE")*.zip") || true
  fi
  if [ -z "${arc:-}" ]; then
    warn "BSLD_45 archive not found in $STAGE."
    warn "  The 45-class set is NOT rayeed045/bangla-sign-language-dataset (that is BdSL47)."
    warn "  Stage the correct 45-class archive (Train/Val/Test/<class>/*.jpg) in $STAGE,"
    warn "  or set BSLD45_KAGGLE=<owner/slug> to a verified Kaggle source and re-run."; return
  fi
  say "extracting BSLD_45 from $(basename "$arc")"
  extract_into "$arc" "$dest.tmp"
  # Guard: refuse a BdSL47-shaped archive (User*/ dirs or per-sample CSVs) — that
  # is the exact mislabel above; populating data/BSLD_45 with it would corrupt the
  # benchmark (wrong class space / wrong split).
  if find "$dest.tmp" -maxdepth 4 -type d -iname 'User *' -print -quit | grep -q . \
     || find "$dest.tmp" -name '*.csv' -print -quit | grep -q .; then
    warn "staged archive looks like BdSL47 (User*/ folders or .csv files), NOT the"
    warn "  45-class BSLD_45 — refusing to populate data/BSLD_45. Fix the source."
    rm -rf "$dest.tmp"; return
  fi
  local tr; tr=$(find "$dest.tmp" -maxdepth 3 -type d -name 'Train' | head -n1)
  if [ -z "$tr" ]; then
    warn "no Train/ folder inside the archive — not the expected BSLD_45 layout; leaving in $dest.tmp for inspection"; return
  fi
  mkdir -p "$dest"; mv "$(dirname "$tr")"/* "$dest"/
  rm -rf "$dest.tmp"
}

# ---------------------------------------------------------------------
# 5. BDSL 49 Recognition  (Mendeley k5yk4j8z8s/6)
#   -> data/bdsl49_extracted/Recognition_1/Recognition_1/{train,test}/<class>/...
#   GOTCHA: keep the DOUBLED Recognition_1/Recognition_1 nesting — the code reads it.
# ---------------------------------------------------------------------
fetch_bdsl49() {
  local dest="$DATA_DIR/bdsl49_extracted"
  [ -d "$dest/Recognition_1/Recognition_1/train" ] && { say "BDSL49 present, skip"; return; }
  local arc; arc=$(find_archive 'k5yk4j8z8s*.zip' '*Recognition*.zip' 'BDSL*49*.zip') || {
    warn "BDSL49 archive not in $STAGE."
    warn "  Download the Recognition task from https://data.mendeley.com/datasets/k5yk4j8z8s/6"
    warn "  and place the .zip in $STAGE, then re-run. (Do NOT flatten the double nesting.)"; return; }
  say "extracting BDSL49 from $(basename "$arc")"
  extract_into "$arc" "$dest"
  if [ ! -d "$dest/Recognition_1/Recognition_1/train" ]; then
    local t; t=$(find "$dest" -type d -path '*Recognition*train' | head -n1)
    [ -n "$t" ] && say "note: train found at $t — if the path differs from the expected double-nesting, update DEFAULT_SOURCES."
  fi
}

say "DATA_DIR=$DATA_DIR   STAGE=$STAGE"
fetch_bdsl_mnist
fetch_bdsl47
fetch_bsld45
fetch_bdsl49

# ---------------------------------------------------------------------
# Verify: discover_default should list the sources with correct class counts.
# ---------------------------------------------------------------------
say "verifying with discover_default (expect: bdsl_mnist 37, bdsl47_digits 10, bdsl47_letters 37, bsld_45 45, bdsl49_recognition 49)"
"$PY" - <<'PYEOF'
from bangla_handshape.class_alignment import discover_default
srcs = discover_default(repo_root=".")
if not srcs:
    print("  [!] no sources discovered — check DATA_DIR paths against DEFAULT_SOURCES")
for s in srcs:
    print(f"  {s.name:<20} {s.num_classes:>3} classes  root={s.root}")
PYEOF
say "done. Next: python -m pytest tests/test_bangla_handshape_smoke.py -q"
