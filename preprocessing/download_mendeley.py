#!/usr/bin/env python3
"""Download BdSL-MNIST and BDSL49 from Mendeley Data via the public API, then
extract + normalise into the paths ``class_alignment.DEFAULT_SOURCES`` expects.

Why this exists
---------------
``scripts/refetch_data.sh`` is *staging-based*: it only extracts archives you have
already downloaded by hand, because the Mendeley web UI is click-through. But
Mendeley's *public* API (``data.mendeley.com/public-api/...``) exposes per-file
``download_url`` values that need **no authentication** (verified 2026-08). So the two
Mendeley image sources can, in fact, be fetched head-less:

  * BdSL-MNIST (6f2wm5p3vf v1) -> one zip, ``BDSL resized 64x64.zip`` (~216 MB)
  * BDSL49     (k5yk4j8z8s v6) -> ``Recognition_1.zip`` (~1.03 GB) is the only archive
    the image benchmark needs; ``Detection_*.zip`` are the object-detection task.

(BdSL47's images are NOT here -- its Mendeley record only hosts a Readme; the 2.83 GB
image ``.rar`` lives on Dryad behind an OAuth-gated API, so it still needs a manual
browser download. See EXPERIMENT_PLAN_REVIEWER.md / HPC_SISTER_PAPER.md.)

This downloader is idempotent: a completed archive or an already-populated destination
is skipped, and partial downloads resume via HTTP Range.

Usage
-----
    python preprocessing/download_mendeley.py                    # bdsl_mnist + bdsl49
    python preprocessing/download_mendeley.py --only bdsl_mnist
    python preprocessing/download_mendeley.py --only bdsl49 --all-bdsl49   # all 10 zips
    python preprocessing/download_mendeley.py --no-extract       # stage archives only
    DATA_DIR=data STAGE=data/_downloads python preprocessing/download_mendeley.py

After it finishes you can sanity-check with:
    python -c "from bangla_handshape.class_alignment import discover_default; \
[print(s.name, s.num_classes) for s in discover_default('.')]"
"""
from __future__ import annotations
import argparse
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import requests

try:
    from tqdm import tqdm
except Exception:  # tqdm optional
    tqdm = None

API = "https://data.mendeley.com/public-api/datasets/{id}/files?folder_id=root&version={ver}"

# Per-source config. ``want`` = archive filenames to fetch (None = all root files).
# ``normalise`` picks how the extracted tree is relocated into DATA_DIR/<dest>.
SOURCES = {
    "bdsl_mnist": dict(
        id="6f2wm5p3vf", version=1,
        want=["BDSL resized 64x64.zip"],
        dest="BdSL-MNIST",
        normalise="classdirs",   # move the folder full of integer-named class dirs
    ),
    "bdsl49": dict(
        id="k5yk4j8z8s", version=6,
        # The 49-class recognition task ships split across two archives
        # (Recognition_1.zip = classes 0-22, Recognition_2.zip = 23-48); we need
        # both to reconstruct the full 49-class space. --all-bdsl49 also pulls the
        # 7 Detection_*.zip (object-detection task; not used by this benchmark).
        want=["Recognition_1.zip", "Recognition_2.zip"],
        dest="bdsl49_extracted",
        normalise="bdsl49",                    # merge -> Recognition_1/Recognition_1/{train,test}
    ),
}


def log(msg: str) -> None:
    print(f"[mendeley] {msg}", flush=True)


def list_root_files(ds_id: str, version: int):
    """Return [{filename, size, url}] for a dataset version's root files."""
    r = requests.get(API.format(id=ds_id, ver=version), timeout=60)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict):
        raise RuntimeError(f"Mendeley API error for {ds_id} v{version}: {data}")
    out = []
    for f in data:
        cd = f.get("content_details") or {}
        url = cd.get("download_url")
        if url:
            out.append(dict(filename=f.get("filename"), size=f.get("size", 0), url=url))
    return out


def download(url: str, out_path: Path, expected: int | None = None, chunk: int = 1 << 20) -> Path:
    """Stream ``url`` to ``out_path`` with resume + size verification."""
    out_path = Path(out_path)
    part = out_path.with_name(out_path.name + ".part")
    if out_path.exists() and (expected is None or out_path.stat().st_size == expected):
        log(f"  [skip] {out_path.name} already complete ({out_path.stat().st_size} B)")
        return out_path

    existing = part.stat().st_size if part.exists() else 0
    if expected is not None and existing >= expected:  # stale oversized part
        existing = 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    with requests.get(url, headers=headers, stream=True, timeout=120) as r:
        if r.status_code == 416:  # already fully downloaded
            part.rename(out_path)
            log(f"  [done] {out_path.name} (server: range complete)")
            return out_path
        if existing and r.status_code == 200:  # server ignored Range -> restart
            existing = 0
        r.raise_for_status()
        total = expected or (int(r.headers.get("Content-Length", 0)) + existing) or None
        mode = "ab" if existing else "wb"
        bar = tqdm(total=total, initial=existing, unit="B", unit_scale=True,
                   desc=out_path.name) if tqdm else None
        with open(part, mode) as fh:
            for block in r.iter_content(chunk_size=chunk):
                if not block:
                    continue
                fh.write(block)
                if bar:
                    bar.update(len(block))
        if bar:
            bar.close()

    got = part.stat().st_size
    if expected is not None and got != expected:
        raise IOError(f"size mismatch for {out_path.name}: {got} != {expected} "
                      f"(leaving .part for resume)")
    part.rename(out_path)
    log(f"  [ok] {out_path.name} ({got} B)")
    return out_path


# --------------------------------------------------------------------------- #
# extraction / normalisation
# --------------------------------------------------------------------------- #
def _extract(zip_path: Path, into: Path) -> Path:
    into.mkdir(parents=True, exist_ok=True)
    log(f"  extracting {zip_path.name} -> {into}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(into)
    return into


def _dir_with_most_int_children(root: Path):
    best, best_n = None, 0
    for d, subdirs, _ in os.walk(root):
        n = sum(1 for s in subdirs if s.isdigit())
        if n > best_n:
            best, best_n = Path(d), n
    return best, best_n


def _all_dirs_containing(root: Path, names):
    names = set(names)
    hits = []
    for d, subdirs, _ in os.walk(root):
        if names.issubset(set(subdirs)):
            hits.append(Path(d))
    return hits


def _replace_dir(dest: Path, src: Path):
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dest))


def normalise_classdirs(tmp: Path, dest: Path):
    """BdSL-MNIST: dest/<class>/*.png (37 integer-named class folders)."""
    root, n = _dir_with_most_int_children(tmp)
    if not root or n < 2:
        raise RuntimeError(f"could not find class folders under {tmp}")
    log(f"  class root: {root} ({n} integer-named class dirs)")
    _replace_dir(dest, root)


def normalise_bdsl49(tmp: Path, dest: Path):
    """BDSL49: union every archive's train/test into
    dest/Recognition_1/Recognition_1/{train,test}/<class>/*.jpg.

    The recognition task is split across Recognition_1.zip (classes 0-22) and
    Recognition_2.zip (23-48), so we merge their class folders. Disjoint class
    names => a clean 49-class union; any collision is flagged (would signal
    per-archive local numbering that needs a real remap, not a blind merge)."""
    roots = _all_dirs_containing(tmp, {"train", "test"})
    if not roots:
        raise RuntimeError(f"could not find any train/ + test/ root under {tmp}")
    target = dest / "Recognition_1" / "Recognition_1"
    for split in ("train", "test"):
        (target / split).mkdir(parents=True, exist_ok=True)
    collisions = []
    for root in sorted(roots, key=lambda p: p.name):
        log(f"  merging {root.name}/ -> {target}")
        for split in ("train", "test"):
            sp = root / split
            if not sp.is_dir():
                continue
            for cls in sp.iterdir():
                if not cls.is_dir():
                    continue
                dst = target / split / cls.name
                if dst.exists():
                    collisions.append(f"{split}/{cls.name}")
                    for item in cls.iterdir():
                        shutil.move(str(item), str(dst / item.name))
                else:
                    shutil.move(str(cls), str(dst))
    n_tr = sum(1 for p in (target / "train").iterdir() if p.is_dir())
    log(f"  BDSL49 merged: {n_tr} train classes at {target}")
    if collisions:
        log(f"  WARNING: {len(collisions)} class-name collision(s) across archives "
            f"(possible per-archive local numbering) e.g. {collisions[:5]} -- "
            f"verify labels before trusting the 49-class space!")


NORMALISERS = {"classdirs": normalise_classdirs, "bdsl49": normalise_bdsl49}


# --------------------------------------------------------------------------- #
def process_source(key: str, cfg: dict, data_dir: Path, stage: Path,
                   want_all: bool, do_extract: bool):
    log(f"=== {key} (Mendeley {cfg['id']} v{cfg['version']}) ===")
    dest = data_dir / cfg["dest"]
    if dest.exists() and any(dest.iterdir()):
        log(f"  destination {dest} already populated -- skipping "
            f"(delete it to re-download)")
        return

    files = list_root_files(cfg["id"], cfg["version"])
    want = None if want_all else set(cfg.get("want") or [])
    picked = [f for f in files if (want is None or f["filename"] in want)]
    if want and not picked:
        raise RuntimeError(f"wanted {want} but root files are "
                           f"{[f['filename'] for f in files]}")
    total = sum(f["size"] for f in picked)
    log(f"  {len(picked)} archive(s), ~{total/1e6:.0f} MB")

    stage.mkdir(parents=True, exist_ok=True)
    archives = []
    for f in picked:
        safe = f["filename"].replace(" ", "_")
        archives.append(download(f["url"], stage / safe, expected=f["size"]))

    if not do_extract:
        log("  --no-extract: archives staged, skipping extraction")
        return

    with tempfile.TemporaryDirectory(dir=str(data_dir)) as td:
        tmp = Path(td)
        for arc in archives:
            _extract(arc, tmp)
        NORMALISERS[cfg["normalise"]](tmp, dest)
    log(f"  normalised into {dest}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=list(SOURCES),
                    help="subset of sources (default: all)")
    ap.add_argument("--all-bdsl49", action="store_true",
                    help="fetch all 10 BDSL49 zips (Detection_* + Recognition_*), "
                         "not just Recognition_1")
    ap.add_argument("--no-extract", action="store_true",
                    help="download archives into STAGE but do not extract/normalise")
    ap.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "data"))
    ap.add_argument("--stage", default=os.environ.get("STAGE"))
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    stage = Path(args.stage) if args.stage else data_dir / "_downloads"
    keys = args.only or list(SOURCES)

    failures = []
    for key in keys:
        try:
            process_source(key, SOURCES[key], data_dir, stage,
                           want_all=(key == "bdsl49" and args.all_bdsl49),
                           do_extract=not args.no_extract)
        except Exception as e:  # keep going; report at the end
            log(f"  [FAIL] {key}: {e}")
            failures.append(key)

    log("done.")
    log("verify: python -c \"from bangla_handshape.class_alignment import "
        "discover_default; [print(s.name, s.num_classes) for s in discover_default('.')]\"")
    if failures:
        log(f"FAILED: {failures}")
        sys.exit(1)


if __name__ == "__main__":
    main()
