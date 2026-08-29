#!/usr/bin/env python3
"""Extract + normalise the BdSL47 Dryad .rar into the layout the loader expects.

Why a dedicated tool (not the generic ``refetch_data.sh``): the BdSL47 release has
*wildly inconsistent* internal folder naming, especially in the Sign Letters set,
where a naive "flatten ``Input Images`` up" misses most users. The actual variants
found in the archive (per user):

  class folder : ``Sign 00`` | ``sign 00`` | ``sign00``          (case + zero-pad vary)
  input subdir : ``Input Images`` | ``input images`` | ``inut images`` (typo)
                 ``input imagess`` | ``input iimages`` | ``input image``
                 ``Input Images - sign 00`` | ``Input Images - Sign 00``
  output subdir: ``Output Images`` | ``Output Images - Sign_00`` | ``... - sign 00``

The loader (`handshape_dataset._enumerate_with_user`) reads images **directly** at
``<root>/User XX/Sign N/<image>`` and `class_alignment.discover_source` enumerates
the *distinct* class-folder names across users -- so inconsistent names inflate the
class count (Letters came out 111 classes instead of 37) and buried input folders
drop whole users (users 1,2,3,10 vanished). This normaliser fixes both:

  * class folder  -> canonical ``Sign %02d`` (parsed from ``sign[ _]*<int>``)
  * every image whose path does NOT contain "output" (case-insensitive) is lifted
    into that canonical class folder; "output" skeleton overlays + CSVs are dropped.

Idempotent: re-extracts cleanly into a scratch temp, writes ``data/BdSL47/<dataset>/``,
then verifies with discover_default + enumerate_source. Expected result:
  bdsl47_digits  : 10 classes, 10 users, 10000 images
  bdsl47_letters : 37 classes, 10 users, 37000 images

Usage:
  python data_prep/normalize_bdsl47.py                 # extract .rar + normalise
  python data_prep/normalize_bdsl47.py --from-extracted data/_bdsl47_tmp
  RAR=data/_downloads/BdSL47.rar DATA_DIR=data python data_prep/normalize_bdsl47.py
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

_IMG = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
_SIGN_RE = re.compile(r"sign[\s_]*0*(\d+)", re.I)   # "Sign 00"/"sign_5"/"sign26" -> int
_USER_RE = re.compile(r"^User\s*\d+", re.I)
ARCHIVE_ROOT = "BdSL47 - A Complete Dataset of Signs in Bangla Sign Language (BdSL)"
DATASETS = ("Bangla Sign Language Dataset - Sign Digits",
            "Bangla Sign Language Dataset - Sign Letters")


def log(m): print(f"[bdsl47] {m}", flush=True)


def _is_img(name): return os.path.splitext(name)[1].lower() in _IMG


def _extract(rar: Path, tmp: Path):
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    log(f"extracting {rar} -> {tmp} (bsdtar)")
    subprocess.run(["bsdtar", "-xf", str(rar), "-C", str(tmp)], check=True)


def _normalise_dataset(src_root: Path, dst_root: Path):
    """src_root/<User..>/<class..>/<...>/<img>  ->  dst_root/<User..>/Sign NN/<img>."""
    if dst_root.exists():
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True)
    n_img = n_cls_bad = 0
    users = set()
    for user in sorted(os.listdir(src_root)):
        udir = src_root / user
        if not udir.is_dir() or not _USER_RE.match(user):
            continue
        users.add(user)
        (dst_root / user).mkdir(exist_ok=True)
        for cls in sorted(os.listdir(udir)):
            cdir = udir / cls
            if not cdir.is_dir():
                continue
            m = _SIGN_RE.search(cls)
            if not m:
                n_cls_bad += 1
                continue
            canon = f"Sign {int(m.group(1)):02d}"
            out = dst_root / user / canon
            out.mkdir(exist_ok=True)
            # lift every non-"output" image anywhere under this class folder
            for dirpath, _dn, files in os.walk(cdir):
                if "output" in dirpath.lower():
                    continue
                for fn in files:
                    if not _is_img(fn):
                        continue
                    dst = out / fn
                    if dst.exists():                     # keep both on name clash
                        stem, ext = os.path.splitext(fn)
                        k = 1
                        while (out / f"{stem}__{k}{ext}").exists():
                            k += 1
                        dst = out / f"{stem}__{k}{ext}"
                    shutil.move(os.path.join(dirpath, fn), str(dst))
                    n_img += 1
    log(f"  {dst_root.name}: {n_img} images, {len(users)} users"
        + (f"  (skipped {n_cls_bad} unpar-seable class dirs)" if n_cls_bad else ""))
    return n_img, len(users)


def _verify(repo_root="."):
    from banglahandshape.class_alignment import discover_default
    from banglahandshape.handshape_dataset import enumerate_source
    log("verify (discover_default):")
    for s in discover_default(repo_root=repo_root):
        tag = ""
        if s.name.startswith("bdsl47"):
            e = enumerate_source(s)
            tag = f"  images={len(e)} users={sorted({u for _, _, u in e})}"
        log(f"  {s.name:<20} {s.num_classes:>3} classes{tag}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rar", default=os.environ.get("RAR", "data/_downloads/BdSL47.rar"))
    ap.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "data"))
    ap.add_argument("--from-extracted", default=None,
                    help="skip extraction; point at an already-extracted tree")
    ap.add_argument("--keep-tmp", action="store_true")
    args = ap.parse_args()
    sys.path.insert(0, os.getcwd())

    data_dir = Path(args.data_dir)
    tmp = Path(args.from_extracted) if args.from_extracted else data_dir / "_bdsl47_tmp"
    if not args.from_extracted:
        _extract(Path(args.rar), tmp)

    base = tmp / ARCHIVE_ROOT
    if not base.is_dir():                                 # some releases omit the wrapper
        cands = [p for p in tmp.iterdir() if p.is_dir() and "BdSL47" in p.name]
        base = cands[0] if cands else tmp

    dest = data_dir / "BdSL47"
    for ds in DATASETS:
        src = base / ds
        if not src.is_dir():
            log(f"  [warn] missing dataset dir: {src}")
            continue
        _normalise_dataset(src, dest / ds)

    if not args.from_extracted and not args.keep_tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    _verify(".")
    log("NORMALIZE_DONE")


if __name__ == "__main__":
    main()
