"""T4 driver — train per-source encoders, then build the cross-dataset matrix.

`eval_cross_dataset.py` needs one encoder PER source ("train on A"), but the
headline sweep trains a single JOINT multi-head encoder. This driver closes that
gap:

  1. For each source in the config, train a SINGLE-SOURCE LoRA encoder (reusing
     train_baseline._train_one_seed on an in-memory single-source config). Each
     source's split follows the standard protocol automatically (user-disjoint
     for BdSL47, author-provided test for BSLD_45/BDSL49, else random).
  2. Collect each `encoder_seed<N>_epoch<E>.pt` into a shared `matrix_dir`,
     renamed `encoder_<source>_seed<N>_epoch<E>.pt` — the exact name
     eval_cross_dataset resolves per source.
  3. Invoke eval_cross_dataset for `eval_seed` to write the N×N matrix.

Idempotent: a source/seed whose collected checkpoint already exists is skipped,
so re-running only fills gaps and re-runs the (cheap) eval.

Usage (bdsl_graph):
    python -m path3_handshape_benchmark.run_transfer_matrix \
        --config path3_handshape_benchmark/configs/transfer_matrix.yaml
    # re-draw the matrix from already-collected encoders:
    python -m path3_handshape_benchmark.run_transfer_matrix --config ... --eval-only
"""

from __future__ import annotations

import argparse
import copy
import os
import shutil
import subprocess
import sys

import yaml

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from path3_handshape_benchmark.train_baseline import _train_one_seed


def _ckpt_src(work_dir, seed, epoch):
    return os.path.join(work_dir, f"encoder_seed{seed}_epoch{epoch}.pt")


def _ckpt_dst(matrix_dir, source, seed, epoch):
    return os.path.join(matrix_dir, f"encoder_{source}_seed{seed}_epoch{epoch}.pt")


def _train_and_collect(cfg, present, seeds, epoch, matrix_dir):
    base = cfg.get("Experiment_name", "bhc_xfer")
    os.makedirs(matrix_dir, exist_ok=True)
    for name, root in present.items():
        sub = copy.deepcopy(cfg)
        sub["sources"] = {name: root}
        sub["Experiment_name"] = f"{base}_{name}"
        sub["work_dir"] = f"./work_dir/{base}_{name}"
        for seed in seeds:
            dst = _ckpt_dst(matrix_dir, name, seed, epoch)
            if os.path.exists(dst):
                print(f"[skip] {name} seed{seed}: collected checkpoint exists", flush=True)
                continue
            src = _ckpt_src(sub["work_dir"], seed, epoch)
            if not os.path.exists(src):
                print(f"=== train single-source encoder: {name} seed{seed} ===", flush=True)
                _train_one_seed(sub, seed)
            if not os.path.exists(src):
                print(f"[WARN] expected {src} after training but it is missing; "
                      f"check num_epoch/save_interval", flush=True)
                continue
            shutil.copy2(src, dst)
            print(f"[collect] {os.path.basename(src)} -> {os.path.basename(dst)}", flush=True)


def _run_eval(cfg, epoch, eval_seed, matrix_dir, output):
    enc = cfg["encoder"]
    sp = cfg.get("split", {})
    cmd = [
        sys.executable, "-m", "path3_handshape_benchmark.eval_cross_dataset",
        "--encoder-dir", matrix_dir,
        "--epoch", str(epoch),
        "--seed", str(eval_seed),
        "--timm-name", str(enc.get("timm_name", "vit_small_patch14_dinov2.lvd142m")),
        "--lora-rank", str(int(enc.get("lora_rank", 8))),
        "--lora-alpha", str(float(enc.get("lora_alpha", 16.0))),
        "--image-size", str(int(cfg.get("image_size", 224))),
        "--batch-size", str(int(cfg.get("batch_size", 64))),
        "--num-workers", str(int(cfg.get("num_workers", 0))),
        "--output", output,
    ]
    targets = enc.get("lora_targets") or ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"]
    cmd += ["--lora-targets", *targets]
    cmd += ["--val-users", *[str(u) for u in sp.get("val_users", [4])]]
    cmd += ["--test-users", *[str(u) for u in sp.get("test_users", [5])]]
    print("=== eval_cross_dataset ===\n  " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="path3_handshape_benchmark/configs/transfer_matrix.yaml")
    ap.add_argument("--eval-only", action="store_true",
                    help="skip training/collection; only (re)draw the matrix")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seeds = [int(s) for s in cfg.get("seeds", [0, 1, 2])]
    epoch = int(cfg.get("num_epoch", 50))
    eval_seed = int(cfg.get("eval_seed", seeds[0]))
    matrix_dir = cfg.get("matrix_dir", "./work_dir/bhc_xfer_matrix")
    output = cfg.get("output", "results/T4_transfer_matrix.md")

    present = {n: r for n, r in cfg["sources"].items() if os.path.isdir(r)}
    missing = [n for n in cfg["sources"] if n not in present]
    if missing:
        print(f"[WARN] sources not on disk (skipped): {missing}", flush=True)
    if len(present) < 2:
        sys.exit("need at least 2 sources on disk for a transfer matrix")
    print(f"sources: {list(present)}  seeds={seeds}  eval_seed={eval_seed}  epoch={epoch}")

    if not args.eval_only:
        _train_and_collect(cfg, present, seeds, epoch, matrix_dir)

    have = [n for n in present
            if os.path.exists(_ckpt_dst(matrix_dir, n, eval_seed, epoch))]
    if len(have) < 2:
        sys.exit(f"only {len(have)} collected encoder(s) for eval_seed={eval_seed}; "
                 f"run without --eval-only first")
    _run_eval(cfg, epoch, eval_seed, matrix_dir, output)
    print(f"T4 done -> {output}")


if __name__ == "__main__":
    main()
