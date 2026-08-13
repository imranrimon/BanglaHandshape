"""T4 helper: train ONE (source, seed) single-source encoder and collect it into
the shared matrix_dir — the parallelizable unit of run_transfer_matrix's serial
_train_and_collect. Driven by a SLURM array (one task per source*seed), so the 12
encoders train concurrently instead of back-to-back. After the array finishes, run
`run_transfer_matrix --eval-only` to build the N*N matrix from the collected .pt's.

    python -m path3_handshape_benchmark.run_transfer_train_one --task <ARRAY_TASK_ID>

Task index -> (source, seed): src_i, seed_i = divmod(task, len(seeds)); source is the
src_i-th ON-DISK source in config order (identical ordering to run_transfer_matrix,
so eval-only resolves every collected encoder).
"""
import argparse
import copy
import os
import shutil
import sys

import yaml

from path3_handshape_benchmark.run_transfer_matrix import _ckpt_src, _ckpt_dst
from path3_handshape_benchmark.train_baseline import _train_one_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="path3_handshape_benchmark/configs/transfer_matrix.yaml")
    ap.add_argument("--task", type=int, required=True, help="SLURM array task id")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    seeds = [int(s) for s in cfg.get("seeds", [0, 1, 2])]
    epoch = int(cfg.get("num_epoch", 50))
    matrix_dir = cfg.get("matrix_dir", "./work_dir/bhc_xfer_matrix")
    base = cfg.get("Experiment_name", "bhc_xfer")

    present = {n: r for n, r in cfg["sources"].items() if os.path.isdir(r)}
    names = list(present)                       # config order == run_transfer_matrix
    src_i, seed_i = divmod(args.task, len(seeds))
    if src_i >= len(names):
        print(f"[task {args.task}] no work (only {len(names)} sources x {len(seeds)} seeds)")
        return
    name, seed = names[src_i], seeds[seed_i]
    print(f"[task {args.task}] source={name} seed={seed} epoch={epoch}", flush=True)

    sub = copy.deepcopy(cfg)
    sub["sources"] = {name: present[name]}
    sub["Experiment_name"] = f"{base}_{name}"
    sub["work_dir"] = f"./work_dir/{base}_{name}"
    os.makedirs(matrix_dir, exist_ok=True)

    dst = _ckpt_dst(matrix_dir, name, seed, epoch)
    if os.path.exists(dst):
        print(f"[skip] {name} seed{seed}: collected checkpoint exists")
        return
    src = _ckpt_src(sub["work_dir"], seed, epoch)
    if not os.path.exists(src):
        _train_one_seed(sub, seed)
    if not os.path.exists(src):
        sys.exit(f"[WARN] expected {src} after training but missing; check num_epoch/save_interval")
    shutil.copy2(src, dst)
    print(f"[collect] {os.path.basename(src)} -> {os.path.basename(dst)}", flush=True)


if __name__ == "__main__":
    main()
