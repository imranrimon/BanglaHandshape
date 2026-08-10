"""Robust single-process campaign driver for the sister-paper experiments.

Runs the given config names (each x 3 seeds) IN ONE PROCESS by importing
_train_one_seed (no bash loop / subprocess to break if the harness stops the
launcher). Idempotent: a (config, seed) whose rows already exist in
results_final.csv is skipped, so relaunching resumes where it left off. Also
waits for any external `-m ...train_baseline` run (e.g. a still-running config)
to finish first, to avoid GPU contention / duplicate work.

Usage (run with cwd = repo root):
    python -u scripts/run_campaign.py lora probe_imagenet_vit_s
"""
import os
import subprocess
import sys
import time

sys.path.insert(0, os.getcwd())
import pandas as pd
import yaml

from path3_handshape_benchmark.train_baseline import _train_one_seed
from path3_handshape_benchmark.train_probe_cached import is_probe_config, run_probe_cached

# CSV path is env-overridable so a SLURM job ARRAY can give each task its own
# file (results/bhc_<config>.csv) — concurrent appends from many nodes to one
# shared CSV would interleave/corrupt rows. Merge the per-task CSVs afterwards.
CSV = os.environ.get("CAMPAIGN_CSV", "results_final.csv")
SEEDS = [int(s) for s in os.environ.get("CAMPAIGN_SEEDS", "0 1 2").split()]
CONFIG_DIR = "path3_handshape_benchmark/configs"


def seed_done(base, seed):
    if not os.path.exists(CSV):
        return False
    try:
        df = pd.read_csv(CSV)
    except Exception:
        return False
    if "Experiment" not in df.columns:
        return False
    return bool(df["Experiment"].astype(str).str.match(rf"^{base}_.+_seed{seed}$").any())


def external_train_running():
    """True if a separate `-m ...train_baseline` python process is running.
    (This driver imports the function, so its own cmdline does NOT contain
    'train_baseline'.)

    Cross-platform: `pgrep` on Linux/HPC (DollySods), PowerShell/Win32 on
    Windows. Either failure path returns False, so at worst the driver skips the
    GPU-contention wait and starts immediately — never a hang."""
    my_pid = str(os.getpid())
    if os.name != "nt":
        # -f matches the full command line; exclude our own pid so importing
        # the module here doesn't count as an external run.
        try:
            out = subprocess.run(["pgrep", "-af", "train_baseline"],
                                 capture_output=True, text=True, timeout=30).stdout
            for line in out.splitlines():
                pid = line.split(None, 1)[0]
                if pid and pid != my_pid:
                    return True
            return False
        except Exception:
            return False
    ps = ("(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
          "Where-Object { $_.CommandLine -match 'train_baseline' } | "
          "Measure-Object).Count")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        return int(out) > 0
    except Exception:
        return False


def main(configs):
    while external_train_running():
        print("[wait] external train_baseline still running; sleep 60s", flush=True)
        time.sleep(60)
    for cfg in configs:
        with open(os.path.join(CONFIG_DIR, cfg + ".yaml")) as f:
            c = yaml.safe_load(f)
        # Parallel loading — nw=0 was ~6x slower (24 vs ~4 min/epoch). num_workers>0
        # can intermittently hit the Windows "shared file mapping ... 1455"
        # paging-commitment crash, so the per-seed loop below auto-retries.
        c["num_workers"] = int(os.environ.get("CAMPAIGN_NW", "4"))
        # Full-budget: >=50 epochs for every config (Imran's directive). Harmless
        # with best-over-epochs selection — it just captures the true val peak.
        c["num_epoch"] = max(int(c.get("num_epoch", 5)), 50)
        base = c.get("Experiment_name", "bhc_" + cfg)
        todo = [s for s in SEEDS if not seed_done(base, s)]
        if not todo:
            print(f"[skip] {base} all seeds already in CSV", flush=True)
            continue

        # Frozen-backbone probe configs use feature-caching (extract once, train
        # heads on cached vectors) — identical result, ~10x+ faster than image-based.
        if is_probe_config(c):
            print(f"=== PROBE-CACHED {cfg} seeds={todo} ===", flush=True)
            for attempt in range(1, 4):
                try:
                    run_probe_cached(c, todo, CSV)
                    print(f"=== DONE {cfg} (cached) ===", flush=True)
                    break
                except Exception as e:
                    print(f"=== RETRY {cfg} cached (attempt {attempt} failed: "
                          f"{type(e).__name__}: {str(e)[:120]}) ===", flush=True)
                    time.sleep(15)
            else:
                print(f"=== GAVEUP {cfg} cached after 3 attempts ===", flush=True)
            continue

        # Image-based training (LoRA / full-FT / CNN): backbone trains, needs images.
        for s in todo:
            print(f"=== START {cfg} seed{s} ===", flush=True)
            for attempt in range(1, 4):
                try:
                    _train_one_seed(c, s, results_csv=CSV)
                    print(f"=== DONE {cfg} seed{s} ===", flush=True)
                    break
                except Exception as e:
                    print(f"=== RETRY {cfg} seed{s} (attempt {attempt} failed: "
                          f"{type(e).__name__}: {str(e)[:120]}) ===", flush=True)
                    time.sleep(15)
            else:
                print(f"=== GAVEUP {cfg} seed{s} after 3 attempts ===", flush=True)
    print("=== CAMPAIGN BATCH COMPLETE ===", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
