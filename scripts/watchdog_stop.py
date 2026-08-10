"""Detached watchdog: stop the campaign after `lora` and `lora_r4` each reach 3
seeds, WITHOUT starting any further queued config.

Kills Driver A (queue starts 'lora full_ft ...') the instant bhc_lora has 3
seeds, and Driver B (queue starts 'lora_r4 lora_r16 ...') when bhc_lora_r4 has 3
seeds. Killing each driver as soon as its config completes (a) avoids wasting
compute on the next queued config and (b) hands the whole GPU to the survivor so
lora_r4 finishes faster. Launched detached (Start-Process) so it survives session
boundaries. Idempotent/observational: only ever kills the two named drivers.
"""
import re
import subprocess
import time

CSV = "results_final.csv"
SRC = r"_(bdsl_mnist|bdsl47_digits|bdsl47_letters|bsld_45|bdsl49_recognition)$"
# (config base -> unique cmdline substring identifying its driver process)
TARGETS = {
    "bhc_lora":    "lora full_ft",       # Driver A
    "bhc_lora_r4": "lora_r4 lora_r16",   # Driver B
}


def seeds_done(base):
    try:
        import pandas as pd
        df = pd.read_csv(CSV)
    except Exception:
        return 0
    seeds = set()
    for e in df["Experiment"].astype(str):
        m = re.match(r"^(.*)_seed(\d+)$", e)
        if not m:
            continue
        b = re.sub(SRC, "", m.group(1))
        if b == base:
            seeds.add(int(m.group(2)))
    return len(seeds)


def kill_driver(substr):
    ps = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
          "Where-Object { $_.CommandLine -match 'run_campaign' -and "
          f"$_.CommandLine -match '{substr}' " + "} | "
          "ForEach-Object { Stop-Process -Id $_.ProcessId -Force; $_.ProcessId }")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                         capture_output=True, text=True).stdout.strip()
    return out


def main():
    killed = {b: False for b in TARGETS}
    print(f"[watchdog] armed for {list(TARGETS)}", flush=True)
    while not all(killed.values()):
        for base, substr in TARGETS.items():
            if killed[base]:
                continue
            n = seeds_done(base)
            if n >= 3:
                pid = kill_driver(substr)
                killed[base] = True
                print(f"[watchdog] {base} reached 3 seeds -> killed driver "
                      f"'{substr}' (pid {pid})", flush=True)
        time.sleep(45)
    print("[watchdog] both target configs complete; drivers stopped. done.",
          flush=True)


if __name__ == "__main__":
    main()
