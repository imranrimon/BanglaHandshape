#!/bin/bash -l
#SBATCH -J bhc_stage
#SBATCH -c 4 -n 1
#SBATCH -p gpu_2day,gpu_7day
#SBATCH --exclude=dscog035,dscog036,dsdmg002
#SBATCH -t 00:30:00
#SBATCH -o /users/mh00145/bhc_stage_%j.out
#SBATCH -e /users/mh00145/bhc_stage_%j.err
set -eo pipefail
R=/scratch/mh00145/BanglaHandshape
H=/users/mh00145/BH_home

# Clean, single-writer, verified stage of RSBdSL38 (fixes the concurrent-rsync
# corruption in train/37/). NO --ignore-existing: force a fresh, complete copy.
rm -rf "$R/data/RSBdSL38"
mkdir -p "$R/data/RSBdSL38"
rsync -a "$H/data/RSBdSL38/" "$R/data/RSBdSL38/"

source ~/miniconda3/etc/profile.d/conda.sh; conda activate bdsl_graph
python - "$R/data/RSBdSL38" <<'PY'
import os, sys
from PIL import Image
root=sys.argv[1]; n=0; bad=[]
for dp,_,fs in os.walk(root):
    for f in fs:
        if f.lower().endswith(('.jpg','.jpeg','.png')):
            n+=1; p=os.path.join(dp,f)
            try: Image.open(p).convert("RGB").load()
            except Exception as e: bad.append((p, type(e).__name__))
print(f"[verify] scanned {n} images on scratch; {len(bad)} unreadable")
for p,e in bad[:20]: print("  BAD:", e, p)
sys.exit(1 if bad else 0)
PY
echo "== RSBdSL38 staged + verified clean on scratch =="
