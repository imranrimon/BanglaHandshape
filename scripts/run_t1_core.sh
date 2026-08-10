#!/bin/bash
# T1 core baselines: linear probe, LoRA, ImageNet-ViT probe — 3 seeds each,
# all 5 sources. Each config's full output goes to logs/t1_<cfg>.log; START/DONE
# markers go to this driver's stdout (the background-task stream).
cd /f/BanglaHandshape || exit 1
PY='C:/Users/rimon/anaconda3/envs/bdsl_graph/python.exe'
export PYTHONUNBUFFERED=1
mkdir -p logs
for cfg in linear_probe lora probe_imagenet_vit_s; do
  echo "=== START $cfg $(date +%H:%M:%S) ==="
  "$PY" -u -m path3_handshape_benchmark.train_baseline \
      --config "path3_handshape_benchmark/configs/$cfg.yaml" --seeds 0 1 2 \
      > "logs/t1_$cfg.log" 2>&1
  echo "=== DONE $cfg exit=$? $(date +%H:%M:%S) ==="
done
echo "=== T1 CORE COMPLETE ==="
