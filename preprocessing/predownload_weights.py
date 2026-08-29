#!/usr/bin/env python3
"""Pre-fetch every backbone weight the benchmark needs, on a node WITH internet.

Why this exists: DollySods GPU compute nodes have **no internet**, so a training
job that instantiates ``timm.create_model(..., pretrained=True)`` dies with
"Network is unreachable" while trying to HEAD huggingface.co. The login node *does*
have internet, so run this there once to populate the HF hub cache
(``~/.cache/huggingface``) and the torch hub cache (``~/.cache/torch``); the SLURM
jobs then set ``HF_HUB_OFFLINE=1`` / ``TRANSFORMERS_OFFLINE=1`` and load from cache.

Covers every backbone referenced by the T1 configs + the reviewer (§E/§A) probes:
DINOv2 S/B/L, ImageNet ViT-S, SigLIP-B, MAE-B (timm) and ResNet-50 ImageNet
(torchvision, for cnn_resnet50_imagenet). ResNet-18-scratch needs no weights.

Idempotent: an already-cached weight is a fast no-op. Run:
    python preprocessing/predownload_weights.py
Then submit the arrays. If the cache lives on purge-prone scratch, re-run after a purge.
"""
from __future__ import annotations

import sys
import traceback

TIMM_BACKBONES = [
    "vit_small_patch14_dinov2.lvd142m",   # linear_probe / lora / bdsl47_si|sd
    "vit_base_patch14_dinov2.lvd142m",     # probe_dinov2_b + reviewer headline
    "vit_large_patch14_dinov2.lvd142m",    # probe_dinov2_l
    "vit_small_patch16_224.augreg_in1k",   # probe_imagenet_vit_s
    "vit_base_patch16_siglip_224.webli",   # probe_siglip_b
    "vit_base_patch16_224.mae",            # probe_mae_b
]


def main():
    ok, fail = [], []
    import timm
    for name in TIMM_BACKBONES:
        try:
            timm.create_model(name, pretrained=True, num_classes=0, dynamic_img_size=True)
            print(f"[OK]   timm {name}", flush=True); ok.append(name)
        except Exception as e:
            print(f"[FAIL] timm {name}: {e}", flush=True); traceback.print_exc(); fail.append(name)
    try:
        from torchvision import models
        models.resnet50(weights="DEFAULT")   # cnn_resnet50_imagenet
        print("[OK]   torchvision resnet50 (ImageNet DEFAULT)", flush=True)
        ok.append("resnet50")
    except Exception as e:
        print(f"[FAIL] torchvision resnet50: {e}", flush=True); fail.append("resnet50")

    print(f"\n[predownload] {len(ok)} cached, {len(fail)} failed", flush=True)
    if fail:
        print(f"[predownload] FAILED: {fail} -- are you on a node with internet?", flush=True)
        sys.exit(1)
    print("PREDOWNLOAD_DONE", flush=True)


if __name__ == "__main__":
    main()
