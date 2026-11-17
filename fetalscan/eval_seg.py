"""Evaluate a segmentation checkpoint on the fixed HC18 test split, with and without test-time augmentation.

    python -m fetalscan.eval_seg --data data/HC18 --ckpt runs/seg/best.pt --out runs/seg/eval.json
"""
from __future__ import annotations

import argparse

import torch

from .datasets import HC18Dataset, random_split, read_hc18_csv
from .models import UNet
from .train_seg import evaluate
from .utils import pick_device, save_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0, help="must match the training split seed")
    a = p.parse_args()
    device = pick_device()
    ck = torch.load(a.ckpt, map_location=device)
    model = UNet(base=ck["base"]).to(device)
    model.load_state_dict(ck["model"])
    _, _, te = random_split(read_hc18_csv(a.data), seed=a.seed)
    ds = HC18Dataset(te, ck["size"])
    res = {"size": ck["size"], "no_tta": evaluate(model, ds, device), "flip_tta": evaluate(model, ds, device, tta=True)}
    save_json(res, a.out)
    for k in ("no_tta", "flip_tta"):
        r = res[k]
        print(f"{k}: dice={r['dice_mean']:.4f} HC_MAE={r['hc_mae_mm']:.2f}mm within3mm={r['hc_within_3mm']:.1%}")


if __name__ == "__main__":
    main()
