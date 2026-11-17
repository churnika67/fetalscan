"""Create an HC18 Grand Challenge submission CSV from a trained U-Net.

Download test_set.zip and test_set_pixel_size.csv from Zenodo record 1327317, then:
    python -m fetalscan.hc18_submit --data data/HC18 --ckpt runs/seg/best.pt --tta --out hc18_submission.csv
Upload the CSV at https://hc18.grand-challenge.org (needs a free account). The challenge expects
filename, center_x_mm, center_y_mm, semi_axes_a_mm, semi_axes_b_mm, angle_rad; check the current
submission page for the exact column spec before uploading.
"""
from __future__ import annotations

import argparse
import math
import os

import cv2
import numpy as np
import pandas as pd
import torch

from .biometry import ellipse_perimeter, fit_ellipse
from .models import UNet
from .preprocess import apply_clahe, letterbox, load_gray, to_tensor_input
from .train_seg import predict_prob
from .utils import pick_device


def predict_ellipse_original(model, gray: np.ndarray, size: int, device, tta: bool):
    """Segment at model resolution, map the probability map back to the original image, then fit."""
    h, w = gray.shape
    img, scale, (px, py) = letterbox(apply_clahe(gray), size)
    prob = predict_prob(model, torch.from_numpy(to_tensor_input(img)), device, tta)
    nh, nw = round(h * scale), round(w * scale)
    prob = cv2.resize(prob[py:py + nh, px:px + nw], (w, h), interpolation=cv2.INTER_LINEAR)
    return fit_ellipse((prob > 0.5).astype(np.uint8))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="HC18 folder containing test_set/ and test_set_pixel_size.csv")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="hc18_submission.csv")
    p.add_argument("--tta", action="store_true")
    a = p.parse_args()
    device = pick_device()
    ck = torch.load(a.ckpt, map_location=device)
    model = UNet(base=ck["base"]).to(device).eval()
    model.load_state_dict(ck["model"])
    px = pd.read_csv(os.path.join(a.data, "test_set_pixel_size.csv"))
    px.columns = [c.strip() for c in px.columns]
    rows, fails = [], 0
    for fn, mm in zip(px["filename"], px["pixel size(mm)"]):
        e = predict_ellipse_original(model, load_gray(os.path.join(a.data, "test_set", fn)), ck["size"], device, a.tta)
        if e is None:
            fails += 1
            rows.append([fn, 0, 0, 0, 0, 0, 0])
            continue
        rows.append([fn, e.cx * mm, e.cy * mm, e.a * mm, e.b * mm, math.radians(e.angle_deg),
                     ellipse_perimeter(e.a, e.b) * mm])
    out = pd.DataFrame(rows, columns=["filename", "center_x_mm", "center_y_mm", "semi_axes_a_mm",
                                      "semi_axes_b_mm", "angle_rad", "hc_mm"])
    out.drop(columns=["hc_mm"]).to_csv(a.out, index=False, float_format="%.6f")
    out.to_csv(a.out.replace(".csv", "_with_hc.csv"), index=False, float_format="%.4f")
    print(f"wrote {a.out}: {len(out)} images, {fails} with no head found, mean HC {out.hc_mm.mean():.1f} mm")


if __name__ == "__main__":
    main()
