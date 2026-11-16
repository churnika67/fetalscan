"""Generate small synthetic datasets in the exact FETAL_PLANES_DB and HC18 folder layouts.

These are NOT ultrasound data. They exist only so the full pipeline (and CI) can run end to end
without downloading the real datasets. Never report metrics from synthetic data.

    python scripts/make_synthetic.py --out data/synthetic --n-per-class 30 --n-hc 60
"""
from __future__ import annotations

import argparse
import math
import os

import cv2
import numpy as np


def speckle(h, w, rng):
    base = rng.rayleigh(scale=40, size=(h, w)).astype(np.float32)
    return cv2.GaussianBlur(base, (5, 5), 0)


def plane_image(label: int, rng, h=180, w=240):
    img = speckle(h, w, rng)
    cx, cy = w // 2 + rng.integers(-20, 20), h // 2 + rng.integers(-15, 15)
    br = int(rng.integers(170, 230))
    if label == 0:  # abdomen: round bright ring
        cv2.circle(img, (cx, cy), int(rng.integers(45, 60)), br, 4)
    elif label == 1:  # brain: ellipse with a midline
        a, b = int(rng.integers(55, 70)), int(rng.integers(40, 50))
        cv2.ellipse(img, (cx, cy), (a, b), 0, 0, 360, br, 4)
        cv2.line(img, (cx - a + 10, cy), (cx + a - 10, cy), br, 2)
    elif label == 2:  # femur: bright bar
        ang = rng.uniform(-0.4, 0.4)
        dx, dy = int(70 * math.cos(ang)), int(70 * math.sin(ang))
        cv2.line(img, (cx - dx, cy - dy), (cx + dx, cy + dy), br, 8)
    elif label == 3:  # thorax: ring with a filled blob
        cv2.circle(img, (cx, cy), int(rng.integers(45, 60)), br, 3)
        cv2.circle(img, (cx - 15, cy), 15, br, -1)
    elif label == 4:  # cervix: dark wedge
        pts = np.array([[cx - 60, cy + 40], [cx + 60, cy + 40], [cx, cy - 40]], np.int32)
        cv2.fillPoly(img, [pts], 20)
    # 5 = Other: speckle only
    return np.clip(img, 0, 255).astype(np.uint8)


def make_planes(out, n_per_class, rng):
    classes = ["Fetal abdomen", "Fetal brain", "Fetal femur", "Fetal thorax", "Maternal cervix", "Other"]
    os.makedirs(os.path.join(out, "Images"), exist_ok=True)
    rows = ["Image_name;Patient_num;Plane;Brain_plane;Operator;US_Machine;Train "]
    k = 0
    for label, name in enumerate(classes):
        for i in range(n_per_class):
            fn = f"Patient{k:05d}_Plane{label}_{i}_of_{n_per_class}"
            cv2.imwrite(os.path.join(out, "Images", fn + ".png"), plane_image(label, rng))
            patient = k // 3  # 3 images per synthetic patient
            train = 0 if (patient % 5 == 0) else 1
            rows.append(f"{fn};{patient};{name};Not A Brain;Op. 1;Voluson E6;{train}")
            k += 1
    with open(os.path.join(out, "FETAL_PLANES_DB_data.csv"), "w") as f:
        f.write("\n".join(rows) + "\n")


def make_hc18(out, n, rng, h=270, w=400):
    d = os.path.join(out, "training_set")
    os.makedirs(d, exist_ok=True)
    rows = ["filename,pixel size(mm),head circumference (mm)"]
    for i in range(n):
        img = speckle(h, w, rng)
        cx, cy = int(rng.integers(150, 250)), int(rng.integers(110, 160))
        a, b = int(rng.integers(60, 100)), int(rng.integers(45, 70))
        ang = float(rng.uniform(0, 180))
        cv2.ellipse(img, (cx, cy), (a, b), ang, 0, 360, 210, 6)  # bright skull
        inner = np.zeros((h, w), np.uint8)
        cv2.ellipse(inner, (cx, cy), (a - 6, b - 6), ang, 0, 360, 1, -1)
        img[inner > 0] *= 0.6
        ann = np.zeros((h, w), np.uint8)
        cv2.ellipse(ann, (cx, cy), (a, b), ang, 0, 360, 255, 1)
        pix = float(rng.uniform(0.1, 0.3))
        hh = ((a - b) ** 2) / ((a + b) ** 2)
        hc = math.pi * (a + b) * (1 + 3 * hh / (10 + math.sqrt(4 - 3 * hh))) * pix
        fn = f"{i:03d}_HC.png"
        cv2.imwrite(os.path.join(d, fn), np.clip(img, 0, 255).astype(np.uint8))
        cv2.imwrite(os.path.join(d, f"{i:03d}_HC_Annotation.png"), ann)
        rows.append(f"{fn},{pix:.6f},{hc:.2f}")
    with open(os.path.join(out, "training_set_pixel_size_and_HC.csv"), "w") as f:
        f.write("\n".join(rows) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/synthetic")
    p.add_argument("--n-per-class", type=int, default=30)
    p.add_argument("--n-hc", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    make_planes(os.path.join(a.out, "planes"), a.n_per_class, rng)
    make_hc18(os.path.join(a.out, "hc18"), a.n_hc, rng)
    print("wrote", a.out)
