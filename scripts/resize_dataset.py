"""Pre-resize FETAL_PLANES_DB images once (longest side = N px) to speed up CPU/GPU training.

Decoding ~1000x700 PNGs every epoch dominates data-loading time; this keeps the same folder layout.

    python scripts/resize_dataset.py --src data/FETAL_PLANES_ZENODO --dst data/FETAL_PLANES_256 --max-side 256
"""
from __future__ import annotations

import argparse
import os
import shutil
from concurrent.futures import ProcessPoolExecutor

import cv2


def _one(args):
    src, dst, n = args
    img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    s = n / max(h, w)
    if s < 1:
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(dst, img)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--max-side", type=int, default=256)
    a = p.parse_args()
    os.makedirs(os.path.join(a.dst, "Images"), exist_ok=True)
    shutil.copy(os.path.join(a.src, "FETAL_PLANES_DB_data.csv"), a.dst)
    names = [f for f in os.listdir(os.path.join(a.src, "Images")) if f.endswith(".png")]
    jobs = [(os.path.join(a.src, "Images", f), os.path.join(a.dst, "Images", f), a.max_side) for f in names]
    with ProcessPoolExecutor() as ex:
        list(ex.map(_one, jobs, chunksize=64))
    print(f"resized {len(jobs)} images -> {a.dst}")


if __name__ == "__main__":
    main()
