"""Write Grad-CAM overlays for test images to see where the classifier looks.

    python -m fetalscan.explain --ckpt runs/cls/best.pt --images img1.png img2.png --out runs/cls/gradcam
"""
from __future__ import annotations

import argparse
import os

import cv2
import torch

from .gradcam import GradCAM, overlay
from .models import build_classifier
from .preprocess import apply_clahe, letterbox, load_gray, to_tensor_input


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--images", nargs="+", required=True)
    p.add_argument("--out", default="runs/gradcam")
    a = p.parse_args()
    ckpt = torch.load(a.ckpt, map_location="cpu")
    classes, size = ckpt["classes"], ckpt["size"]
    model = build_classifier(len(classes), pretrained=False)
    model.load_state_dict(ckpt["model"])
    cam = GradCAM(model, model.layer4)
    os.makedirs(a.out, exist_ok=True)
    for path in a.images:
        img, _, _ = letterbox(apply_clahe(load_gray(path)), size)
        heat, k = cam(torch.from_numpy(to_tensor_input(img))[None])
        vis = overlay(img, heat)
        cv2.putText(vis, classes[k], (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        dst = os.path.join(a.out, os.path.basename(path))
        cv2.imwrite(dst, vis)
        print(f"{path}: {classes[k]} -> {dst}")
    cam.remove()


if __name__ == "__main__":
    main()
