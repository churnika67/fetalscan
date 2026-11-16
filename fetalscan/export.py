"""Export trained models to ONNX and benchmark inference latency with ONNX Runtime.

    python -m fetalscan.export --cls runs/cls/best.pt --seg runs/seg/best.pt --out exports
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from .models import UNet, build_classifier
from .utils import save_json


def export_classifier(ckpt_path: str, out: str) -> str:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    m = build_classifier(len(ckpt["classes"]), pretrained=False)
    m.load_state_dict(ckpt["model"])
    m.eval()
    dst = os.path.join(out, "classifier.onnx")
    x = torch.zeros(1, 1, ckpt["size"], ckpt["size"])
    torch.onnx.export(m, x, dst, input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17, dynamo=False)
    return dst


def export_segmenter(ckpt_path: str, out: str) -> str:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    m = UNet(base=ckpt["base"])
    m.load_state_dict(ckpt["model"])
    m.eval()
    dst = os.path.join(out, "segmenter.onnx")
    x = torch.zeros(1, 1, ckpt["size"], ckpt["size"])
    torch.onnx.export(m, x, dst, input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17, dynamo=False)
    return dst


def benchmark(onnx_path: str, runs: int = 50) -> dict:
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=ort.get_available_providers())
    inp = sess.get_inputs()[0]
    size = inp.shape[-1] if isinstance(inp.shape[-1], int) else 224
    x = np.random.rand(1, 1, size, size).astype(np.float32)
    for _ in range(5):
        sess.run(None, {inp.name: x})
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {inp.name: x})
        times.append((time.perf_counter() - t0) * 1000)
    return {"p50_ms": float(np.percentile(times, 50)), "p95_ms": float(np.percentile(times, 95)),
            "fps_p50": float(1000 / np.percentile(times, 50)), "providers": sess.get_providers()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cls")
    p.add_argument("--seg")
    p.add_argument("--out", default="exports")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    report = {}
    if a.cls:
        report["classifier"] = benchmark(export_classifier(a.cls, a.out))
    if a.seg:
        report["segmenter"] = benchmark(export_segmenter(a.seg, a.out))
    save_json(report, os.path.join(a.out, "latency.json"))
    print(report)


if __name__ == "__main__":
    main()
