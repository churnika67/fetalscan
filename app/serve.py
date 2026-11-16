"""FastAPI service for real-time inference.

    FETALSCAN_CLS=exports/classifier.onnx FETALSCAN_SEG=exports/segmenter.onnx uvicorn app.serve:app --port 8000
"""
from __future__ import annotations

import io
import os
import time

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image

from fetalscan.inference import FetalScanRuntime

app = FastAPI(title="FetalScan", version="0.1.0",
              description="Research demo only. Not a medical device; not for clinical use.")
_rt: FetalScanRuntime | None = None


def runtime() -> FetalScanRuntime:
    global _rt
    if _rt is None:
        _rt = FetalScanRuntime(os.getenv("FETALSCAN_CLS"), os.getenv("FETALSCAN_SEG"))
    return _rt


async def _read_gray(file: UploadFile) -> np.ndarray:
    data = await file.read()
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(413, "image larger than 10 MB")
    try:
        with Image.open(io.BytesIO(data)) as im:
            return np.asarray(im.convert("L"), dtype=np.uint8).copy()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, "could not decode image") from exc


@app.get("/health")
def health():
    rt = runtime()
    return {"classifier": rt.cls is not None, "segmenter": rt.seg is not None}


@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    rt = runtime()
    if rt.cls is None:
        raise HTTPException(503, "classifier model not loaded")
    gray = await _read_gray(file)
    t0 = time.perf_counter()
    out = rt.classify(gray)
    out["latency_ms"] = (time.perf_counter() - t0) * 1000
    return out


@app.post("/measure-head")
async def measure_head(file: UploadFile = File(...), pixel_size_mm: float = Query(..., gt=0, lt=5)):
    rt = runtime()
    if rt.seg is None:
        raise HTTPException(503, "segmenter model not loaded")
    gray = await _read_gray(file)
    t0 = time.perf_counter()
    out = rt.measure_head(gray, pixel_size_mm)
    resp = {"head_found": out["head_found"], "latency_ms": (time.perf_counter() - t0) * 1000}
    if out["head_found"]:
        resp["hc_mm"] = round(out["hc_mm"], 2)
        resp["minor_axis_mm"] = round(out["bpd_like_minor_axis_mm"], 2)
    return resp
