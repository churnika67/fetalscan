"""ONNX Runtime inference used by the API and the demo app."""
from __future__ import annotations

import numpy as np

from .biometry import ellipse_perimeter, fit_ellipse
from .datasets import PLANE_CLASSES
from .preprocess import apply_clahe, letterbox, to_tensor_input


class FetalScanRuntime:
    def __init__(self, cls_onnx: str | None = None, seg_onnx: str | None = None):
        import onnxruntime as ort

        prov = ort.get_available_providers()
        self.cls = ort.InferenceSession(cls_onnx, providers=prov) if cls_onnx else None
        self.seg = ort.InferenceSession(seg_onnx, providers=prov) if seg_onnx else None

    @staticmethod
    def _size(sess) -> int:
        s = sess.get_inputs()[0].shape[-1]
        return s if isinstance(s, int) else 224

    def classify(self, gray: np.ndarray) -> dict:
        size = self._size(self.cls)
        img, _, _ = letterbox(apply_clahe(gray), size)
        logits = self.cls.run(None, {"image": to_tensor_input(img)[None]})[0][0]
        p = np.exp(logits - logits.max())
        p /= p.sum()
        k = int(p.argmax())
        return {"plane": PLANE_CLASSES[k], "confidence": float(p[k]),
                "probabilities": {c: float(v) for c, v in zip(PLANE_CLASSES, p)}}

    def measure_head(self, gray: np.ndarray, pixel_size_mm: float) -> dict:
        size = self._size(self.seg)
        img, scale, _ = letterbox(apply_clahe(gray), size)
        logits = self.seg.run(None, {"image": to_tensor_input(img)[None]})[0][0, 0]
        mask = (logits > 0).astype(np.uint8)  # sigmoid(x) > 0.5  <=>  x > 0
        e = fit_ellipse(mask)
        if e is None:
            return {"head_found": False, "mask": mask}
        mm = pixel_size_mm / scale
        return {"head_found": True, "hc_mm": ellipse_perimeter(e.a, e.b) * mm,
                "bpd_like_minor_axis_mm": 2 * min(e.a, e.b) * mm, "ellipse": e, "mask": mask, "image": img}
