"""Image loading and preprocessing for grayscale ultrasound frames (PIL + OpenCV)."""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def load_gray(path: str) -> np.ndarray:
    """Load an image as a uint8 grayscale array using PIL (handles PNG/JPG, 8/16-bit, RGBA)."""
    with Image.open(path) as im:
        im = im.convert("L")
        return np.asarray(im, dtype=np.uint8).copy()


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, tile: int = 8) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalization: boosts local contrast in speckled ultrasound."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    return clahe.apply(img)


def letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize keeping aspect ratio and pad to a square. Returns (image, scale, (pad_x, pad_y)).

    Keeping the aspect ratio matters for biometry: a stretched head would bias the ellipse fit.
    """
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = round(h * scale), round(w * scale)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    out = np.zeros((size, size), dtype=img.dtype)
    px, py = (size - nw) // 2, (size - nh) // 2
    out[py:py + nh, px:px + nw] = resized
    return out, scale, (px, py)


def letterbox_mask(mask: np.ndarray, size: int) -> np.ndarray:
    """Same geometry as letterbox() but nearest-neighbour so labels stay binary."""
    h, w = mask.shape[:2]
    scale = size / max(h, w)
    nh, nw = round(h * scale), round(w * scale)
    resized = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    out = np.zeros((size, size), dtype=mask.dtype)
    px, py = (size - nw) // 2, (size - nh) // 2
    out[py:py + nh, px:px + nw] = resized
    return out


def to_tensor_input(img: np.ndarray) -> np.ndarray:
    """uint8 HxW -> float32 1xHxW in [0, 1]."""
    return (img.astype(np.float32) / 255.0)[None, ...]
