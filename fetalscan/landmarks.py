"""Fetal head biometry landmark detection (heatmap regression).

HC18 ships ellipse annotations, not point labels, so the 4 landmarks are derived from each annotated
ellipse: the 2 endpoints of the minor axis (BPD-style biparietal points) and the 2 endpoints of the major
axis (OFD-style occipitofrontal points). A U-Net predicts one heatmap per landmark; soft-argmax turns each into sub-pixel coordinates.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch.utils.data import Dataset

from .biometry import fit_ellipse, outline_to_filled_mask
from .datasets import HC18Dataset, _augment_pair
from .preprocess import apply_clahe, letterbox, letterbox_mask, load_gray, to_tensor_input

LANDMARKS = ["BPD_1", "BPD_2", "OFD_1", "OFD_2"]


def _order(p, q):
    """Canonical order for an endpoint pair so labels are consistent across images (left first, then top)."""
    return (p, q) if (p[0], p[1]) <= (q[0], q[1]) else (q, p)


def ellipse_landmarks(mask: np.ndarray) -> np.ndarray | None:
    """4x2 array (x, y): minor-axis endpoints (BPD) then major-axis endpoints (OFD)."""
    e = fit_ellipse(mask)
    if e is None:
        return None
    t = math.radians(e.angle_deg)
    # cv2.fitEllipse: first axis (a) lies along angle t, second (b) along t + 90 degrees
    ua, ub = np.array([math.cos(t), math.sin(t)]), np.array([-math.sin(t), math.cos(t)])
    c = np.array([e.cx, e.cy])
    ends_a = (c - e.a * ua, c + e.a * ua)
    ends_b = (c - e.b * ub, c + e.b * ub)
    minor, major = (ends_a, ends_b) if e.a <= e.b else (ends_b, ends_a)
    return np.array([*_order(*minor), *_order(*major)], dtype=np.float32)


def gaussian_heatmaps(points: np.ndarray, size: int, sigma: float) -> np.ndarray:
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    return np.stack([np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2 * sigma**2)) for x, y in points]).astype(np.float32)


def decode_heatmaps(hm: np.ndarray, window: int = 2) -> np.ndarray:
    """Argmax per channel, refined to sub-pixel with a weighted mean in a small window."""
    pts = []
    for h in hm:
        y, x = np.unravel_index(int(h.argmax()), h.shape)
        y0, y1 = max(0, y - window), min(h.shape[0], y + window + 1)
        x0, x1 = max(0, x - window), min(h.shape[1], x + window + 1)
        patch = np.clip(h[y0:y1, x0:x1], 0, None)
        if patch.sum() > 0:
            gy, gx = np.mgrid[y0:y1, x0:x1]
            pts.append(((gx * patch).sum() / patch.sum(), (gy * patch).sum() / patch.sum()))
        else:
            pts.append((float(x), float(y)))
    return np.array(pts, dtype=np.float32)


def heatmap_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE weighted toward the peaks (kept as an auxiliary term)."""
    return (((pred - target) ** 2) * (1 + 10 * target)).mean()


def soft_argmax(logits: torch.Tensor) -> torch.Tensor:
    """Differentiable landmark coordinates: spatial softmax per channel, then expected (x, y). Bx K x2."""
    b, k, h, w = logits.shape
    prob = torch.softmax(logits.flatten(2), dim=-1).view(b, k, h, w)
    xs = torch.arange(w, device=logits.device, dtype=logits.dtype)
    ys = torch.arange(h, device=logits.device, dtype=logits.dtype)
    return torch.stack([(prob.sum(2) * xs).sum(-1), (prob.sum(3) * ys).sum(-1)], dim=-1)


def pair_invariant_l1(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Per-image L1 where the two endpoints of each axis may be swapped (an axis has no natural direction).

    Without this, near-vertical axes flip their canonical order between images and the model learns to
    predict the midpoint. pred, gt: B x 4 x 2 (BPD_1, BPD_2, OFD_1, OFD_2). Returns B x 4 point errors.
    """
    out = []
    for i in (0, 2):
        p, g = pred[:, i:i + 2], gt[:, i:i + 2]
        keep = (p - g).abs().sum(-1)
        swap = (p - g.flip(1)).abs().sum(-1)
        out.append(torch.where((keep.sum(1) <= swap.sum(1))[:, None], keep, swap))
    return torch.cat(out, dim=1)


def match_pairs(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Reorder pred endpoints within each axis to best match gt (used for evaluation)."""
    pred = pred.copy()
    for i in (0, 2):
        keep = np.linalg.norm(pred[i:i + 2] - gt[i:i + 2], axis=1).sum()
        swap = np.linalg.norm(pred[i:i + 2][::-1] - gt[i:i + 2], axis=1).sum()
        if swap < keep:
            pred[i:i + 2] = pred[i:i + 2][::-1].copy()
    return pred


class HC18LandmarkDataset(Dataset):
    """Returns (image 1xSxS, heatmaps 4xSxS, landmarks 4x2 in pixels, pixel size in mm at size S)."""

    def __init__(self, df, size: int = 256, train: bool = False, sigma: float = 4.0, seed: int = 0):
        self.df, self.size, self.train, self.sigma = df, size, train, sigma
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = apply_clahe(load_gray(row["img_path"]))
        mask = outline_to_filled_mask(load_gray(row["ann_path"]))
        img, scale, _ = letterbox(img, self.size)
        mask = letterbox_mask(mask, self.size)
        if self.train:
            img, mask = _augment_pair(img, mask, self.rng)
        pts = ellipse_landmarks(mask)  # recomputed after augmentation, so labels follow the geometry
        if pts is None:  # augmentation pushed the head out of frame: fall back to the unaugmented sample
            return HC18LandmarkDataset(self.df.iloc[[i]], self.size, False, self.sigma)[0]
        hm = gaussian_heatmaps(pts, self.size, self.sigma)
        return (torch.from_numpy(to_tensor_input(img)), torch.from_numpy(hm), torch.from_numpy(pts),
                float(row["pixel_mm"]) / scale)


__all__ = ["LANDMARKS", "HC18Dataset", "HC18LandmarkDataset", "decode_heatmaps", "ellipse_landmarks",
           "gaussian_heatmaps", "heatmap_loss", "match_pairs", "pair_invariant_l1", "soft_argmax"]
