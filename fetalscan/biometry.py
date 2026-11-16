"""Head circumference (HC) from a segmentation mask via OpenCV ellipse fitting."""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Ellipse:
    cx: float
    cy: float
    a: float  # semi-axis (pixels)
    b: float  # semi-axis (pixels)
    angle_deg: float


def ellipse_perimeter(a: float, b: float) -> float:
    """Ramanujan's second approximation of the ellipse perimeter (error < 1e-9 relative for HC shapes)."""
    if a <= 0 or b <= 0:
        return 0.0
    h = ((a - b) ** 2) / ((a + b) ** 2)
    return math.pi * (a + b) * (1 + (3 * h) / (10 + math.sqrt(4 - 3 * h)))


def largest_contour(mask: np.ndarray) -> np.ndarray | None:
    m = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def fit_ellipse(mask: np.ndarray) -> Ellipse | None:
    """Fit an ellipse to the largest connected region of a binary mask."""
    c = largest_contour(mask)
    if c is None or len(c) < 5:  # cv2.fitEllipse needs >= 5 points
        return None
    (cx, cy), (d1, d2), angle = cv2.fitEllipse(c)
    return Ellipse(cx, cy, d1 / 2.0, d2 / 2.0, angle)


def head_circumference_mm(mask: np.ndarray, pixel_size_mm: float) -> float | None:
    """HC in millimetres. pixel_size_mm must be the size of one pixel *in the mask's resolution*."""
    e = fit_ellipse(mask)
    if e is None:
        return None
    return ellipse_perimeter(e.a, e.b) * pixel_size_mm


def outline_to_filled_mask(outline: np.ndarray) -> np.ndarray:
    """HC18 annotations are a thin ellipse outline; fill the inside to get a segmentation mask."""
    binary = (outline > 127).astype(np.uint8)
    # close small gaps in the drawn outline before filling
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(binary)
    if contours:
        cv2.drawContours(filled, contours, -1, color=1, thickness=cv2.FILLED)
    return filled


def draw_ellipse(img_gray: np.ndarray, e: Ellipse, color=(0, 255, 0)) -> np.ndarray:
    vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    cv2.ellipse(vis, ((e.cx, e.cy), (2 * e.a, 2 * e.b), e.angle_deg), color, 2)
    return vis
