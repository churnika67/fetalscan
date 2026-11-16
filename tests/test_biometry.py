import math

import cv2
import numpy as np
import pytest

from fetalscan.biometry import ellipse_perimeter, fit_ellipse, head_circumference_mm, outline_to_filled_mask


def test_perimeter_circle_exact():
    assert ellipse_perimeter(10, 10) == pytest.approx(2 * math.pi * 10, rel=1e-12)


def test_perimeter_known_ellipse():
    # a=2, b=1: exact perimeter 9.688448220547675...
    assert ellipse_perimeter(2, 1) == pytest.approx(9.688448220547675, rel=1e-6)


def test_perimeter_degenerate():
    assert ellipse_perimeter(0, 5) == 0.0


@pytest.mark.parametrize("a,b,angle", [(80, 50, 0), (90, 60, 35), (70, 69, 120)])
def test_fit_ellipse_recovers_axes(a, b, angle):
    mask = np.zeros((300, 400), np.uint8)
    cv2.ellipse(mask, (200, 150), (a, b), angle, 0, 360, 1, -1)
    e = fit_ellipse(mask)
    assert max(e.a, e.b) == pytest.approx(a, abs=1.5)
    assert min(e.a, e.b) == pytest.approx(b, abs=1.5)
    assert (e.cx, e.cy) == pytest.approx((200, 150), abs=1.0)


def test_hc_mm_uses_pixel_size():
    mask = np.zeros((300, 400), np.uint8)
    cv2.ellipse(mask, (200, 150), (80, 50), 0, 0, 360, 1, -1)
    hc1 = head_circumference_mm(mask, 0.1)
    hc2 = head_circumference_mm(mask, 0.2)
    assert hc2 == pytest.approx(2 * hc1, rel=1e-9)
    assert hc1 == pytest.approx(ellipse_perimeter(80, 50) * 0.1, rel=0.02)


def test_empty_mask_returns_none():
    assert head_circumference_mm(np.zeros((50, 50), np.uint8), 0.1) is None


def test_largest_region_wins():
    mask = np.zeros((300, 400), np.uint8)
    cv2.ellipse(mask, (150, 150), (80, 50), 0, 0, 360, 1, -1)
    cv2.circle(mask, (360, 40), 8, 1, -1)  # small false-positive blob
    e = fit_ellipse(mask)
    assert (e.cx, e.cy) == pytest.approx((150, 150), abs=1.0)


def test_outline_filled_matches_solid():
    outline = np.zeros((300, 400), np.uint8)
    cv2.ellipse(outline, (200, 150), (90, 60), 20, 0, 360, 255, 1)
    solid = np.zeros_like(outline)
    cv2.ellipse(solid, (200, 150), (90, 60), 20, 0, 360, 1, -1)
    filled = outline_to_filled_mask(outline)
    inter = np.logical_and(filled, solid).sum()
    assert 2 * inter / (filled.sum() + solid.sum()) > 0.98
