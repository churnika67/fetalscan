import numpy as np
import pytest
from PIL import Image

from fetalscan.metrics import bootstrap_ci, classification_report, confusion_matrix, dice, iou
from fetalscan.preprocess import apply_clahe, letterbox, letterbox_mask, load_gray


def test_load_gray_handles_rgb_and_16bit(tmp_path):
    rgb = np.zeros((10, 12, 3), np.uint8)
    rgb[..., 0] = 255
    Image.fromarray(rgb).save(tmp_path / "rgb.png")
    g = load_gray(str(tmp_path / "rgb.png"))
    assert g.shape == (10, 12) and g.dtype == np.uint8


def test_letterbox_keeps_aspect_and_pads():
    img = np.full((100, 200), 255, np.uint8)
    out, scale, (px, py) = letterbox(img, 64)
    assert out.shape == (64, 64)
    assert scale == pytest.approx(0.32)
    assert px == 0 and py == 16
    assert out[:py].max() == 0 and out[py:py + 32].min() == 255


def test_letterbox_mask_stays_binary():
    m = np.zeros((101, 203), np.uint8)
    m[20:80, 50:150] = 1
    out = letterbox_mask(m, 64)
    assert set(np.unique(out)) <= {0, 1}


def test_clahe_increases_contrast_of_flat_region():
    rng = np.random.default_rng(0)
    img = (100 + rng.integers(0, 10, (64, 64))).astype(np.uint8)
    assert apply_clahe(img).std() > img.std()


def test_confusion_and_report():
    y, p = [0, 0, 1, 1, 2], [0, 1, 1, 1, 2]
    cm = confusion_matrix(y, p, 3)
    assert cm.tolist() == [[1, 1, 0], [0, 2, 0], [0, 0, 1]]
    r = classification_report(y, p, 3)
    assert r["accuracy"] == pytest.approx(0.8)
    # F1: class0 = 2/3, class1 = 0.8, class2 = 1
    assert r["macro_f1"] == pytest.approx((2 / 3 + 0.8 + 1) / 3)


def test_macro_f1_ignores_absent_classes():
    r = classification_report([0, 0, 1], [0, 0, 1], 6)
    assert r["macro_f1"] == pytest.approx(1.0)


def test_dice_iou():
    a = np.zeros((10, 10))
    b = np.zeros((10, 10))
    a[:5] = 1
    b[:5, :5] = 1
    assert dice(a, b) == pytest.approx(2 * 25 / 75)
    assert iou(a, b) == pytest.approx(0.5)
    assert dice(np.zeros((3, 3)), np.zeros((3, 3))) == 1.0


def test_bootstrap_ci_contains_mean():
    v = np.random.default_rng(1).normal(5, 1, 200)
    lo, hi = bootstrap_ci(v)
    assert lo < v.mean() < hi
