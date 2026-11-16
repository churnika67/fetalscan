import numpy as np
import torch

from fetalscan.gradcam import GradCAM, overlay
from fetalscan.models import UNet, build_classifier, dice_loss, seg_loss


def test_classifier_grayscale_shape():
    m = build_classifier(6, pretrained=False)
    assert m(torch.zeros(2, 1, 96, 96)).shape == (2, 6)


def test_unet_shape_including_odd_size():
    m = UNet(base=8)
    assert m(torch.zeros(1, 1, 64, 64)).shape == (1, 1, 64, 64)
    assert m(torch.zeros(1, 1, 70, 90)).shape == (1, 1, 70, 90)


def test_dice_loss_perfect_vs_wrong():
    t = torch.zeros(1, 1, 8, 8)
    t[..., 2:6, 2:6] = 1
    good = (t * 20 - 10)
    bad = -good
    assert dice_loss(good, t) < 0.01
    assert dice_loss(bad, t) > 0.9
    assert seg_loss(good, t) < seg_loss(bad, t)


def test_unet_overfits_one_example():
    torch.manual_seed(0)
    x = torch.rand(1, 1, 32, 32)
    y = torch.zeros(1, 1, 32, 32)
    y[..., 8:24, 8:24] = 1
    m = UNet(base=8)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    for _ in range(60):
        opt.zero_grad()
        loss = seg_loss(m(x), y)
        loss.backward()
        opt.step()
    assert loss.item() < 0.3


def test_gradcam_heatmap():
    m = build_classifier(6, pretrained=False)
    cam = GradCAM(m, m.layer4)
    heat, k = cam(torch.rand(1, 1, 64, 64))
    cam.remove()
    assert heat.shape == (64, 64) and 0 <= k < 6
    assert heat.min() >= 0 and heat.max() <= 1 + 1e-6
    vis = overlay((np.random.rand(64, 64) * 255).astype(np.uint8), heat)
    assert vis.shape == (64, 64, 3)
