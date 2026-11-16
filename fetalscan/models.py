"""Models: ResNet-18 plane classifier and a compact U-Net for head segmentation."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18


def build_classifier(num_classes: int, pretrained: bool = True) -> nn.Module:
    """ResNet-18 adapted to 1-channel ultrasound input.

    ImageNet weights expect RGB, so the first conv's weights are averaged over the RGB axis
    instead of being thrown away; this keeps the transfer-learning benefit for grayscale input.
    """
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    m = resnet18(weights=weights)
    old = m.conv1
    m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        m.conv1.weight.copy_(old.weight.mean(dim=1, keepdim=True))
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


class _DoubleConv(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """U-Net (Ronneberger et al., 2015) with 4 down/up stages; outputs 1-channel logits."""

    def __init__(self, in_ch: int = 1, out_ch: int = 1, base: int = 32):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 16]
        self.downs = nn.ModuleList([_DoubleConv(in_ch, c[0])] + [_DoubleConv(c[i], c[i + 1]) for i in range(4)])
        self.ups = nn.ModuleList([nn.ConvTranspose2d(c[i + 1], c[i], 2, stride=2) for i in reversed(range(4))])
        self.dec = nn.ModuleList([_DoubleConv(c[i] * 2, c[i]) for i in reversed(range(4))])
        self.head = nn.Conv2d(c[0], out_ch, 1)

    def forward(self, x):
        skips = []
        for i, down in enumerate(self.downs):
            x = down(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for up, dec in zip(self.ups, self.dec):
            x = up(x)
            skip = skips.pop()
            if x.shape[-2:] != skip.shape[-2:]:  # odd input sizes
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = dec(torch.cat([skip, x], dim=1))
        return self.head(x)


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(1, 2, 3))
    denom = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1 - (2 * inter + eps) / (denom + eps)).mean()


def seg_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """BCE + Dice: BCE gives stable pixel gradients, Dice handles foreground/background imbalance."""
    return F.binary_cross_entropy_with_logits(logits, target) + dice_loss(logits, target)
