"""Grad-CAM (Selvaraju et al., 2017) for interpreting the plane classifier."""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self._acts = None
        self._grads = None
        self._h1 = target_layer.register_forward_hook(self._save_act)
        self._h2 = target_layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, _m, _i, out):
        self._acts = out.detach()

    def _save_grad(self, _m, _gi, go):
        self._grads = go[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> tuple[np.ndarray, int]:
        """x: 1xCxHxW. Returns (heatmap HxW in [0,1], class index explained)."""
        self.model.eval()
        x = x.requires_grad_(True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(1).item())
        self.model.zero_grad(set_to_none=True)
        logits[0, class_idx].backward()
        weights = self._grads.mean(dim=(2, 3), keepdim=True)  # global-average-pooled gradients
        cam = F.relu((weights * self._acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy(), class_idx

    def remove(self):
        self._h1.remove()
        self._h2.remove()


def overlay(img_gray: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    base = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(base, 1 - alpha, heat, alpha, 0)
