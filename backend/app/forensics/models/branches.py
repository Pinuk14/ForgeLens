"""
Feature-extraction branches for multi-stream forgery detection.

Backbone selection rationale
----------------------------
* **EfficientNet-B0** (RGB & FFT-Noise branches) — offers an excellent
  accuracy-to-parameter ratio thanks to compound scaling.  Its 1280-dim
  final feature map is rich enough for forgery cues while remaining
  lightweight for real-time inference.

* **MobileNet-V3-Small** (ELA branch) — ELA highlights compression
  artefacts that are spatially coarse, so a heavier backbone is
  unnecessary.  MobileNet-V3-Small keeps latency low and its
  hard-swish / squeeze-excite blocks still capture the subtle
  intensity shifts present in ELA images.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights,
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
)

if TYPE_CHECKING:
    from torch import Tensor


# ---------------------------------------------------------------------------
# RGB branch – EfficientNet-B0
# ---------------------------------------------------------------------------

class RGBBranch(nn.Module):
    """Extract spatial features from the unmodified RGB image.

    Parameters
    ----------
    freeze_backbone : bool
        If *True*, all parameters in ``self.features`` are frozen so that
        only the projection head is trained.
    """

    def __init__(self, freeze_backbone: bool = True) -> None:
        super().__init__()

        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        self.features = backbone.features  # kept for Grad-CAM hook
        self.pool = backbone.avgpool

        self.proj = nn.Sequential(
            nn.Linear(1280, 256),
            nn.GELU(),
            nn.Dropout(0.3),
        )

        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(projected_features, feature_map)``.

        *feature_map* is the spatial output of ``self.features`` **before**
        global average pooling — useful for Grad-CAM visualisation.
        """
        feature_map: Tensor = self.features(x)
        pooled = self.pool(feature_map)
        pooled = torch.flatten(pooled, 1)
        projected = self.proj(pooled)
        return projected, feature_map


# ---------------------------------------------------------------------------
# ELA branch – MobileNet-V3-Small
# ---------------------------------------------------------------------------

class ELABranch(nn.Module):
    """Extract features from an Error Level Analysis (ELA) image.

    Parameters
    ----------
    freeze_backbone : bool
        If *True*, all parameters in ``self.features`` are frozen so that
        only the projection head is trained.
    """

    def __init__(self, freeze_backbone: bool = True) -> None:
        super().__init__()

        backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)

        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.proj = nn.Sequential(
            nn.Linear(576, 256),
            nn.GELU(),
            nn.Dropout(0.3),
        )

        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(projected_features, feature_map)``."""
        feature_map: Tensor = self.features(x)
        pooled = self.pool(feature_map)
        pooled = torch.flatten(pooled, 1)
        projected = self.proj(pooled)
        return projected, feature_map


# ---------------------------------------------------------------------------
# FFT / Noise branch – EfficientNet-B0 (4-channel input)
# ---------------------------------------------------------------------------

class FFTNoiseBranch(nn.Module):
    """Extract features from a 4-channel FFT + noise-residual tensor.

    The first convolutional layer of EfficientNet-B0 is patched to accept
    **4** input channels instead of 3.  Pretrained weights for channels 0-2
    are copied verbatim; channel 3 is initialised as the mean of the
    original three channels so the network starts from a sensible point.

    Parameters
    ----------
    freeze_backbone : bool
        If *True*, ``backbone.features[1:]`` are frozen.  Layer 0 (the
        modified convolution) is **always** left trainable so it can adapt
        to the novel fourth channel.
    """

    def __init__(self, freeze_backbone: bool = True) -> None:
        super().__init__()

        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        # --- patch first conv: 3 → 4 input channels -------------------
        old_conv: nn.Conv2d = backbone.features[0][0]
        new_conv = nn.Conv2d(
            in_channels=4,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None),
        )

        with torch.no_grad():
            # Copy pretrained weights for channels 0-2
            new_conv.weight[:, :3, :, :] = old_conv.weight
            # Initialise channel 3 as the mean of the original 3 channels
            new_conv.weight[:, 3:4, :, :] = old_conv.weight.mean(dim=1, keepdim=True)
            if old_conv.bias is not None:
                new_conv.bias = copy.deepcopy(old_conv.bias)

        backbone.features[0][0] = new_conv

        self.features = backbone.features
        self.pool = backbone.avgpool

        self.proj = nn.Sequential(
            nn.Linear(1280, 256),
            nn.GELU(),
            nn.Dropout(0.3),
        )

        if freeze_backbone:
            # Layer 0 was modified — keep it trainable
            for param in self.features[1:].parameters():
                param.requires_grad = False

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(projected_features, feature_map)``."""
        feature_map: Tensor = self.features(x)
        pooled = self.pool(feature_map)
        pooled = torch.flatten(pooled, 1)
        projected = self.proj(pooled)
        return projected, feature_map
