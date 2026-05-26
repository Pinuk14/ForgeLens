"""
Top-level ForgeLens model and factory function.

Composes the three feature-extraction branches (RGB, ELA, FFT-Noise) with the
late-fusion classifier head into a single ``nn.Module`` ready for training or
inference.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from .branches import ELABranch, FFTNoiseBranch, RGBBranch
from .fusion import FusionHead

if TYPE_CHECKING:
    from torch import Tensor

logger = logging.getLogger(__name__)


class ForgeLensModel(nn.Module):
    """Multi-stream forgery-detection model.

    Architecture overview::

        RGB  image  ──► RGBBranch      ──► rgb_feat  (batch, 256)
        ELA  image  ──► ELABranch      ──► ela_feat  (batch, 256)
        FFT+Noise   ──► FFTNoiseBranch ──► fft_feat  (batch, 256)
        Metadata    ────────────────────► meta_feat (batch, 16)
                                              │
                                         FusionHead
                                         ╱        ╲
                                    logits     suspicion

    After each forward pass the intermediate spatial feature maps are stored
    as ``self.last_rgb_map``, ``self.last_ela_map``, and ``self.last_fft_map``
    so that Grad-CAM can hook into them without re-running the model.

    Parameters
    ----------
    freeze_backbones : bool
        Forwarded to every branch's ``freeze_backbone`` flag.
    num_classes : int
        Number of target classes (passed to :class:`FusionHead`).
    """

    def __init__(
        self,
        freeze_backbones: bool = True,
        num_classes: int = 3,
    ) -> None:
        super().__init__()

        self.rgb_branch = RGBBranch(freeze_backbone=freeze_backbones)
        self.ela_branch = ELABranch(freeze_backbone=freeze_backbones)
        self.fft_branch = FFTNoiseBranch(freeze_backbone=freeze_backbones)
        self.fusion = FusionHead(num_classes=num_classes)

        # Grad-CAM feature-map cache (populated on each forward pass)
        self.last_rgb_map: Tensor | None = None
        self.last_ela_map: Tensor | None = None
        self.last_fft_map: Tensor | None = None

    def forward(
        self,
        rgb: Tensor,
        ela: Tensor,
        fft_noise: Tensor,
        meta: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Run the full multi-stream pipeline.

        Parameters
        ----------
        rgb : Tensor
            ``(batch, 3, H, W)`` RGB image tensor.
        ela : Tensor
            ``(batch, 3, H, W)`` ELA image tensor.
        fft_noise : Tensor
            ``(batch, 4, H, W)`` FFT magnitude + noise-residual tensor.
        meta : Tensor
            ``(batch, 16)`` float-32 metadata feature vector (from
            ``preprocessors.metadata.feature_vector``).

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(logits, suspicion_score)``

            * **logits** — ``(batch, num_classes)`` raw class logits (no
              softmax).
            * **suspicion_score** — ``(batch,)`` scalar in ``[0.0, 1.0]``.
        """
        rgb_feat, self.last_rgb_map = self.rgb_branch(rgb)
        ela_feat, self.last_ela_map = self.ela_branch(ela)
        fft_feat, self.last_fft_map = self.fft_branch(fft_noise)

        logits, suspicion_score = self.fusion(rgb_feat, ela_feat, fft_feat, meta)
        return logits, suspicion_score


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def load_model(
    checkpoint_path: str | None = None,
    device: str = "cpu",
) -> ForgeLensModel:
    """Create a :class:`ForgeLensModel` and optionally load a checkpoint.

    Parameters
    ----------
    checkpoint_path : str | None
        Path to a ``.pt`` / ``.pth`` file containing a saved ``state_dict``.
        If *None* or the file does not exist, the model is returned with only
        its pretrained backbone weights (no crash).
    device : str
        Target device (e.g. ``"cpu"``, ``"cuda"``, ``"cuda:0"``).

    Returns
    -------
    ForgeLensModel
        The model in **eval** mode, moved to *device*.
    """
    model = ForgeLensModel(freeze_backbones=True)

    if checkpoint_path is not None and Path(checkpoint_path).is_file():
        state_dict = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(state_dict)
        logger.info("Loaded checkpoint from %s", checkpoint_path)
    else:
        if checkpoint_path is None:
            logger.warning(
                "No checkpoint path provided — using pretrained backbone "
                "weights only."
            )
        else:
            logger.warning(
                "Checkpoint not found at %s — using pretrained backbone "
                "weights only.",
                checkpoint_path,
            )

    model.eval()
    model.to(device)
    return model
