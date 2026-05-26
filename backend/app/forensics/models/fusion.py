"""
Fusion head that merges multi-branch features and produces final predictions.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class FusionHead(nn.Module):
    """Late-fusion classifier that combines RGB, ELA, FFT, and metadata features.

    Input layout (concatenated on ``dim=1``):

    ======== ========= ============================================
    Source   Dimension Notes
    ======== ========= ============================================
    RGB      256       From the RGB branch backbone
    ELA      256       From the ELA branch backbone
    FFT      256       From the FFT branch backbone
    Metadata 16        Float-32 vector from
                       ``preprocessors.metadata.feature_vector``
    ======== ========= ============================================

    Total input dimension: **784**.

    Outputs
    -------
    logits : Tensor, shape ``(batch, num_classes)``
        Raw class logits — **no** softmax is applied; the loss function
        (e.g. ``CrossEntropyLoss``) handles that internally.
    suspicion : Tensor, shape ``(batch,)``
        Scalar suspicion score in ``[0.0, 1.0]`` (sigmoid-activated).

    Parameters
    ----------
    num_classes : int
        Number of target classes (default **3**).
    metadata_dim : int
        Dimensionality of the metadata feature vector (default **16**).
        Must match the output of ``preprocessors.metadata.feature_vector``.
    """

    def __init__(self, num_classes: int = 3, metadata_dim: int = 16) -> None:
        super().__init__()

        in_dim = 256 + 256 + 256 + metadata_dim  # 784

        self.fusion = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.3),
        )

        self.classifier = nn.Linear(256, num_classes)

        self.suspicion_head = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        rgb_feat: Tensor,
        ela_feat: Tensor,
        fft_feat: Tensor,
        meta_feat: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Run the fusion head.

        Parameters
        ----------
        rgb_feat : Tensor
            ``(batch, 256)`` features from the RGB branch.
        ela_feat : Tensor
            ``(batch, 256)`` features from the ELA branch.
        fft_feat : Tensor
            ``(batch, 256)`` features from the FFT branch.
        meta_feat : Tensor
            ``(batch, 16)`` float-32 feature vector produced by
            ``preprocessors.metadata.feature_vector``.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(logits, suspicion)`` — raw class logits and a per-sample
            suspicion score in ``[0.0, 1.0]``.
        """
        x = torch.cat([rgb_feat, ela_feat, fft_feat, meta_feat], dim=1)
        h = self.fusion(x)
        logits = self.classifier(h)
        suspicion = self.suspicion_head(h).squeeze(1)
        return logits, suspicion
