"""
Grad-CAM implementation for per-branch explainability.

Each branch exposes a ``self.features`` sequential whose final sub-layer is
used as the Grad-CAM target.  :func:`compute_all_cams` is the main entry
point — it builds a :class:`BranchGradCAM` for every branch and returns the
resulting heatmaps in a single dictionary.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

if TYPE_CHECKING:
    from torch import Tensor

    from backend.app.forensics.models.forgelens_model import ForgeLensModel


class BranchGradCAM:
    """Grad-CAM for a single feature-extraction branch.

    Parameters
    ----------
    branch : nn.Module
        One of ``RGBBranch``, ``ELABranch``, or ``FFTNoiseBranch``.
    target_layer : nn.Module
        The layer whose activations and gradients are captured (typically the
        last sub-layer of ``branch.features``).
    """

    def __init__(self, branch: nn.Module, target_layer: nn.Module) -> None:
        self._branch = branch
        self._target_layer = target_layer

        self._activations: Tensor | None = None
        self._gradients: Tensor | None = None

        # Register hooks -------------------------------------------------
        self._fwd_handle = target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

    # -- hook callbacks ---------------------------------------------------

    def _save_activation(
        self,
        _module: nn.Module,
        _input: tuple[Tensor, ...],
        output: Tensor,
    ) -> None:
        self._activations = output.detach()

    def _save_gradient(
        self,
        _module: nn.Module,
        _grad_input: tuple[Tensor | None, ...],
        grad_output: tuple[Tensor, ...],
    ) -> None:
        self._gradients = grad_output[0].detach()

    # -- public API -------------------------------------------------------

    def compute(self, logits: Tensor, class_idx: int) -> np.ndarray:
        """Compute a Grad-CAM heatmap for *class_idx*.

        Parameters
        ----------
        logits : Tensor
            ``(1, num_classes)`` model output (a single-sample batch).
        class_idx : int
            Index of the class to explain.

        Returns
        -------
        np.ndarray
            Float-32 heatmap of shape ``(224, 224)`` normalised to ``[0, 1]``.
        """
        # Back-propagate the target class score
        logits[0, class_idx].backward(retain_graph=True)

        if self._gradients is None or self._activations is None:
            raise RuntimeError(
                "Hooks did not capture activations / gradients.  Make sure a "
                "forward pass through the branch was executed before calling "
                "compute()."
            )

        # Channel weights via Global Average Pooling of gradients
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination → ReLU
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)

        # Resize to 224 × 224
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam.squeeze(0).squeeze(0)  # (224, 224)

        # Normalise to [0, 1]
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max - cam_min > 0:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = cam - cam_min  # all zeros

        return cam.cpu().numpy().astype(np.float32)

    # -- cleanup ----------------------------------------------------------

    def remove_hooks(self) -> None:
        """Remove the registered forward and backward hooks."""
        self._fwd_handle.remove()
        self._bwd_handle.remove()


# ---------------------------------------------------------------------------
# Convenience helper
# ---------------------------------------------------------------------------

def compute_all_cams(
    model: ForgeLensModel,
    logits: Tensor,
    class_idx: int,
) -> dict[str, np.ndarray]:
    """Compute Grad-CAM heatmaps for every branch in *model*.

    Parameters
    ----------
    model : ForgeLensModel
        The full multi-stream model (must have just completed a forward pass
        so that activations are available).
    logits : Tensor
        ``(1, num_classes)`` logits returned by the most recent forward pass.
    class_idx : int
        Target class index to explain.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping of ``{"rgb": cam, "ela": cam, "fft": cam}`` where each value
        is a float-32 ``(224, 224)`` array normalised to ``[0, 1]``.
    """
    branches = {
        "rgb": model.rgb_branch,
        "ela": model.ela_branch,
        "fft": model.fft_branch,
    }

    results: dict[str, np.ndarray] = {}
    cams: list[BranchGradCAM] = []

    for name, branch in branches.items():
        # Target the last sub-layer of each branch's feature extractor
        target_layer = branch.features[-1]
        gc = BranchGradCAM(branch, target_layer)
        cams.append(gc)
        results[name] = gc.compute(logits, class_idx)

    # Clean up hooks so they don't leak on repeated calls
    for gc in cams:
        gc.remove_hooks()

    return results
