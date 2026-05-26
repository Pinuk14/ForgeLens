"""
Heatmap composition and forensic overlay utilities.

These functions sit downstream of :mod:`.gradcam` and turn per-branch
Grad-CAM arrays into a single visual explanation blended onto the original
image.
"""

from __future__ import annotations

import cv2
import numpy as np


def composite_heatmap(
    rgb_cam: np.ndarray,
    ela_cam: np.ndarray,
    fft_cam: np.ndarray,
    weights: tuple[float, float, float] = (0.5, 0.3, 0.2),
) -> np.ndarray:
    """Weighted average of per-branch Grad-CAM heatmaps.

    Parameters
    ----------
    rgb_cam, ela_cam, fft_cam : np.ndarray
        Float-32 arrays of shape ``(224, 224)`` normalised to ``[0, 1]``.
    weights : tuple[float, float, float]
        Blending weights for ``(rgb, ela, fft)`` respectively.  They do not
        need to sum to 1 — the result is re-normalised.

    Returns
    -------
    np.ndarray
        Float-32 composite heatmap of shape ``(224, 224)`` in ``[0, 1]``.
    """
    w_rgb, w_ela, w_fft = weights
    composite = w_rgb * rgb_cam + w_ela * ela_cam + w_fft * fft_cam

    # Normalise to [0, 1]
    c_min = composite.min()
    c_max = composite.max()
    if c_max - c_min > 0:
        composite = (composite - c_min) / (c_max - c_min)
    else:
        composite = composite - c_min  # all zeros

    return composite.astype(np.float32)


def apply_forensic_overlay(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Blend a heatmap onto *original_image* using the Inferno colour map.

    Parameters
    ----------
    original_image : np.ndarray
        BGR uint-8 image of arbitrary size ``(H, W, 3)``.
    heatmap : np.ndarray
        Float-32 heatmap in ``[0, 1]`` (any spatial size — it will be
        resized to match *original_image*).
    alpha : float
        Blending strength of the heatmap.  ``0.0`` = original only,
        ``1.0`` = heatmap only.

    Returns
    -------
    np.ndarray
        BGR uint-8 blended image with the same size as *original_image*.
    """
    h, w = original_image.shape[:2]

    # Resize heatmap to match original image dimensions
    heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

    # Convert to uint8 for colour mapping
    heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)

    # Apply INFERNO colour map (not JET — perceptually uniform, avoids
    # misleading rainbow artefacts)
    heatmap_colour = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_INFERNO)

    # Blend
    blended = cv2.addWeighted(original_image, 1.0 - alpha, heatmap_colour, alpha, 0)
    return blended.astype(np.uint8)
