"""Noise residual preprocessor for image forgery detection."""

import cv2
import numpy as np


def generate_noise_residual(
    image: np.ndarray, sigma: float = 2.0
) -> np.ndarray:
    """Extract the high-frequency noise residual from a BGR image.

    The noise residual is obtained by subtracting a Gaussian-blurred
    (low-pass filtered) version of the image from the original.  In an
    authentic photograph the residual is roughly uniform — sensor noise is
    spread evenly across the frame.  Spliced or in-painted regions often
    exhibit a markedly different noise pattern because they originate from a
    different camera sensor or have been processed with different filters.

    Args:
        image: BGR uint8 ndarray as returned by ``cv2.imread``.
        sigma: Standard deviation of the Gaussian kernel applied along both
               axes.  Larger values produce a stronger low-pass filter,
               retaining only coarser noise.  Defaults to 2.0.

    Returns:
        A float32 ndarray of shape ``(H, W, 3)`` with values in ``[0, 1]``
        representing the absolute, normalised noise residual per channel.
    """

    # Normalise to [0, 1] so the residual is scale-independent.
    image_f = image.astype(np.float32) / 255.0

    # Gaussian blur — kernel size of (0, 0) lets OpenCV derive it from sigma.
    blurred = cv2.GaussianBlur(image_f, (0, 0), sigmaX=sigma)

    residual = np.abs(image_f - blurred)

    # Normalise each channel independently to [0, 1] for maximum contrast.
    max_val = residual.max()
    if max_val > 0:
        residual = residual / max_val

    return residual.astype(np.float32)
