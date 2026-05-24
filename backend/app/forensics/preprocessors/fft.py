"""FFT spectrum preprocessor and FFT+noise stacking utility."""

import cv2
import numpy as np

from backend.app.forensics.preprocessors.noise import generate_noise_residual


def generate_fft_spectrum(image: np.ndarray) -> np.ndarray:
    """Compute the centred, log-scaled FFT magnitude spectrum of a BGR image.

    Periodic patterns that are invisible in the spatial domain — such as
    resampling artefacts left by rotation, scaling, or copy-move operations —
    appear as bright peaks or regular grids in the frequency domain.  The
    log-magnitude spectrum therefore serves as a strong forgery cue,
    especially for geometric manipulations.

    Args:
        image: BGR uint8 ndarray as returned by ``cv2.imread``.

    Returns:
        A float32 ndarray of shape ``(H, W, 1)`` with values in ``[0, 1]``,
        representing the log-scaled magnitude spectrum with the
        zero-frequency component shifted to the centre.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # 2-D DFT → shift DC to centre → magnitude.
    dft = np.fft.fft2(gray)
    dft_shifted = np.fft.fftshift(dft)
    magnitude = np.abs(dft_shifted)

    # Log-scale to compress the dynamic range (add 1 to avoid log(0)).
    log_magnitude = np.log1p(magnitude)

    # Normalise to [0, 1].
    min_val = log_magnitude.min()
    max_val = log_magnitude.max()
    if max_val - min_val > 0:
        spectrum = (log_magnitude - min_val) / (max_val - min_val)
    else:
        spectrum = np.zeros_like(log_magnitude)

    return spectrum.astype(np.float32)[..., np.newaxis]  # (H, W, 1)


def generate_fft_noise_stack(image: np.ndarray) -> np.ndarray:
    """Build the 4-channel input tensor for the FFT branch model.

    Concatenates the 3-channel noise residual and the 1-channel FFT magnitude
    spectrum along the last axis, then scales the result to ``[0, 255]``
    uint8 so it can be treated like a regular image by downstream data
    loaders.

    Args:
        image: BGR uint8 ndarray as returned by ``cv2.imread``.

    Returns:
        A uint8 ndarray of shape ``(H, W, 4)`` — channels 0-2 are the
        noise residual and channel 3 is the FFT spectrum.
    """

    noise = generate_noise_residual(image)     # (H, W, 3) float32 [0, 1]
    fft = generate_fft_spectrum(image)          # (H, W, 1) float32 [0, 1]

    stacked = np.concatenate([noise, fft], axis=-1)  # (H, W, 4) float32

    # Scale to uint8.
    stacked_uint8 = np.clip(stacked * 255.0, 0, 255).astype(np.uint8)

    return stacked_uint8
