"""Error Level Analysis (ELA) preprocessor for image forgery detection."""

from io import BytesIO

import numpy as np
from PIL import Image


def generate_ela(
    image_bytes: bytes, quality: int = 90, scale: int = 10
) -> np.ndarray:
    """Generate an Error Level Analysis (ELA) map from raw image bytes.

    ELA works by re-saving an image at a known JPEG quality level and then
    computing the pixel-wise difference between the original and the
    recompressed version.  Regions that were previously saved at a different
    quality level (e.g. a spliced patch pasted into the image) will show
    noticeably higher error levels than the surrounding, uniformly-compressed
    background.

    A quality of 90 is the de-facto standard because it sits in the
    "sweet-spot" of the JPEG quantisation curve — high enough to preserve
    most detail, yet low enough for the quantisation artefacts introduced by
    a *second* compression pass to be clearly visible in tampered regions.
    Lower quality values amplify differences but also introduce noise in
    untampered areas; higher values suppress useful signal.

    Args:
        image_bytes: Raw image file content (e.g. from a FastAPI UploadFile).
        quality: JPEG quality level used for the recompression step.
                 Defaults to 90.
        scale: Multiplier applied to the absolute difference map so that
               subtle variations become visible.  Defaults to 10.

    Returns:
        A uint8 NumPy array of shape ``(H, W, 3)`` whose pixel intensities
        represent the scaled, clipped error levels.
    """

    original = Image.open(BytesIO(image_bytes)).convert("RGB")

    # If the source is a lossless format (e.g. PNG), it has never been through
    # JPEG quantisation, so every pixel will show a high error level.  To make
    # the ELA map meaningful we first "normalise" the image by compressing it
    # once at a high quality (95) — this simulates what a camera or editor
    # would produce — and then treat *that* as our baseline for the real ELA
    # pass at the requested quality.
    if _is_lossless(image_bytes):
        buf = BytesIO()
        original.save(buf, format="JPEG", quality=95)
        buf.seek(0)
        original = Image.open(buf).convert("RGB")

    # Re-save at the target quality into an in-memory buffer.
    resaved_buf = BytesIO()
    original.save(resaved_buf, format="JPEG", quality=quality)
    resaved_buf.seek(0)
    resaved = Image.open(resaved_buf).convert("RGB")

    # Compute the scaled absolute difference, clip to valid uint8 range.
    original_arr = np.asarray(original, dtype=np.float32)
    resaved_arr = np.asarray(resaved, dtype=np.float32)

    diff = np.abs(original_arr - resaved_arr) * scale
    ela_map = np.clip(diff, 0, 255).astype(np.uint8)

    return ela_map


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_lossless(image_bytes: bytes) -> bool:
    """Return True if *image_bytes* begins with a PNG or WebP-lossless header.

    This is a fast heuristic — we only inspect the magic bytes rather than
    fully parsing the file.  JPEG files start with ``\\xff\\xd8``, PNG files
    with the 8-byte PNG signature, and WebP files with ``RIFF…WEBP``.
    """
    if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return True
    # WebP container: bytes 0-3 == "RIFF", bytes 8-11 == "WEBP".
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return True
    return False
