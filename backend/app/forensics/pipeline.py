"""
End-to-end forensic analysis pipeline.

Orchestrates preprocessing → inference → Grad-CAM → overlay rendering and
returns a result dictionary that conforms to the ``ForensicReport`` schema
defined in ``schemas/response.py``.
"""

from __future__ import annotations

import base64
import logging
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from backend.app.core.config import settings
from backend.app.forensics.explainability.gradcam import compute_all_cams
from backend.app.forensics.explainability.overlay import (
    apply_forensic_overlay,
    composite_heatmap,
)
from backend.app.forensics.models.forgelens_model import load_model
from backend.app.forensics.preprocessors.ela import generate_ela
from backend.app.forensics.preprocessors.fft import generate_fft_noise_stack
from backend.app.forensics.preprocessors.metadata import extract_metadata

logger = logging.getLogger(__name__)

# Class labels in index order — must stay in sync with training config.
_CLASS_LABELS: list[str] = ["authentic", "manipulated", "ai_generated"]

# ImageNet normalisation constants
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

# ELA / FFT-Noise normalisation constants
_AUX_MEAN = 0.5
_AUX_STD = 0.25


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _encode_image_b64(arr: np.ndarray) -> str:
    """Encode a BGR uint8 NumPy array as a base64-encoded PNG string.

    Parameters
    ----------
    arr : np.ndarray
        ``(H, W, 3)`` BGR uint8 image (OpenCV convention).

    Returns
    -------
    str
        Base64-encoded PNG prefixed with the data-URI header so it can be
        embedded directly in ``<img>`` tags or JSON responses.
    """
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------

def _to_normalised_tensor(
    arr: np.ndarray,
    mean: list[float] | float,
    std: list[float] | float,
    device: torch.device,
) -> torch.Tensor:
    """Convert a ``(H, W, C)`` uint8 array to a normalised ``(1, C, H, W)`` tensor."""
    t = torch.from_numpy(arr).float().div(255.0)  # (H, W, C) in [0, 1]
    t = t.permute(2, 0, 1)  # (C, H, W)

    if isinstance(mean, (int, float)):
        mean = [mean] * t.shape[0]
    if isinstance(std, (int, float)):
        std = [std] * t.shape[0]

    mean_t = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)
    t = (t - mean_t) / std_t

    return t.unsqueeze(0).to(device)  # (1, C, H, W)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class ForgeLensPipeline:
    """High-level pipeline: bytes in → forensic report out.

    Reads all configuration from :pydata:`core.config.settings` and
    instantiates the model via :func:`load_model`.
    """

    def __init__(self) -> None:
        self.device = torch.device(settings.DEVICE)

        checkpoint_path = str(
            Path(settings.MODEL_CHECKPOINT_DIR) / "phase3_best.pt"
        )
        self.model = load_model(checkpoint_path=checkpoint_path, device=str(self.device))
        self.model.eval()

        logger.info(
            "ForgeLensPipeline initialised  (device=%s, checkpoint=%s)",
            self.device,
            checkpoint_path,
        )

    # --------------------------------------------------------------------- #
    #  Main entry point                                                      #
    # --------------------------------------------------------------------- #

    def run(self, image_bytes: bytes, analysis_id: str) -> dict:
        """Execute the full forensic analysis pipeline.

        Parameters
        ----------
        image_bytes : bytes
            Raw image file content (JPEG / PNG / WebP).
        analysis_id : str
            Unique identifier for this analysis run (included in the
            returned report).

        Returns
        -------
        dict
            A dictionary conforming to the ``ForensicReport`` response
            schema.
        """

        # ----------------------------------------------------------------- #
        # Step 1 — Decode image                                             #
        # ----------------------------------------------------------------- #
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        rgb_arr = np.asarray(pil_image, dtype=np.uint8)            # (H, W, 3) RGB
        bgr_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)         # (H, W, 3) BGR

        # ----------------------------------------------------------------- #
        # Step 2 — Preprocess all branches                                  #
        # ----------------------------------------------------------------- #
        ela_arr = generate_ela(image_bytes)                         # (H, W, 3) uint8
        fft_noise_arr = generate_fft_noise_stack(bgr_arr)           # (H, W, 4) uint8
        meta = extract_metadata(image_bytes)                        # dict

        # Resize to model input size (224 × 224)
        rgb_resized = cv2.resize(rgb_arr, (224, 224), interpolation=cv2.INTER_LINEAR)
        ela_resized = cv2.resize(ela_arr, (224, 224), interpolation=cv2.INTER_LINEAR)
        fft_noise_resized = cv2.resize(
            fft_noise_arr, (224, 224), interpolation=cv2.INTER_LINEAR,
        )

        # ----------------------------------------------------------------- #
        # Step 3 — Tensorize                                                #
        # ----------------------------------------------------------------- #
        rgb_tensor = _to_normalised_tensor(
            rgb_resized, mean=_IMAGENET_MEAN, std=_IMAGENET_STD, device=self.device,
        )
        ela_tensor = _to_normalised_tensor(
            ela_resized, mean=_AUX_MEAN, std=_AUX_STD, device=self.device,
        )
        fft_tensor = _to_normalised_tensor(
            fft_noise_resized, mean=_AUX_MEAN, std=_AUX_STD, device=self.device,
        )
        meta_tensor = torch.tensor(
            [meta["feature_vector"]], dtype=torch.float32, device=self.device,
        )  # (1, 16)

        # ----------------------------------------------------------------- #
        # Step 4 — Inference (no gradients needed here)                     #
        # ----------------------------------------------------------------- #
        with torch.no_grad():
            logits, suspicion_score = self.model(
                rgb_tensor, ela_tensor, fft_tensor, meta_tensor,
            )
            probabilities = F.softmax(logits, dim=1)  # (1, num_classes)

        predicted_idx = int(probabilities.argmax(dim=1).item())
        predicted_label = _CLASS_LABELS[predicted_idx]
        confidence = float(probabilities[0, predicted_idx].item())
        suspicion = float(suspicion_score[0].item())

        class_probabilities = {
            label: float(probabilities[0, i].item())
            for i, label in enumerate(_CLASS_LABELS)
        }

        # ----------------------------------------------------------------- #
        # Step 5 — Grad-CAM (needs gradients — run outside no_grad)         #
        # ----------------------------------------------------------------- #

        # Re-run forward with gradients enabled so hooks can capture them
        self.model.zero_grad()
        logits_grad, _ = self.model(rgb_tensor, ela_tensor, fft_tensor, meta_tensor)

        cam_maps = compute_all_cams(self.model, logits_grad, predicted_idx)
        fused_heatmap = composite_heatmap(
            cam_maps["rgb"], cam_maps["ela"], cam_maps["fft"],
        )

        # Overlay onto the 224 × 224 RGB image (converted to BGR for cv2)
        bgr_resized = cv2.cvtColor(rgb_resized, cv2.COLOR_RGB2BGR)
        overlay_bgr = apply_forensic_overlay(bgr_resized, fused_heatmap)

        # ----------------------------------------------------------------- #
        # Step 6 — Build result dict (ForensicReport schema)                #
        # ----------------------------------------------------------------- #
        # Calculate anomaly scores (0.0 to 1.0)
        ela_anomaly_score = float(np.mean(ela_arr) / 255.0)
        fft_anomaly_score = float(np.mean(fft_noise_arr[:, :, 3]) / 255.0)
        noise_anomaly_score = float(np.mean(fft_noise_arr[:, :, :3]) / 255.0)
        
        # Metadata anomaly score: 1.0 if suspicious software, 0.5 if no exif, 0.0 otherwise
        if meta["suspicious_software"]:
            metadata_anomaly_score = 1.0
        elif not meta["has_exif"]:
            metadata_anomaly_score = 0.5
        else:
            metadata_anomaly_score = 0.0

        # Construct findings list
        findings = []
        if predicted_label == "ai_generated":
            findings.append("Generative AI signatures detected in image texture.")
        elif predicted_label == "manipulated":
            findings.append("Localized inconsistencies (splicing/copy-move) detected.")
        else:
            findings.append("No clear pixel-level manipulations detected.")

        if ela_anomaly_score > 0.15:
            findings.append("Error Level Analysis (ELA) indicates localized compression differences.")
        if fft_anomaly_score > 0.2:
            findings.append("Frequency-domain anomalies suggest image resizing or rotation.")
        if noise_anomaly_score > 0.2:
            findings.append("Noise-residual pattern inconsistency found in image planes.")
        if meta["suspicious_software"]:
            findings.append(f"Image was processed with editing software: {meta['software']}.")
        elif not meta["has_exif"]:
            findings.append("EXIF metadata is completely stripped, typical of tampered files.")

        # Replicate FFT single channel to 3 channels for encoding
        fft_single_channel = fft_noise_arr[:, :, 3]
        fft_visual = cv2.merge([fft_single_channel, fft_single_channel, fft_single_channel])

        result: dict = {
            "analysis_id": analysis_id,
            "verdict": predicted_label.upper(),  # "AI_GENERATED", "MANIPULATED", "AUTHENTIC"
            "confidence": round(confidence, 4),
            "suspicion_score": round(suspicion, 4),
            "class_probabilities": {
                k: round(v, 4) for k, v in class_probabilities.items()
            },
            "ela_anomaly_score": round(ela_anomaly_score, 4),
            "fft_anomaly_score": round(fft_anomaly_score, 4),
            "noise_anomaly_score": round(noise_anomaly_score, 4),
            "metadata_anomaly_score": round(metadata_anomaly_score, 4),
            "heatmap_b64": _encode_image_b64(overlay_bgr),
            "ela_image_b64": _encode_image_b64(ela_arr),
            "fft_image_b64": _encode_image_b64(fft_visual),
            "noise_image_b64": _encode_image_b64(fft_noise_arr[:, :, :3]),
            "metadata_report": meta,
            "forensic_findings": findings,
            "branch_contributions": {
                "rgb": 0.5,
                "ela": 0.3,
                "fft": 0.2
            },
            "inference_time_ms": 0.0,  # Will be populated by controller
        }

        logger.info(
            "Analysis %s complete — verdict=%s  confidence=%.2f  suspicion=%.2f",
            analysis_id,
            result["verdict"],
            confidence,
            suspicion,
        )

        return result
