"""
Response schemas for the ForgeLens forensic analysis API.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class ForensicReport(BaseModel):
    """Complete forensic analysis result returned by the ``/analyse`` endpoint.

    Every image field (``heatmap_b64``, ``ela_image_b64``, etc.) is a
    base64-encoded PNG string with the ``data:image/png;base64,`` prefix so
    that front-ends can embed them directly in ``<img>`` tags.
    """

    # -- identification ---------------------------------------------------
    analysis_id: str = Field(
        ..., description="Unique identifier for this analysis run."
    )

    # -- verdict ----------------------------------------------------------
    verdict: Literal["AI_GENERATED", "MANIPULATED", "AUTHENTIC"] = Field(
        ..., description="Top-level classification label."
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability assigned to the predicted class (0.0–1.0).",
    )
    suspicion_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Overall suspicion score produced by the fusion head's sigmoid "
            "output.  0.0 = likely clean, 1.0 = highly suspicious."
        ),
    )
    class_probabilities: dict[str, float] = Field(
        ...,
        description=(
            'Per-class softmax probabilities.  Keys: "ai_generated", '
            '"manipulated", "authentic".'
        ),
    )

    # -- per-branch anomaly scores ----------------------------------------
    ela_anomaly_score: float = Field(
        ..., description="ELA branch anomaly intensity (0.0–1.0)."
    )
    fft_anomaly_score: float = Field(
        ..., description="FFT branch anomaly intensity (0.0–1.0)."
    )
    noise_anomaly_score: float = Field(
        ..., description="Noise-residual branch anomaly intensity (0.0–1.0)."
    )
    metadata_anomaly_score: float = Field(
        ..., description="Metadata-derived anomaly score (0.0–1.0)."
    )

    # -- encoded images ---------------------------------------------------
    heatmap_b64: str = Field(
        ..., description="Composite Grad-CAM heatmap overlay (base64 PNG)."
    )
    ela_image_b64: str = Field(
        ..., description="ELA visualisation image (base64 PNG)."
    )
    fft_image_b64: str = Field(
        ..., description="FFT magnitude spectrum image (base64 PNG)."
    )
    noise_image_b64: str = Field(
        ..., description="Noise-residual image (base64 PNG)."
    )

    # -- structured data --------------------------------------------------
    metadata_report: dict[str, Any] = Field(
        ..., description="Parsed EXIF / metadata dictionary."
    )
    forensic_findings: list[str] = Field(
        ...,
        description=(
            "Human-readable evidence list summarising the key forensic "
            "observations."
        ),
    )
    branch_contributions: dict[str, float] = Field(
        ...,
        description=(
            "Relative weight of each branch in the final verdict.  "
            'e.g. {"rgb": 0.5, "ela": 0.3, "fft": 0.2}.'
        ),
    )

    # -- timing -----------------------------------------------------------
    inference_time_ms: float = Field(
        ..., description="Wall-clock inference time in milliseconds."
    )

    # ------------------------------------------------------------------ #
    #  Validators                                                         #
    # ------------------------------------------------------------------ #

    @field_validator("confidence")
    @classmethod
    def _confidence_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {v}")
        return v

    @field_validator("suspicion_score")
    @classmethod
    def _suspicion_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(
                f"suspicion_score must be between 0.0 and 1.0, got {v}"
            )
        return v

    @model_validator(mode="after")
    def _class_probs_sum_to_one(self) -> ForensicReport:
        total = sum(self.class_probabilities.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"class_probabilities values must sum to ≈1.0 "
                f"(tolerance 0.01), got {total:.4f}"
            )
        return self
