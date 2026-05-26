"""
API route for image forensic analysis.
"""

from __future__ import annotations

import time
import uuid
from fastapi import APIRouter, File, HTTPException, UploadFile, status

from backend.app.core.config import settings
from backend.app.forensics.pipeline import ForgeLensPipeline
from backend.app.schemas.response import ForensicReport

router = APIRouter()

# Instantiate the pipeline once at the module level
pipeline = ForgeLensPipeline()

ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}


@router.post(
    "/analyze",
    response_model=ForensicReport,
    summary="Analyze an image for manipulation or AI generation",
)
async def analyze_image(file: UploadFile = File(...)) -> ForensicReport:
    """Analyze an uploaded image using multi-branch neural network fusion.

    Checks:
    - Content-type validation (JPEG, PNG, WebP)
    - File size validation (up to MAX_IMAGE_SIZE_MB)
    """
    # 1. Validate MIME type
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported media type: {file.content_type}. "
                f"Only JPEG, PNG, and WebP images are allowed."
            ),
        )

    # 2. Validate file size
    # We must read/seek or check from headers/stream size to avoid downloading massive payloads in memory.
    # To check the actual size, we can read the file size or check if the file size exceeds limit.
    # Let's read a chunk or read the whole file, but check total bytes.
    # Since we need the full bytes anyway for pipeline processing, we read the bytes.
    start_time = time.perf_counter()
    image_bytes = await file.read()
    file_size_bytes = len(image_bytes)
    max_bytes = settings.MAX_IMAGE_SIZE_MB * 1024 * 1024

    if file_size_bytes > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File size ({file_size_bytes / (1024 * 1024):.2f} MB) "
                f"exceeds the maximum allowed limit of {settings.MAX_IMAGE_SIZE_MB} MB."
            ),
        )

    # 3. Generate analysis_id
    analysis_id = str(uuid.uuid4())

    # 4. Run pipeline
    try:
        result = pipeline.run(image_bytes, analysis_id)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Forensic analysis pipeline failed: {str(e)}",
        ) from e

    # 5. Calculate timing and inject
    end_time = time.perf_counter()
    inference_time_ms = (end_time - start_time) * 1000.0
    result["inference_time_ms"] = round(inference_time_ms, 2)

    return ForensicReport(**result)
