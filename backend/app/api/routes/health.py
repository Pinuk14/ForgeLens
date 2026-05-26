"""
API route for health and diagnostics check.
"""

from __future__ import annotations

from fastapi import APIRouter

from backend.app.core.config import settings

router = APIRouter()


@router.get(
    "/health",
    summary="Health check endpoint",
)
async def health_check() -> dict:
    """Return the health status of the API and target computation device."""
    return {
        "status": "ok",
        "device": settings.DEVICE,
    }
