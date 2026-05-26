"""
Main entry point for the ForgeLens FastAPI application.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes.analyze import router as analyze_router
from backend.app.api.routes.health import router as health_router
from backend.app.core.config import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup event logic
    logger.info("ForgeLens API started on device: %s", settings.DEVICE)
    yield
    # Shutdown logic (if any)
    logger.info("ForgeLens API shutting down")


app = FastAPI(
    title="ForgeLens API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware configuration for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers under prefix /api/v1
app.include_router(analyze_router, prefix="/api/v1", tags=["Analysis"])
app.include_router(health_router, prefix="/api/v1", tags=["Health"])
