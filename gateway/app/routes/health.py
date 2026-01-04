"""Health check endpoints."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ..config import Settings, get_settings
from ..services.runpod_client import RunPodClient

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


class EndpointHealth(BaseModel):
    """Health status for a single endpoint."""
    status: str
    workers: dict | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    """Overall health response."""
    status: str
    gateway: str
    endpoints: dict[str, EndpointHealth]


@router.get("/health", response_model=HealthResponse)
async def health_check(
    settings: Annotated[Settings, Depends(get_settings)]
) -> HealthResponse:
    """
    Check health of gateway and inference endpoints.

    No authentication required.
    """
    client = RunPodClient(
        api_key=settings.runpod_api_key,
        classify_endpoint=settings.runpod_classify_endpoint,
        extract_endpoint=settings.runpod_extract_endpoint
    )

    try:
        endpoint_health = await client.health_check()

        # Determine overall status
        all_healthy = all(
            ep.get("status") in ("healthy", "idle")
            for ep in endpoint_health.values()
        )

        return HealthResponse(
            status="healthy" if all_healthy else "degraded",
            gateway="healthy",
            endpoints={
                name: EndpointHealth(**data)
                for name, data in endpoint_health.items()
            }
        )

    finally:
        await client.close()


@router.get("/ready")
async def readiness_check():
    """Simple readiness check for load balancers."""
    return {"status": "ready"}


@router.get("/live")
async def liveness_check():
    """Simple liveness check for container orchestration."""
    return {"status": "live"}
