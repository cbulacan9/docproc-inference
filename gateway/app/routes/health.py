"""Health check endpoints."""

import logging

from fastapi import APIRouter, Request
from pydantic import BaseModel

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


def get_runpod_client(request: Request) -> RunPodClient:
    """Get shared RunPod client from app state."""
    return request.app.state.runpod_client


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """
    Check health of gateway and inference endpoints.

    No authentication required.
    """
    client = get_runpod_client(request)
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


@router.get("/ready")
async def readiness_check():
    """Simple readiness check for load balancers."""
    return {"status": "ready"}


@router.get("/live")
async def liveness_check():
    """Simple liveness check for container orchestration."""
    return {"status": "live"}
