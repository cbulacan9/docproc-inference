"""Classification endpoint."""

import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..services.runpod_client import RunPodClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["classification"])


class ClassifyRequest(BaseModel):
    """Classification request body."""
    image_urls: list[str] = Field(
        ...,
        min_length=1,
        description="ShareFile pre-signed URLs for document pages"
    )
    prompt: str | None = Field(
        None,
        description="Optional custom classification prompt"
    )


class ClassifyResponse(BaseModel):
    """Classification response body."""
    type: str = Field(..., description="Predicted document type")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
    reasoning: str = Field("", description="Model's reasoning")
    latency_ms: int = Field(..., description="Inference latency in milliseconds")


def get_runpod_client(request: Request) -> RunPodClient:
    """Get shared RunPod client from app state."""
    return request.app.state.runpod_client


@router.post("/classify", response_model=ClassifyResponse)
async def classify_document(
    request: Request,
    body: ClassifyRequest
) -> ClassifyResponse:
    """
    Classify a document from page images.

    Returns the predicted document type and confidence score.
    """
    client = get_runpod_client(request)

    # Call classification endpoint
    response = await client.classify(
        image_urls=body.image_urls,
        prompt=body.prompt
    )

    if not response.success:
        logger.error(f"Classification failed: {response.error}")
        raise HTTPException(
            status_code=502,
            detail=f"Inference failed: {response.error}"
        )

    # Extract result
    data = response.data or {}

    return ClassifyResponse(
        type=data.get("type", "other"),
        confidence=data.get("confidence", 0.5),
        reasoning=data.get("reasoning", ""),
        latency_ms=response.latency_ms
    )
