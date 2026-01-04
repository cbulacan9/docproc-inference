"""Classification endpoint."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..config import Settings, get_settings
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


@router.post("/classify", response_model=ClassifyResponse)
async def classify_document(
    request: ClassifyRequest,
    settings: Annotated[Settings, Depends(get_settings)]
) -> ClassifyResponse:
    """
    Classify a document from page images.

    Returns the predicted document type and confidence score.
    """
    # Create RunPod client
    client = RunPodClient(
        api_key=settings.runpod_api_key,
        classify_endpoint=settings.runpod_classify_endpoint,
        extract_endpoint=settings.runpod_extract_endpoint,
        timeout=settings.runpod_timeout,
        max_retries=settings.runpod_max_retries
    )

    try:
        # Call classification endpoint
        response = await client.classify(
            image_urls=request.image_urls,
            prompt=request.prompt
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

    finally:
        await client.close()
