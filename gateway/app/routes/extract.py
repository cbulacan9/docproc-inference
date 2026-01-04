"""Extraction endpoint."""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..services.runpod_client import RunPodClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["extraction"])


class ExtractRequest(BaseModel):
    """Extraction request body."""
    image_urls: list[str] = Field(
        ...,
        min_length=1,
        description="ShareFile pre-signed URLs for document pages"
    )
    doc_type: str = Field(
        ...,
        description="Document type (W2, bank_statement, etc.)"
    )
    prompt: str | None = Field(
        None,
        description="Optional custom extraction prompt"
    )


class ExtractResponse(BaseModel):
    """Extraction response body."""
    data: dict[str, Any] = Field(..., description="Extracted structured data")
    confidence: dict[str, Any] = Field(..., description="Confidence scores")
    doc_type: str = Field(..., description="Document type used for extraction")
    page_count: int = Field(..., description="Number of pages processed")
    latency_ms: int = Field(..., description="Inference latency in milliseconds")


def get_runpod_client(request: Request) -> RunPodClient:
    """Get shared RunPod client from app state."""
    return request.app.state.runpod_client


@router.post("/extract", response_model=ExtractResponse)
async def extract_document(
    request: Request,
    body: ExtractRequest
) -> ExtractResponse:
    """
    Extract structured data from document images.

    Returns extracted fields with confidence scores.
    """
    client = get_runpod_client(request)

    # Call extraction endpoint
    response = await client.extract(
        image_urls=body.image_urls,
        doc_type=body.doc_type,
        prompt=body.prompt
    )

    if not response.success:
        logger.error(f"Extraction failed: {response.error}")
        raise HTTPException(
            status_code=502,
            detail=f"Inference failed: {response.error}"
        )

    # Extract result
    data = response.data or {}

    return ExtractResponse(
        data=data.get("data", {}),
        confidence=data.get("confidence", {}),
        doc_type=data.get("doc_type", body.doc_type),
        page_count=data.get("page_count", len(body.image_urls)),
        latency_ms=response.latency_ms
    )
