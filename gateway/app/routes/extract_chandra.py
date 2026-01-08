"""Chandra extraction endpoint for evaluation."""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..services.runpod_client import RunPodClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["extraction-chandra"])


class ExtractChandraRequest(BaseModel):
    """Chandra extraction request body."""
    image_urls: list[str] = Field(
        ...,
        min_length=1,
        description="ShareFile pre-signed URLs for document pages"
    )
    doc_type: str = Field(
        ...,
        description="Document type (W2, bank_statement, etc.)"
    )


class ConfidenceResponse(BaseModel):
    """Per-field confidence scores."""
    overall: float = Field(..., ge=0.0, le=1.0, description="Overall extraction confidence")
    fields: dict[str, Any] = Field(
        ...,
        description="Per-field confidence scores using dot notation keys"
    )


class ExtractChandraResponse(BaseModel):
    """Chandra extraction response body."""
    data: dict[str, Any] = Field(..., description="Extracted structured data")
    confidence: ConfidenceResponse = Field(..., description="Confidence scores")
    doc_type: str = Field(..., description="Document type used for extraction")
    page_count: int = Field(..., description="Number of pages processed")
    latency_ms: int = Field(..., description="Inference latency in milliseconds")
    model: str = Field(default="chandra", description="Model used for extraction")


def get_runpod_client(request: Request) -> RunPodClient:
    """Get shared RunPod client from app state."""
    return request.app.state.runpod_client


@router.post("/extract-chandra", response_model=ExtractChandraResponse)
async def extract_document_chandra(
    request: Request,
    body: ExtractChandraRequest
) -> ExtractChandraResponse:
    """
    Extract structured data using Chandra model (evaluation endpoint).

    This endpoint is for evaluating Chandra OCR accuracy compared to
    the primary dots.ocr extraction endpoint. Returns the same response
    schema for direct comparison.
    """
    client = get_runpod_client(request)

    # Call Chandra extraction endpoint
    response = await client.extract_chandra(
        image_urls=body.image_urls,
        doc_type=body.doc_type
    )

    if not response.success:
        logger.error(f"Chandra extraction failed: {response.error}")
        raise HTTPException(
            status_code=502,
            detail=f"Inference failed: {response.error}"
        )

    # Extract result
    data = response.data or {}

    # Build confidence response
    raw_confidence = data.get("confidence", {})
    confidence = ConfidenceResponse(
        overall=raw_confidence.get("overall", 0.5),
        fields=raw_confidence.get("fields", {})
    )

    return ExtractChandraResponse(
        data=data.get("data", {}),
        confidence=confidence,
        doc_type=data.get("doc_type", body.doc_type),
        page_count=data.get("page_count", len(body.image_urls)),
        latency_ms=response.latency_ms,
        model="chandra"
    )
