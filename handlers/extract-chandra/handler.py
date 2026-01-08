"""
RunPod Serverless Handler for Document Data Extraction using Chandra.

Model: datalab-to/chandra
Framework: vLLM
Input: ShareFile pre-signed URLs + document type
Output: Structured extracted data with per-field confidence

Chandra is a document understanding model that outputs structured
Markdown, which is then parsed and transformed into our document schemas.
"""

import asyncio
import base64
import logging
import os
import time
from io import BytesIO
from typing import Any

import nest_asyncio
nest_asyncio.apply()  # Allow nested asyncio.run() calls

import httpx
import requests
import runpod
from PIL import Image

from chandra_parser import parse_chandra_output, combine_page_results
from doc_transformers import transform_to_document_schema

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# vLLM server configuration
VLLM_PORT = os.environ.get('VLLM_PORT', '8080')
VLLM_BASE_URL = f"http://localhost:{VLLM_PORT}/v1"
MODEL_NAME = os.environ.get('MODEL_NAME', 'datalab-to/chandra')

# Chandra prompt for document extraction
CHANDRA_PROMPT = "Convert this document to Markdown, preserving all text, tables, and structure."


def fetch_image(url: str, timeout: int = 30) -> Image.Image:
    """
    Fetch an image from a URL (ShareFile pre-signed URL).

    Args:
        url: Pre-signed download URL
        timeout: Request timeout in seconds

    Returns:
        PIL Image object

    Raises:
        ValueError: If image cannot be fetched or decoded
    """
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert('RGB')
    except requests.RequestException as e:
        raise ValueError(f"Failed to fetch image: {e}")
    except Exception as e:
        raise ValueError(f"Failed to decode image: {e}")


def image_to_base64(image: Image.Image) -> str:
    """
    Convert PIL Image to base64 string.

    Args:
        image: PIL Image object

    Returns:
        Base64-encoded PNG string
    """
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


async def call_vllm(
    image_b64: str,
    max_tokens: int = 8192
) -> dict:
    """
    Call vLLM server with Chandra model.

    Args:
        image_b64: Base64-encoded image
        max_tokens: Maximum tokens to generate

    Returns:
        Dict with raw_output and usage keys

    Raises:
        ValueError: If vLLM call fails
    """
    async with httpx.AsyncClient(timeout=120.0) as client:
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": CHANDRA_PROMPT
                        }
                    ]
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,  # Deterministic for OCR
        }

        try:
            response = await client.post(
                f"{VLLM_BASE_URL}/chat/completions",
                json=payload
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise ValueError(f"vLLM request failed: {e}")
        except httpx.RequestError as e:
            raise ValueError(f"vLLM connection error: {e}")

        result = response.json()
        content = result["choices"][0]["message"]["content"]

        return {
            "raw_output": content,
            "usage": result.get("usage", {})
        }


async def extract(
    image_urls: list[str],
    doc_type: str
) -> dict:
    """
    Extract structured data from document images using Chandra.

    Args:
        image_urls: List of ShareFile pre-signed URLs
        doc_type: Document type for post-processing transformation

    Returns:
        Dict with data, confidence, and page_count

    Raises:
        ValueError: If no images can be processed
    """
    if not image_urls:
        raise ValueError("No image URLs provided")

    # Process each page
    page_results = []

    for i, url in enumerate(image_urls):
        try:
            logger.info(f"Processing page {i+1}/{len(image_urls)}")

            # Fetch and encode image
            image = fetch_image(url)
            image_b64 = image_to_base64(image)

            # Call Chandra via vLLM
            result = await call_vllm(image_b64)

            # Parse Markdown output
            parsed = parse_chandra_output(result["raw_output"])
            parsed["page"] = i + 1
            parsed["usage"] = result.get("usage", {})

            page_results.append(parsed)

        except ValueError as e:
            logger.warning(f"Skipping page {i+1}: {e}")
            continue
        except httpx.RequestError as e:
            logger.error(f"vLLM connection error on page {i+1}: {e}")
            continue
        except httpx.HTTPStatusError as e:
            logger.error(f"vLLM HTTP error on page {i+1}: {e}")
            continue
        except Exception as e:
            logger.exception(f"Unexpected error processing page {i+1}: {e}")
            continue

    if not page_results:
        raise ValueError("Could not process any images")

    # Combine results from all pages
    combined_parsed = combine_page_results(page_results)

    # Transform to document-specific schema with confidence
    structured_data, confidence = transform_to_document_schema(
        doc_type=doc_type,
        parsed=combined_parsed
    )

    return {
        "data": structured_data,
        "raw_ocr": {
            "pages": page_results,
            "combined_markdown": combined_parsed.get("raw_text", "")
        },
        "confidence": confidence,
        "page_count": len(page_results)
    }


def handler(job: dict) -> dict:
    """
    RunPod serverless handler entry point.

    Expected input format:
    {
        "input": {
            "image_urls": ["https://..."],
            "doc_type": "bank_statement"
        }
    }

    Returns:
    {
        "data": { ... structured fields per doc_type schema ... },
        "raw_ocr": { ... raw Chandra Markdown output ... },
        "confidence": {
            "overall": 0.89,
            "fields": {
                "header.bank_name": 0.90,
                "transactions": [{"date": 0.88, "description": 0.85, "amount": 0.90}],
                ...
            }
        },
        "page_count": 2,
        "doc_type": "bank_statement",
        "model": "chandra",
        "latency_ms": 3456
    }
    """
    start_time = time.time()

    try:
        input_data = job.get("input", {})

        image_urls = input_data.get("image_urls", [])
        doc_type = input_data.get("doc_type", "other")

        if not image_urls:
            return {"error": "No image_urls provided"}

        # Run async extraction (nest_asyncio allows nested asyncio.run)
        result = asyncio.run(extract(image_urls, doc_type))

        # Add metadata
        result["latency_ms"] = int((time.time() - start_time) * 1000)
        result["doc_type"] = doc_type
        result["model"] = "chandra"

        logger.info(
            f"Extracted {doc_type} ({result['page_count']} pages, "
            f"confidence={result['confidence']['overall']:.2f}) "
            f"in {result['latency_ms']}ms"
        )

        return result

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.exception(f"Handler error: {e}")
        return {"error": f"Internal error: {str(e)}"}


# RunPod entry point
if __name__ == "__main__":
    logger.info("Starting Chandra extraction handler...")
    runpod.serverless.start({"handler": handler})
