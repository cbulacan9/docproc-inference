"""
RunPod Serverless Handler for Document Data Extraction.

Model: dots.ocr (rednote-hilab/dots.ocr)
Framework: vLLM 0.11.0+ (native support)
Input: ShareFile pre-signed URLs + document type
Output: Structured extracted data with layout information

dots.ocr is a SOTA document parsing model that provides:
- Layout detection with bounding boxes
- OCR with reading order preservation
- Table structure recognition
- Multi-language support (100+ languages)
"""

import asyncio
import base64
import json
import logging
import os
import re
import time
from io import BytesIO
from typing import Any

import nest_asyncio
nest_asyncio.apply()  # Allow nested asyncio.run() calls

import httpx
import requests
import runpod
from PIL import Image

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
MODEL_NAME = os.environ.get('MODEL_NAME', 'rednote-hilab/dots.ocr')

# dots.ocr prompt modes
PROMPT_MODES = {
    "layout_all": "Parse this document image. Extract all text with layout structure.",
    "ocr": "Extract all text from this document image.",
    "table": "Extract tables from this document image with structure preserved.",
}


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
    prompt_mode: str = "layout_all",
    max_tokens: int = 16384
) -> dict:
    """
    Call vLLM server with dots.ocr model.

    Args:
        image_b64: Base64-encoded image
        prompt_mode: dots.ocr prompt mode key
        max_tokens: Maximum tokens to generate

    Returns:
        Dict with raw_output and usage keys

    Raises:
        ValueError: If vLLM call fails
    """
    if prompt_mode not in PROMPT_MODES:
        logger.warning(f"Invalid prompt_mode '{prompt_mode}', falling back to 'layout_all'")
    prompt = PROMPT_MODES.get(prompt_mode, PROMPT_MODES["layout_all"])

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
                            "text": prompt
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


def parse_dots_ocr_output(raw_output: str) -> dict:
    """
    Parse dots.ocr output into structured format.

    dots.ocr returns structured data with layout elements.
    Each element has: bbox, category, text, and optional confidence.

    Args:
        raw_output: Raw text output from dots.ocr

    Returns:
        Parsed structure with layout_elements and raw_text
    """
    if not raw_output or not raw_output.strip():
        return {
            "layout_elements": [],
            "raw_text": ""
        }

    # Try to extract JSON if present
    try:
        json_match = re.search(r'\{[\s\S]*\}', raw_output)
        if json_match:
            parsed = json.loads(json_match.group())
            # Ensure expected structure
            if "layout_elements" not in parsed:
                parsed = {
                    "layout_elements": parsed.get("elements", []),
                    "raw_text": raw_output
                }
            return parsed
    except json.JSONDecodeError:
        pass

    # Parse line-by-line format
    lines = raw_output.strip().split('\n')

    elements = []
    current_text = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check for bbox markers (dots.ocr format: [x1,y1,x2,y2] text)
        bbox_match = re.match(r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\s*(.*)', line)
        if bbox_match:
            x1, y1, x2, y2 = map(int, bbox_match.groups()[:4])
            text = bbox_match.group(5)
            elements.append({
                "bbox": [x1, y1, x2, y2],
                "category": "Text",
                "text": text,
                "confidence": 0.95
            })
        else:
            # Plain text line
            current_text.append(line)

    # If no structured elements found, return as raw text
    if not elements and current_text:
        return {
            "layout_elements": [{
                "category": "Text",
                "text": '\n'.join(current_text),
                "confidence": 0.90
            }],
            "raw_text": '\n'.join(current_text)
        }

    return {
        "layout_elements": elements,
        "raw_text": raw_output
    }


def estimate_confidence_from_elements(elements: list) -> float:
    """
    Estimate overall confidence from layout elements.

    Args:
        elements: List of layout elements from dots.ocr

    Returns:
        Average confidence score
    """
    if not elements:
        return 0.5

    confidences = [
        el.get("confidence", 0.85)
        for el in elements
        if isinstance(el, dict)
    ]

    if not confidences:
        return 0.85

    return sum(confidences) / len(confidences)


async def extract(
    image_urls: list[str],
    doc_type: str,
    prompt_mode: str = "layout_all"
) -> dict:
    """
    Extract structured data from document images using dots.ocr.

    Args:
        image_urls: List of ShareFile pre-signed URLs
        doc_type: Document type for post-processing transformation
        prompt_mode: dots.ocr prompt mode

    Returns:
        Dict with data, raw_ocr, confidence, and page_count

    Raises:
        ValueError: If no images can be processed
    """
    if not image_urls:
        raise ValueError("No image URLs provided")

    # Process each page
    all_elements = []
    page_results = []

    for i, url in enumerate(image_urls):
        try:
            logger.info(f"Processing page {i+1}/{len(image_urls)}")

            # Fetch and encode image
            image = fetch_image(url)
            image_b64 = image_to_base64(image)

            # Call dots.ocr via vLLM
            result = await call_vllm(image_b64, prompt_mode)

            # Parse output
            parsed = parse_dots_ocr_output(result["raw_output"])
            parsed["page"] = i + 1
            parsed["usage"] = result.get("usage", {})

            page_results.append(parsed)
            all_elements.extend(parsed.get("layout_elements", []))

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

    # Combine all raw text
    combined_text = '\n\n'.join(
        p.get("raw_text", "") for p in page_results
    )

    # Transform raw OCR to document-specific schema
    structured_data = transform_to_document_schema(
        doc_type=doc_type,
        layout_elements=all_elements,
        raw_text=combined_text
    )

    # Calculate confidence
    overall_confidence = estimate_confidence_from_elements(all_elements)

    return {
        "data": structured_data,
        "raw_ocr": {
            "pages": page_results,
            "all_elements": all_elements,
            "combined_text": combined_text
        },
        "confidence": {
            "overall": overall_confidence,
            "page_count": len(page_results),
            "element_count": len(all_elements)
        },
        "page_count": len(page_results)
    }


def handler(job: dict) -> dict:
    """
    RunPod serverless handler entry point.

    Expected input format:
    {
        "input": {
            "image_urls": ["https://..."],
            "doc_type": "bank_statement",
            "prompt_mode": "layout_all"  # optional
        }
    }

    Returns:
    {
        "data": { ... structured fields per doc_type schema ... },
        "raw_ocr": { ... raw dots.ocr output ... },
        "confidence": { "overall": 0.95, ... },
        "page_count": 2,
        "doc_type": "bank_statement",
        "latency_ms": 1234
    }
    """
    start_time = time.time()

    try:
        input_data = job.get("input", {})

        image_urls = input_data.get("image_urls", [])
        doc_type = input_data.get("doc_type", "other")
        prompt_mode = input_data.get("prompt_mode", "layout_all")

        if not image_urls:
            return {"error": "No image_urls provided"}

        # Run async extraction (nest_asyncio allows nested asyncio.run)
        result = asyncio.run(extract(image_urls, doc_type, prompt_mode))

        # Add metadata
        result["latency_ms"] = int((time.time() - start_time) * 1000)
        result["doc_type"] = doc_type
        result["model"] = MODEL_NAME

        logger.info(
            f"Extracted {doc_type} ({result['page_count']} pages, "
            f"{result['confidence']['element_count']} elements) "
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
    logger.info("Starting dots.ocr extraction handler...")
    runpod.serverless.start({"handler": handler})
