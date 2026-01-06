# RunPod Serverless Implementation Guide

**Version:** 1.0  
**Created:** January 2025  
**Status:** Ready for Implementation  
**Estimated Effort:** 8-12 hours

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Model Selection](#3-model-selection)
4. [Classification Handler](#4-classification-handler)
5. [Extraction Handler](#5-extraction-handler)
6. [Gateway Server](#6-gateway-server)
7. [Deployment](#7-deployment)
8. [Testing](#8-testing)
9. [Cost Optimization](#9-cost-optimization)

---

## 1. Overview

### 1.1 Purpose

This document provides implementation details for deploying ML inference infrastructure on RunPod Serverless, including:

- **Classification Handler**: Qwen2.5-VL-7B for document type classification
- **Extraction Handler**: dots.ocr (1.7B) for structured data extraction (SOTA on OmniDocBench)
- **Gateway Server**: FastAPI service for authentication, routing, and monitoring

### 1.2 Why a Gateway?

While RunPod endpoints can be called directly, a gateway provides:

| Benefit | Description |
|---------|-------------|
| **Single Auth Point** | One API key for your system; RunPod keys stay server-side |
| **Request Logging** | Centralized logging for debugging and analytics |
| **Abstraction** | Swap RunPod for local GPU later without client changes |
| **Retry Logic** | Centralized retry/fallback handling |
| **Rate Limiting** | Protect against runaway costs |
| **Health Aggregation** | Single health endpoint for all inference services |

### 1.3 Repository Structure

```
inference/
├── handlers/
│   ├── classify/
│   │   ├── Dockerfile
│   │   ├── handler.py
│   │   ├── requirements.txt
│   │   └── test_handler.py
│   │
│   └── extract/
│       ├── Dockerfile
│       ├── handler.py
│       ├── requirements.txt
│       └── test_handler.py
│
├── gateway/
│   ├── Dockerfile
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── classify.py
│   │   │   ├── extract.py
│   │   │   └── health.py
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   └── runpod_client.py
│   │   └── middleware/
│   │       ├── __init__.py
│   │       ├── auth.py
│   │       └── logging.py
│   ├── requirements.txt
│   └── tests/
│       └── test_routes.py
│
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## 2. Architecture

### 2.1 Request Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           INFERENCE ARCHITECTURE                                │
└─────────────────────────────────────────────────────────────────────────────────┘

  Document Pipeline                Gateway (Your Server)              RunPod Serverless
  ─────────────────                ─────────────────────              ─────────────────

  ┌─────────────────┐             ┌─────────────────────┐
  │ classify_worker │             │                     │            ┌──────────────────┐
  │                 │────────────►│   /v1/classify      │───────────►│ classify-endpoint│
  │ (doc pipeline)  │   HTTPS     │                     │   HTTPS    │ (Qwen2.5-VL-7B) │
  └─────────────────┘             │                     │            └──────────────────┘
                                  │                     │
  ┌─────────────────┐             │   Gateway Server    │
  │ extract_worker  │             │   (FastAPI)         │            ┌──────────────────┐
  │                 │────────────►│                     │───────────►│ extract-endpoint │
  │ (doc pipeline)  │   HTTPS     │   /v1/extract       │   HTTPS    │ (dots.ocr 1.7B)  │
  └─────────────────┘             │                     │            └──────────────────┘
                                  │                     │
                                  │   - Auth (API Key)  │
                                  │   - Logging         │
                                  │   - Retry Logic     │
                                  │   - Rate Limiting   │
                                  └─────────────────────┘
                                           │
                                           ▼
                                  ┌─────────────────────┐
                                  │   ShareFile         │
                                  │   (Image URLs)      │◄─────────── RunPod fetches
                                  └─────────────────────┘             images directly
```

### 2.2 Sequence Diagram

```
┌──────────┐     ┌─────────┐     ┌────────┐     ┌───────────┐
│ Pipeline │     │ Gateway │     │ RunPod │     │ ShareFile │
└────┬─────┘     └────┬────┘     └───┬────┘     └─────┬─────┘
     │                │              │                │
     │ POST /classify │              │                │
     │ {image_urls,   │              │                │
     │  prompt}       │              │                │
     │───────────────►│              │                │
     │                │              │                │
     │                │ Validate     │                │
     │                │ API Key      │                │
     │                │              │                │
     │                │ POST /runsync│                │
     │                │─────────────►│                │
     │                │              │                │
     │                │              │ GET image_url  │
     │                │              │───────────────►│
     │                │              │                │
     │                │              │◄───────────────│
     │                │              │ image bytes    │
     │                │              │                │
     │                │              │ Run inference  │
     │                │              │                │
     │                │◄─────────────│                │
     │                │ {type, conf} │                │
     │                │              │                │
     │◄───────────────│              │                │
     │ {type, conf}   │              │                │
     │                │              │                │
```

---

## 3. Model Selection

### 3.1 Decision: Two Specialized Models

After evaluating options, I recommend a **two-model architecture** optimized for each task:

| Task | Model | Parameters | VRAM | Rationale |
|------|-------|------------|------|-----------|}
| **Classification** | Qwen2.5-VL-7B-Instruct | 7B | ~16GB | General VLM, good at document type recognition |
| **Extraction** | dots.ocr | 1.7B | ~8GB | SOTA OCR model, beats GPT-4o on OmniDocBench |

### 3.2 Why dots.ocr for Extraction?

[dots.ocr](https://github.com/rednote-hilab/dots.ocr) is a specialized document parsing model from Rednote that achieves state-of-the-art results:

| Benchmark | dots.ocr | GPT-4o | Qwen2.5-VL-72B | Gemini 2.5 Pro |
|-----------|----------|--------|----------------|----------------|
| OmniDocBench Overall (EN) | **0.125** | 0.233 | 0.252 | 0.148 |
| OmniDocBench Overall (ZH) | **0.160** | 0.399 | 0.327 | 0.212 |
| Table TEDS (EN) | **88.6%** | 72.0% | 76.8% | 85.8% |
| Text Edit (EN) | **0.032** | 0.144 | 0.096 | 0.055 |

**Key advantages:**
- **Compact**: 1.7B LLM backbone = fast inference, low VRAM
- **Structured output**: Returns JSON with bounding boxes, categories, and text
- **Native vLLM support**: Officially integrated since vLLM 0.11.0
- **Layout-aware**: Maintains reading order, handles tables/formulas
- **Multilingual**: Strong performance across 100+ languages

### 3.3 Model Details

**Classification Model:**
```
Model: Qwen/Qwen2.5-VL-7B-Instruct
Framework: vLLM
VRAM: ~16GB
Context: 32K tokens
Use case: Document type recognition (W2, 1099, bank statement, etc.)
```

**Extraction Model:**
```
Model: rednote-hilab/dots.ocr
Framework: vLLM 0.11.0+ (native support)
VRAM: ~8GB
Context: 24K tokens (max_new_tokens=24000)
Use case: Layout detection, OCR, table extraction
```

### 3.4 GPU Requirements

| Endpoint | Minimum GPU | Recommended | Est. Cost (RunPod) |
|----------|-------------|-------------|--------------------|
| Classification (Qwen2.5-VL-7B) | T4 (16GB) | L4 (24GB) | ~$0.00031/sec |
| Extraction (dots.ocr) | T4 (16GB) | L4 (24GB) | ~$0.00031/sec |

**Note:** Both models can share the same GPU type, simplifying deployment.

### 3.5 dots.ocr Prompt Modes

dots.ocr supports different prompts for different tasks:

| Prompt Mode | Use Case | Output |
|-------------|----------|--------|
| `prompt_layout_all_en` | Full parsing (default) | Layout + OCR + Tables |
| `prompt_layout_only_en` | Detection only | Bounding boxes only |
| `prompt_ocr` | Text extraction | Text without headers/footers |
| `prompt_grounding_ocr` | Region-specific OCR | Text from specific bbox |

For CPA document extraction, use `prompt_layout_all_en` to get structured data with layout information.

---

## 4. Classification Handler

### 4.1 Directory Structure

```
handlers/classify/
├── Dockerfile
├── handler.py
├── requirements.txt
├── prompts.py
└── test_handler.py
```

### 4.2 Dockerfile

```dockerfile
# handlers/classify/Dockerfile

FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# Install vLLM and dependencies
RUN pip install --no-cache-dir \
    vllm>=0.4.0 \
    transformers>=4.40.0 \
    accelerate>=0.27.0 \
    requests>=2.31.0 \
    pillow>=10.0.0 \
    runpod>=1.6.0

# Copy handler code
COPY handler.py .
COPY prompts.py .

# Pre-download model during build (optional but reduces cold start)
# Uncomment for production builds:
# RUN python -c "from vllm import LLM; LLM('Qwen/Qwen2.5-VL-7B-Instruct', trust_remote_code=True)"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/hf_cache
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn

CMD ["python", "-u", "handler.py"]
```

### 4.3 Handler Implementation

```python
# handlers/classify/handler.py
"""
RunPod Serverless Handler for Document Classification.

Model: Qwen2.5-VL-7B-Instruct
Input: ShareFile pre-signed URLs + classification prompt
Output: Document type, confidence, reasoning
"""

import base64
import json
import logging
import os
import re
import time
from io import BytesIO
from typing import Any

import requests
import runpod
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Global model instance (loaded once at cold start)
MODEL = None
PROCESSOR = None


def load_model():
    """Load the VLM model (called once at cold start)."""
    global MODEL, PROCESSOR
    
    if MODEL is not None:
        return
    
    logger.info("Loading Qwen2.5-VL-7B-Instruct...")
    start = time.time()
    
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    import torch
    
    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    
    MODEL = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    PROCESSOR = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    logger.info(f"Model loaded in {time.time() - start:.1f}s")


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


def build_classification_prompt() -> str:
    """Return the classification prompt."""
    return """Analyze this document image and classify it into one of these categories:

- W2: Employee wage and tax statement
- 1099-INT: Interest income statement
- 1099-DIV: Dividend income statement
- 1099-MISC: Miscellaneous income statement
- 1099-NEC: Non-employee compensation statement
- 1099-R: Retirement distributions statement
- 1098: Mortgage interest statement
- bank_statement: Bank account statement
- credit_card_statement: Credit card statement
- invoice: Invoice or bill
- receipt: Purchase receipt
- check: Check or payment stub
- other: None of the above

Respond ONLY with a JSON object in this exact format:
{
  "type": "<document_type>",
  "confidence": <0.0-1.0>,
  "reasoning": "<brief explanation>"
}

Do not include any other text."""


def parse_json_response(text: str) -> dict:
    """
    Extract JSON from model response.
    
    Handles responses with markdown code blocks or extra text.
    """
    # Try to find JSON in code blocks first
    patterns = [
        r'```json\s*(.*?)\s*```',
        r'```\s*(.*?)\s*```',
        r'(\{[\s\S]*\})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
    
    # Try parsing the whole response
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        logger.warning(f"Could not parse JSON from: {text[:200]}...")
        return {}


def validate_classification_result(result: dict) -> dict:
    """
    Validate and normalize classification result.
    
    Ensures type is valid and confidence is in range.
    """
    valid_types = [
        'W2', '1099-INT', '1099-DIV', '1099-MISC', '1099-NEC',
        '1099-R', '1098', 'bank_statement', 'credit_card_statement',
        'invoice', 'receipt', 'check', 'other'
    ]
    
    doc_type = result.get('type', 'other')
    if doc_type not in valid_types:
        logger.warning(f"Invalid doc type '{doc_type}', defaulting to 'other'")
        doc_type = 'other'
    
    confidence = result.get('confidence', 0.5)
    try:
        confidence = float(confidence)
        confidence = max(0.0, min(1.0, confidence))
    except (TypeError, ValueError):
        confidence = 0.5
    
    return {
        'type': doc_type,
        'confidence': confidence,
        'reasoning': result.get('reasoning', '')
    }


def classify(image_urls: list[str], prompt: str | None = None) -> dict:
    """
    Classify document from image URLs.
    
    Args:
        image_urls: List of ShareFile pre-signed URLs (usually just first page)
        prompt: Optional custom prompt (uses default if not provided)
        
    Returns:
        Classification result with type, confidence, reasoning
    """
    load_model()
    
    # Use first page only for classification
    if not image_urls:
        raise ValueError("No image URLs provided")
    
    image = fetch_image(image_urls[0])
    
    # Build prompt
    if prompt is None:
        prompt = build_classification_prompt()
    
    # Prepare inputs for Qwen2.5-VL
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    # Process with model
    text_input = PROCESSOR.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = PROCESSOR(
        text=[text_input],
        images=[image],
        return_tensors="pt",
        padding=True
    ).to(MODEL.device)
    
    # Generate response
    import torch
    with torch.no_grad():
        output_ids = MODEL.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=True,
            pad_token_id=PROCESSOR.tokenizer.pad_token_id
        )
    
    # Decode response
    generated_ids = output_ids[:, inputs['input_ids'].shape[1]:]
    response_text = PROCESSOR.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    
    # Parse and validate
    result = parse_json_response(response_text)
    return validate_classification_result(result)


def handler(job: dict) -> dict:
    """
    RunPod serverless handler entry point.
    
    Expected input format:
    {
        "input": {
            "image_urls": ["https://..."],
            "prompt": "optional custom prompt"
        }
    }
    
    Returns:
    {
        "type": "W2",
        "confidence": 0.95,
        "reasoning": "..."
    }
    """
    start_time = time.time()
    
    try:
        input_data = job.get("input", {})
        
        image_urls = input_data.get("image_urls", [])
        prompt = input_data.get("prompt")
        
        if not image_urls:
            return {"error": "No image_urls provided"}
        
        result = classify(image_urls, prompt)
        
        # Add timing info
        result["latency_ms"] = int((time.time() - start_time) * 1000)
        
        logger.info(
            f"Classified as {result['type']} "
            f"(conf={result['confidence']:.2f}) "
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
    logger.info("Starting classification handler...")
    runpod.serverless.start({"handler": handler})
```

### 4.4 Requirements

```
# handlers/classify/requirements.txt
runpod>=1.6.0
torch>=2.1.0
transformers>=4.40.0
accelerate>=0.27.0
qwen-vl-utils>=0.0.2
requests>=2.31.0
pillow>=10.0.0
```

### 4.5 Local Testing

```python
# handlers/classify/test_handler.py
"""Local tests for classification handler."""

import json
from unittest.mock import patch, MagicMock

def test_parse_json_response():
    from handler import parse_json_response
    
    # Test markdown code block
    text = '```json\n{"type": "W2", "confidence": 0.95}\n```'
    result = parse_json_response(text)
    assert result["type"] == "W2"
    
    # Test raw JSON
    text = '{"type": "bank_statement", "confidence": 0.87}'
    result = parse_json_response(text)
    assert result["type"] == "bank_statement"
    
    # Test with extra text
    text = 'Here is the result:\n{"type": "invoice", "confidence": 0.9}\nDone.'
    result = parse_json_response(text)
    assert result["type"] == "invoice"


def test_validate_classification_result():
    from handler import validate_classification_result
    
    # Valid result
    result = validate_classification_result({
        "type": "W2",
        "confidence": 0.95,
        "reasoning": "test"
    })
    assert result["type"] == "W2"
    assert result["confidence"] == 0.95
    
    # Invalid type
    result = validate_classification_result({
        "type": "invalid_type",
        "confidence": 0.8
    })
    assert result["type"] == "other"
    
    # Out of range confidence
    result = validate_classification_result({
        "type": "W2",
        "confidence": 1.5
    })
    assert result["confidence"] == 1.0


if __name__ == "__main__":
    test_parse_json_response()
    test_validate_classification_result()
    print("All tests passed!")
```

---

## 5. Extraction Handler (dots.ocr)

This handler uses [dots.ocr](https://github.com/rednote-hilab/dots.ocr), a 1.7B parameter SOTA document parsing model that outperforms GPT-4o on OmniDocBench. It provides structured output with layout detection, OCR, and table extraction.

### 5.1 Directory Structure

```
handlers/extract/
├── Dockerfile
├── handler.py
├── transformers.py      # Post-processing for structured output
├── requirements.txt
├── start.sh             # Startup script for vLLM + handler
└── test_handler.py
```

### 5.2 Dockerfile

```dockerfile
# handlers/extract/Dockerfile
# 
# dots.ocr extraction handler using native vLLM 0.11.0+ support
# Model: rednote-hilab/dots.ocr (1.7B parameters, ~8GB VRAM)

FROM vllm/vllm-openai:v0.11.0

WORKDIR /app

# Install handler dependencies
RUN pip install --no-cache-dir \
    runpod>=1.6.0 \
    requests>=2.31.0 \
    pillow>=10.0.0 \
    httpx>=0.26.0

# Copy handler code
COPY handler.py .
COPY transformers.py .
COPY start.sh .

# Make startup script executable
RUN chmod +x start.sh

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/hf_cache
ENV MODEL_NAME=rednote-hilab/dots.ocr
ENV VLLM_PORT=8000

# Pre-download model during build (reduces cold start significantly)
# Uncomment for production builds:
# RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('rednote-hilab/dots.ocr')"

# Start vLLM server and handler
CMD ["./start.sh"]
```

### 5.3 Startup Script

```bash
#!/bin/bash
# handlers/extract/start.sh
#
# Starts vLLM server in background, waits for it to be ready,
# then starts the RunPod handler.

set -e

echo "Starting dots.ocr extraction handler..."

# Start vLLM server in background
echo "Launching vLLM server with dots.ocr..."
vllm serve rednote-hilab/dots.ocr \
    --host 0.0.0.0 \
    --port ${VLLM_PORT:-8000} \
    --trust-remote-code \
    --gpu-memory-utilization 0.95 \
    --max-model-len 24000 \
    --dtype float16 \
    &

VLLM_PID=$!

# Wait for vLLM to be ready
echo "Waiting for vLLM server to be ready..."
MAX_RETRIES=60
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s http://localhost:${VLLM_PORT:-8000}/health > /dev/null 2>&1; then
        echo "vLLM server is ready!"
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "Waiting for vLLM... (attempt $RETRY_COUNT/$MAX_RETRIES)"
    sleep 2
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "ERROR: vLLM server failed to start"
    exit 1
fi

# Start RunPod handler
echo "Starting RunPod handler..."
python -u handler.py

# If handler exits, kill vLLM
kill $VLLM_PID 2>/dev/null || true
```

### 5.4 Handler Implementation

```python
# handlers/extract/handler.py
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

import base64
import json
import logging
import os
import re
import time
from io import BytesIO
from typing import Any

import httpx
import requests
import runpod
from PIL import Image

from transformers import transform_to_document_schema

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# vLLM server configuration
VLLM_BASE_URL = f"http://localhost:{os.environ.get('VLLM_PORT', '8000')}/v1"
MODEL_NAME = "rednote-hilab/dots.ocr"

# dots.ocr prompt modes
PROMPT_MODES = {
    "layout_all": "prompt_layout_all_en",      # Full parsing: layout + OCR + tables
    "layout_only": "prompt_layout_only_en",    # Detection only (bounding boxes)
    "ocr": "prompt_ocr",                       # Text extraction (no headers/footers)
    "grounding": "prompt_grounding_ocr",       # Region-specific OCR with bbox
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
    """Convert PIL Image to base64 string."""
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
        prompt_mode: dots.ocr prompt mode
        max_tokens: Maximum tokens to generate
        
    Returns:
        Parsed response from dots.ocr
    """
    async with httpx.AsyncClient(timeout=120.0) as client:
        # dots.ocr uses OpenAI-compatible API via vLLM
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
                            "text": "Parse this document image. Extract all text with layout structure."
                        }
                    ]
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,  # Deterministic for OCR
        }
        
        response = await client.post(
            f"{VLLM_BASE_URL}/chat/completions",
            json=payload
        )
        response.raise_for_status()
        
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
        Parsed structure with layout_elements
    """
    # dots.ocr outputs structured text that we need to parse
    # The format varies but typically includes layout information
    
    # Try to extract JSON if present
    try:
        # Look for JSON block
        json_match = re.search(r'\{[\s\S]*\}', raw_output)
        if json_match:
            return json.loads(json_match.group())
    except json.JSONDecodeError:
        pass
    
    # Parse line-by-line format
    # dots.ocr typically outputs in a structured text format
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
        Dict with:
        - 'data': Extracted fields (transformed to document schema)
        - 'raw_ocr': Raw dots.ocr output with layout elements
        - 'confidence': Confidence scores
        - 'page_count': Number of pages processed
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
        except Exception as e:
            logger.error(f"Error processing page {i+1}: {e}")
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
    import asyncio
    
    start_time = time.time()
    
    try:
        input_data = job.get("input", {})
        
        image_urls = input_data.get("image_urls", [])
        doc_type = input_data.get("doc_type", "other")
        prompt_mode = input_data.get("prompt_mode", "layout_all")
        
        if not image_urls:
            return {"error": "No image_urls provided"}
        
        # Run async extraction
        result = asyncio.run(extract(image_urls, doc_type, prompt_mode))
        
        # Add metadata
        result["latency_ms"] = int((time.time() - start_time) * 1000)
        result["doc_type"] = doc_type
        result["model"] = "dots.ocr"
        
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
```

### 5.5 Document Schema Transformers

```python
# handlers/extract/transformers.py
"""
Transform raw dots.ocr output to document-specific schemas.

dots.ocr provides generic layout elements with bounding boxes and text.
These transformers convert that to structured fields matching our
document type schemas (W2, bank_statement, etc.).
"""

import re
from typing import Any


def extract_amounts(text: str) -> list[dict]:
    """Extract monetary amounts from text."""
    amounts = []
    # Match patterns like $1,234.56 or 1234.56
    pattern = r'\$?([\d,]+\.?\d*)'
    for match in re.finditer(pattern, text):
        try:
            value = float(match.group(1).replace(',', ''))
            amounts.append({
                "raw": match.group(0),
                "value": value,
                "position": match.start()
            })
        except ValueError:
            continue
    return amounts


def extract_dates(text: str) -> list[str]:
    """Extract dates from text."""
    dates = []
    patterns = [
        r'\d{1,2}/\d{1,2}/\d{2,4}',
        r'\d{1,2}-\d{1,2}-\d{2,4}',
        r'\d{4}-\d{2}-\d{2}',
        r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}',
    ]
    for pattern in patterns:
        dates.extend(re.findall(pattern, text, re.IGNORECASE))
    return dates


def transform_bank_statement(elements: list, raw_text: str) -> dict:
    """
    Transform dots.ocr output to bank statement schema.
    
    Returns structured data matching the bank_statement schema.
    """
    result = {
        "header": {
            "bank_name": None,
            "account_number": None,
            "account_type": None,
            "statement_period": None,
            "beginning_balance": None,
            "ending_balance": None
        },
        "transactions": [],
        "summary": {
            "total_credits": None,
            "total_debits": None,
            "net_change": None
        }
    }
    
    # Extract common bank names
    bank_patterns = [
        r'(Chase|Bank of America|Wells Fargo|Citi|Capital One|PNC|TD Bank|US Bank)',
        r'([A-Z][a-z]+ (?:Bank|Credit Union|FCU))',
    ]
    for pattern in bank_patterns:
        match = re.search(pattern, raw_text, re.IGNORECASE)
        if match:
            result["header"]["bank_name"] = match.group(1)
            break
    
    # Extract account number (masked)
    acct_match = re.search(r'(?:Account|Acct)[:\s#]*(\*{2,}\d{4}|\d{4})', raw_text, re.IGNORECASE)
    if acct_match:
        result["header"]["account_number"] = f"****{acct_match.group(1)[-4:]}"
    
    # Extract dates for statement period
    dates = extract_dates(raw_text)
    if len(dates) >= 2:
        result["header"]["statement_period"] = f"{dates[0]} - {dates[1]}"
    
    # Look for beginning/ending balance patterns
    for element in elements:
        text = element.get("text", "").lower()
        if "beginning" in text or "opening" in text:
            amts = extract_amounts(element.get("text", ""))
            if amts:
                result["header"]["beginning_balance"] = amts[-1]["value"]
        elif "ending" in text or "closing" in text:
            amts = extract_amounts(element.get("text", ""))
            if amts:
                result["header"]["ending_balance"] = amts[-1]["value"]
    
    # Extract transactions (look for table-like patterns)
    transaction_pattern = r'(\d{1,2}/\d{1,2})\s+(.+?)\s+(-?\$?[\d,]+\.?\d*)'
    for match in re.finditer(transaction_pattern, raw_text):
        date, desc, amount = match.groups()
        try:
            amt_value = float(amount.replace('

## 6. Gateway Server

### 6.1 Directory Structure

```
gateway/
├── Dockerfile
├── requirements.txt
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── classify.py
│   │   ├── extract.py
│   │   └── health.py
│   ├── services/
│   │   ├── __init__.py
│   │   └── runpod_client.py
│   └── middleware/
│       ├── __init__.py
│       ├── auth.py
│       └── logging.py
└── tests/
    └── test_routes.py
```

### 6.2 Configuration

```python
# gateway/app/config.py
"""Gateway configuration from environment variables."""

import os
from dataclasses import dataclass
from functools import lru_cache


@dataclass
class Settings:
    """Application settings."""
    
    # API Authentication
    api_key: str
    
    # RunPod Configuration
    runpod_api_key: str
    runpod_classify_endpoint: str
    runpod_extract_endpoint: str
    runpod_timeout: int = 120
    runpod_max_retries: int = 3
    
    # Rate Limiting
    rate_limit_requests: int = 100  # per minute
    rate_limit_window: int = 60     # seconds
    
    # Logging
    log_level: str = "INFO"
    
    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from environment variables."""
        return cls(
            api_key=os.environ["GATEWAY_API_KEY"],
            runpod_api_key=os.environ["RUNPOD_API_KEY"],
            runpod_classify_endpoint=os.environ["RUNPOD_CLASSIFY_ENDPOINT"],
            runpod_extract_endpoint=os.environ["RUNPOD_EXTRACT_ENDPOINT"],
            runpod_timeout=int(os.environ.get("RUNPOD_TIMEOUT", "120")),
            runpod_max_retries=int(os.environ.get("RUNPOD_MAX_RETRIES", "3")),
            rate_limit_requests=int(os.environ.get("RATE_LIMIT_REQUESTS", "100")),
            rate_limit_window=int(os.environ.get("RATE_LIMIT_WINDOW", "60")),
            log_level=os.environ.get("LOG_LEVEL", "INFO")
        )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings.from_env()
```

### 6.3 RunPod Client Service

```python
# gateway/app/services/runpod_client.py
"""RunPod Serverless API client."""

import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)

RUNPOD_API_BASE = "https://api.runpod.ai/v2"


@dataclass
class RunPodResponse:
    """Response from RunPod endpoint."""
    success: bool
    data: dict | None
    error: str | None
    latency_ms: int
    job_id: str | None = None


class RunPodClient:
    """
    Async client for RunPod Serverless endpoints.
    
    Handles authentication, retries, and error handling.
    """
    
    def __init__(
        self,
        api_key: str,
        classify_endpoint: str,
        extract_endpoint: str,
        timeout: int = 120,
        max_retries: int = 3
    ):
        self.api_key = api_key
        self.classify_endpoint = classify_endpoint
        self.extract_endpoint = extract_endpoint
        self.timeout = timeout
        self.max_retries = max_retries
        
        self._client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            timeout=httpx.Timeout(timeout, connect=10)
        )
    
    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()
    
    async def classify(
        self,
        image_urls: list[str],
        prompt: str | None = None
    ) -> RunPodResponse:
        """
        Call classification endpoint.
        
        Args:
            image_urls: ShareFile pre-signed URLs
            prompt: Optional custom prompt
            
        Returns:
            RunPodResponse with classification result
        """
        return await self._call(
            endpoint_id=self.classify_endpoint,
            payload={
                "image_urls": image_urls,
                "prompt": prompt
            }
        )
    
    async def extract(
        self,
        image_urls: list[str],
        doc_type: str,
        prompt: str | None = None
    ) -> RunPodResponse:
        """
        Call extraction endpoint.
        
        Args:
            image_urls: ShareFile pre-signed URLs
            doc_type: Document type for prompt selection
            prompt: Optional custom prompt
            
        Returns:
            RunPodResponse with extraction result
        """
        return await self._call(
            endpoint_id=self.extract_endpoint,
            payload={
                "image_urls": image_urls,
                "doc_type": doc_type,
                "prompt": prompt
            }
        )
    
    async def health_check(self) -> dict:
        """Check health of both endpoints."""
        results = {}
        
        for name, endpoint_id in [
            ("classify", self.classify_endpoint),
            ("extract", self.extract_endpoint)
        ]:
            try:
                url = f"{RUNPOD_API_BASE}/{endpoint_id}/health"
                response = await self._client.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    results[name] = {
                        "status": "healthy",
                        "workers": data.get("workers", {})
                    }
                elif response.status_code == 204:
                    results[name] = {
                        "status": "idle",
                        "workers": {"idle": 0, "running": 0}
                    }
                else:
                    results[name] = {
                        "status": "unhealthy",
                        "error": f"HTTP {response.status_code}"
                    }
            except Exception as e:
                results[name] = {
                    "status": "unreachable",
                    "error": str(e)
                }
        
        return results
    
    async def _call(
        self,
        endpoint_id: str,
        payload: dict
    ) -> RunPodResponse:
        """
        Make a synchronous call to a RunPod endpoint.
        
        Uses /runsync for immediate response.
        """
        url = f"{RUNPOD_API_BASE}/{endpoint_id}/runsync"
        
        last_error = None
        base_delay = 5.0
        
        for attempt in range(self.max_retries):
            start_time = time.time()
            
            try:
                response = await self._client.post(
                    url,
                    json={"input": payload}
                )
                
                latency_ms = int((time.time() - start_time) * 1000)
                data = response.json()
                
                status = data.get("status")
                job_id = data.get("id")
                
                # Success
                if status == "COMPLETED":
                    return RunPodResponse(
                        success=True,
                        data=data.get("output", {}),
                        error=None,
                        latency_ms=data.get("executionTime", latency_ms),
                        job_id=job_id
                    )
                
                # Failed
                if status == "FAILED":
                    error_msg = data.get("error", "Unknown error")
                    return RunPodResponse(
                        success=False,
                        data=None,
                        error=error_msg,
                        latency_ms=latency_ms,
                        job_id=job_id
                    )
                
                # Timeout/in-progress
                if status in ("IN_QUEUE", "IN_PROGRESS"):
                    last_error = f"Job timed out in status: {status}"
                    logger.warning(f"RunPod job {job_id} timed out: {status}")
                
                # HTTP errors
                if response.status_code == 401:
                    return RunPodResponse(
                        success=False,
                        data=None,
                        error="Authentication failed",
                        latency_ms=latency_ms
                    )
                
                if response.status_code == 400:
                    return RunPodResponse(
                        success=False,
                        data=None,
                        error=f"Bad request: {data.get('error', 'Unknown')}",
                        latency_ms=latency_ms
                    )
                
                # Retry on server errors
                if response.status_code in (429, 500, 502, 503, 504):
                    last_error = f"HTTP {response.status_code}"
                    logger.warning(
                        f"RunPod error {response.status_code}, "
                        f"attempt {attempt + 1}/{self.max_retries}"
                    )
                    
            except httpx.TimeoutException:
                last_error = "Request timeout"
                logger.warning(
                    f"RunPod timeout, attempt {attempt + 1}/{self.max_retries}"
                )
                
            except httpx.RequestError as e:
                last_error = str(e)
                logger.warning(
                    f"RunPod request error: {e}, "
                    f"attempt {attempt + 1}/{self.max_retries}"
                )
            
            # Exponential backoff
            if attempt < self.max_retries - 1:
                delay = min(base_delay * (2 ** attempt), 60)
                logger.info(f"Retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)
        
        # All retries exhausted
        return RunPodResponse(
            success=False,
            data=None,
            error=f"Max retries exceeded: {last_error}",
            latency_ms=0
        )


import asyncio  # Import at top in real code
```

### 6.4 Authentication Middleware

```python
# gateway/app/middleware/auth.py
"""API key authentication middleware."""

import logging
import secrets
from typing import Callable

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

logger = logging.getLogger(__name__)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def create_auth_dependency(valid_api_key: str) -> Callable:
    """
    Create an API key validation dependency.
    
    Args:
        valid_api_key: The expected API key
        
    Returns:
        FastAPI dependency function
    """
    async def verify_api_key(
        api_key: str = Security(api_key_header)
    ) -> str:
        if api_key is None:
            logger.warning("Missing API key in request")
            raise HTTPException(
                status_code=401,
                detail="Missing API key"
            )
        
        # Constant-time comparison to prevent timing attacks
        if not secrets.compare_digest(api_key, valid_api_key):
            logger.warning("Invalid API key attempt")
            raise HTTPException(
                status_code=401,
                detail="Invalid API key"
            )
        
        return api_key
    
    return verify_api_key
```

### 6.5 Logging Middleware

```python
# gateway/app/middleware/logging.py
"""Request logging middleware."""

import logging
import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all requests with timing and request ID."""
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        # Generate request ID
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        
        # Log request
        logger.info(
            f"[{request_id}] {request.method} {request.url.path} "
            f"from {request.client.host if request.client else 'unknown'}"
        )
        
        # Process request
        start_time = time.time()
        
        try:
            response = await call_next(request)
            
            # Log response
            duration_ms = int((time.time() - start_time) * 1000)
            logger.info(
                f"[{request_id}] {response.status_code} "
                f"in {duration_ms}ms"
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.error(
                f"[{request_id}] Error after {duration_ms}ms: {e}"
            )
            raise
```

### 6.6 Route: Classification

```python
# gateway/app/routes/classify.py
"""Classification endpoint."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..config import Settings, get_settings
from ..middleware.auth import create_auth_dependency
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
    settings: Annotated[Settings, Depends(get_settings)],
    _api_key: Annotated[str, Depends(create_auth_dependency)]
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


# Need to inject the actual API key at startup
def configure_router(api_key: str):
    """Configure the router with the API key for auth."""
    router.dependencies = [Depends(create_auth_dependency(api_key))]
```

### 6.7 Route: Extraction

```python
# gateway/app/routes/extract.py
"""Extraction endpoint."""

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..config import Settings, get_settings
from ..middleware.auth import create_auth_dependency
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
    confidence: dict[str, Any] = Field(..., description="Per-field confidence scores")
    doc_type: str = Field(..., description="Document type used for extraction")
    page_count: int = Field(..., description="Number of pages processed")
    latency_ms: int = Field(..., description="Inference latency in milliseconds")


@router.post("/extract", response_model=ExtractResponse)
async def extract_document(
    request: ExtractRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    _api_key: Annotated[str, Depends(create_auth_dependency)]
) -> ExtractResponse:
    """
    Extract structured data from document images.
    
    Returns extracted fields with per-field confidence scores.
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
        # Call extraction endpoint
        response = await client.extract(
            image_urls=request.image_urls,
            doc_type=request.doc_type,
            prompt=request.prompt
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
            doc_type=data.get("doc_type", request.doc_type),
            page_count=data.get("page_count", len(request.image_urls)),
            latency_ms=response.latency_ms
        )
        
    finally:
        await client.close()
```

### 6.8 Route: Health

```python
# gateway/app/routes/health.py
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
```

### 6.9 Main Application

```python
# gateway/app/main.py
"""Gateway server main application."""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .middleware.logging import RequestLoggingMiddleware
from .middleware.auth import create_auth_dependency
from .routes import classify, extract, health

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title="Document Processing Inference Gateway",
        description="Gateway for document classification and extraction services",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Add CORS middleware (configure for your needs)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Restrict in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add request logging
    app.add_middleware(RequestLoggingMiddleware)
    
    # Create auth dependency with configured API key
    auth_dependency = create_auth_dependency(settings.api_key)
    
    # Include routers
    # Health routes (no auth)
    app.include_router(health.router)
    
    # Inference routes (with auth)
    classify.router.dependencies = [auth_dependency]
    extract.router.dependencies = [auth_dependency]
    app.include_router(classify.router)
    app.include_router(extract.router)
    
    @app.on_event("startup")
    async def startup():
        logger.info("Gateway server starting...")
        logger.info(f"Classify endpoint: {settings.runpod_classify_endpoint}")
        logger.info(f"Extract endpoint: {settings.runpod_extract_endpoint}")
    
    @app.on_event("shutdown")
    async def shutdown():
        logger.info("Gateway server shutting down...")
    
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 6.10 Dockerfile

```dockerfile
# gateway/Dockerfile

FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Create non-root user
RUN useradd -m -u 1000 appuser
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/live || exit 1

# Run with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 6.11 Requirements

```
# gateway/requirements.txt
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
httpx>=0.26.0
pydantic>=2.5.0
python-multipart>=0.0.6
```

---

## 7. Deployment

### 7.1 Build Handler Images

```bash
# Navigate to handlers directory
cd inference/handlers

# Build classification handler
cd classify
docker build -t your-registry/docproc-classify:v1 .
docker push your-registry/docproc-classify:v1

# Build extraction handler
cd ../extract
docker build -t your-registry/docproc-extract:v1 .
docker push your-registry/docproc-extract:v1
```

### 7.2 Deploy to RunPod

1. **Go to RunPod Console**: https://www.runpod.io/console/serverless

2. **Create Classification Endpoint**:
   - Click "New Endpoint"
   - Name: `docproc-classify`
   - Docker Image: `your-registry/docproc-classify:v1`
   - GPU Type: L4 or A10 (16GB+ VRAM)
   - Max Workers: 3
   - Idle Timeout: 60s
   - Note the **Endpoint ID**

3. **Create Extraction Endpoint**:
   - Click "New Endpoint"
   - Name: `docproc-extract`
   - Docker Image: `your-registry/docproc-extract:v1`
   - GPU Type: L4 or A10 (16GB+ VRAM)
   - Max Workers: 3
   - Idle Timeout: 60s
   - Note the **Endpoint ID**

4. **Get API Key**: Settings → API Keys

### 7.3 Deploy Gateway

**Option A: Docker Compose (recommended for pilot)**

```yaml
# inference/docker-compose.yml
version: '3.8'

services:
  gateway:
    build: ./gateway
    ports:
      - "8000:8000"
    environment:
      GATEWAY_API_KEY: ${GATEWAY_API_KEY}
      RUNPOD_API_KEY: ${RUNPOD_API_KEY}
      RUNPOD_CLASSIFY_ENDPOINT: ${RUNPOD_CLASSIFY_ENDPOINT}
      RUNPOD_EXTRACT_ENDPOINT: ${RUNPOD_EXTRACT_ENDPOINT}
      RUNPOD_TIMEOUT: "120"
      LOG_LEVEL: "INFO"
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/live"]
      interval: 30s
      timeout: 10s
      retries: 3
```

**Option B: Cloud Run / Railway / Fly.io**

The gateway is stateless and can deploy to any container platform.

### 7.4 Environment Variables

```bash
# inference/.env.example

# Gateway Authentication
GATEWAY_API_KEY=your-secure-api-key-here

# RunPod Configuration
RUNPOD_API_KEY=your-runpod-api-key
RUNPOD_CLASSIFY_ENDPOINT=abc123xyz
RUNPOD_EXTRACT_ENDPOINT=def456uvw
RUNPOD_TIMEOUT=120

# Optional
LOG_LEVEL=INFO
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60
```

### 7.5 Update Document Pipeline

Update your main document processing config to use the gateway:

```yaml
# config/config.yaml
inference:
  gateway_url: "http://localhost:8000"  # Or your deployed URL
  api_key: ${GATEWAY_API_KEY}
  timeout_seconds: 130  # Slightly more than RunPod timeout
  retry:
    max_attempts: 2  # Gateway handles internal retries
    base_delay_seconds: 5
```

---

## 8. Testing

### 8.1 Test Gateway Locally

```bash
# Start gateway
cd inference/gateway
uvicorn app.main:app --reload

# Test health
curl http://localhost:8000/health

# Test classification (with test image URL)
curl -X POST http://localhost:8000/v1/classify \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "image_urls": ["https://example.com/test-w2.png"]
  }'

# Test extraction
curl -X POST http://localhost:8000/v1/extract \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "image_urls": ["https://example.com/test-w2.png"],
    "doc_type": "W2"
  }'
```

### 8.2 Test RunPod Endpoints Directly

```bash
# Test classification endpoint
curl -X POST "https://api.runpod.ai/v2/${ENDPOINT_ID}/runsync" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "image_urls": ["https://example.com/test.png"]
    }
  }'
```

### 8.3 Integration Test Script

```python
# inference/tests/test_integration.py
"""Integration tests for inference pipeline."""

import os
import httpx
import pytest

GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://localhost:8000")
API_KEY = os.environ.get("GATEWAY_API_KEY", "test-key")

# Test image URLs (replace with real ShareFile URLs)
TEST_W2_URL = "https://your-sharefile.com/test-w2.png"
TEST_BANK_URL = "https://your-sharefile.com/test-bank.png"


@pytest.fixture
def client():
    return httpx.Client(
        base_url=GATEWAY_URL,
        headers={"X-API-Key": API_KEY},
        timeout=180
    )


def test_health(client):
    """Test health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["gateway"] == "healthy"


def test_classify_w2(client):
    """Test W2 classification."""
    response = client.post("/v1/classify", json={
        "image_urls": [TEST_W2_URL]
    })
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "W2"
    assert data["confidence"] >= 0.8


def test_extract_w2(client):
    """Test W2 extraction."""
    response = client.post("/v1/extract", json={
        "image_urls": [TEST_W2_URL],
        "doc_type": "W2"
    })
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert "boxes" in data["data"]
    assert "box1_wages" in data["data"]["boxes"]


def test_extract_bank_statement(client):
    """Test bank statement extraction."""
    response = client.post("/v1/extract", json={
        "image_urls": [TEST_BANK_URL],
        "doc_type": "bank_statement"
    })
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert "transactions" in data["data"]


def test_invalid_api_key(client):
    """Test authentication rejection."""
    bad_client = httpx.Client(
        base_url=GATEWAY_URL,
        headers={"X-API-Key": "wrong-key"}
    )
    response = bad_client.post("/v1/classify", json={
        "image_urls": ["https://example.com/test.png"]
    })
    assert response.status_code == 401


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

---

## 9. Cost Optimization

### 9.1 Expected Costs

| Component | Estimated Cost |
|-----------|----------------|
| RunPod L4 inference | ~$0.20/hr per worker |
| Gateway (Cloud Run) | ~$5-10/month |
| ShareFile | Existing subscription |

**Per-document cost estimate:**
- Classification: ~3-8s = $0.0002-0.0004
- Extraction: ~8-20s = $0.0004-0.0011
- **Total per doc: ~$0.001-0.002**

For 2,500 pages: **~$2.50-5.00**

### 9.2 Optimization Strategies

1. **Batch Classification**: For bulk imports, batch classify first pages only
2. **Idle Timeout**: Set to 30-60s to balance cold starts vs idle cost
3. **Right-size Workers**: Start with max_workers=2, increase if queue builds
4. **FlashBoot**: Enable for faster cold starts (trades startup time for network load)

### 9.3 Monitoring

Track these metrics:

```python
# Add to gateway for monitoring
from prometheus_client import Counter, Histogram

REQUESTS = Counter(
    'inference_requests_total',
    'Total inference requests',
    ['endpoint', 'status']
)

LATENCY = Histogram(
    'inference_latency_seconds',
    'Inference latency',
    ['endpoint'],
    buckets=[1, 2, 5, 10, 20, 30, 60, 120]
)
```

---

## Appendix A: Troubleshooting

### Cold Start Too Slow

**Symptoms:** First request takes 60-120s

**Solutions:**
1. Enable FlashBoot in RunPod endpoint settings
2. Keep min_workers=1 during business hours
3. Pre-warm with scheduled health checks

### Out of Memory

**Symptoms:** Worker crashes, "CUDA out of memory"

**Solutions:**
1. Use smaller batch size (process 1-2 pages at a time)
2. Upgrade to larger GPU (A40 instead of L4)
3. Enable model quantization (AWQ/GPTQ)

### Image Fetch Failures

**Symptoms:** "Failed to fetch image" errors

**Solutions:**
1. Verify ShareFile URL expiry (increase url_expiry_minutes)
2. Check network connectivity from RunPod
3. Add retry logic for transient failures

### JSON Parse Errors

**Symptoms:** "Could not parse JSON" in logs

**Solutions:**
1. Review prompt to emphasize JSON-only output
2. Add more robust parsing patterns
3. Lower temperature further (0.05)

---

## Appendix B: Quick Reference

### API Endpoints

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| GET | `/health` | Full health check | No |
| GET | `/ready` | Readiness probe | No |
| GET | `/live` | Liveness probe | No |
| POST | `/v1/classify` | Classify document | Yes |
| POST | `/v1/extract` | Extract data | Yes |

### Request Headers

```
X-API-Key: your-gateway-api-key
Content-Type: application/json
```

### Response Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 401 | Invalid or missing API key |
| 400 | Invalid request body |
| 502 | Inference endpoint failure |
| 504 | Inference timeout |

---

*End of Implementation Guide*
, '').replace(',', ''))
            result["transactions"].append({
                "date": date,
                "description": desc.strip(),
                "amount": amt_value,
                "type": "credit" if amt_value > 0 else "debit"
            })
        except ValueError:
            continue
    
    return result


def transform_w2(elements: list, raw_text: str) -> dict:
    """
    Transform dots.ocr output to W2 schema.
    
    W2 forms have standardized box positions, so we look for
    specific box labels and nearby amounts.
    """
    result = {
        "tax_year": None,
        "employer": {
            "ein": None,
            "name": None,
            "address": None
        },
        "employee": {
            "ssn": None,
            "name": None,
            "address": None
        },
        "boxes": {
            "box1_wages": None,
            "box2_federal_withheld": None,
            "box3_ss_wages": None,
            "box4_ss_withheld": None,
            "box5_medicare_wages": None,
            "box6_medicare_withheld": None,
        }
    }
    
    # Extract tax year
    year_match = re.search(r'20\d{2}', raw_text)
    if year_match:
        result["tax_year"] = int(year_match.group())
    
    # Extract EIN (XX-XXXXXXX format)
    ein_match = re.search(r'(\d{2}-\d{7})', raw_text)
    if ein_match:
        result["employer"]["ein"] = ein_match.group(1)
    
    # Extract SSN (masked format)
    ssn_match = re.search(r'(\d{3}-\d{2}-\d{4})', raw_text)
    if ssn_match:
        ssn = ssn_match.group(1)
        result["employee"]["ssn"] = f"***-**-{ssn[-4:]}"
    
    # Extract box values by looking for box labels
    box_patterns = {
        "box1_wages": r'(?:Box\s*)?1[.\s]+Wages.*?(\$?[\d,]+\.?\d*)',
        "box2_federal_withheld": r'(?:Box\s*)?2[.\s]+Federal.*?(\$?[\d,]+\.?\d*)',
        "box3_ss_wages": r'(?:Box\s*)?3[.\s]+Social\s+security\s+wages.*?(\$?[\d,]+\.?\d*)',
        "box4_ss_withheld": r'(?:Box\s*)?4[.\s]+Social\s+security\s+tax.*?(\$?[\d,]+\.?\d*)',
        "box5_medicare_wages": r'(?:Box\s*)?5[.\s]+Medicare\s+wages.*?(\$?[\d,]+\.?\d*)',
        "box6_medicare_withheld": r'(?:Box\s*)?6[.\s]+Medicare\s+tax.*?(\$?[\d,]+\.?\d*)',
    }
    
    for box_name, pattern in box_patterns.items():
        match = re.search(pattern, raw_text, re.IGNORECASE)
        if match:
            try:
                value = float(match.group(1).replace('

## 6. Gateway Server

### 6.1 Directory Structure

```
gateway/
├── Dockerfile
├── requirements.txt
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── classify.py
│   │   ├── extract.py
│   │   └── health.py
│   ├── services/
│   │   ├── __init__.py
│   │   └── runpod_client.py
│   └── middleware/
│       ├── __init__.py
│       ├── auth.py
│       └── logging.py
└── tests/
    └── test_routes.py
```

### 6.2 Configuration

```python
# gateway/app/config.py
"""Gateway configuration from environment variables."""

import os
from dataclasses import dataclass
from functools import lru_cache


@dataclass
class Settings:
    """Application settings."""
    
    # API Authentication
    api_key: str
    
    # RunPod Configuration
    runpod_api_key: str
    runpod_classify_endpoint: str
    runpod_extract_endpoint: str
    runpod_timeout: int = 120
    runpod_max_retries: int = 3
    
    # Rate Limiting
    rate_limit_requests: int = 100  # per minute
    rate_limit_window: int = 60     # seconds
    
    # Logging
    log_level: str = "INFO"
    
    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from environment variables."""
        return cls(
            api_key=os.environ["GATEWAY_API_KEY"],
            runpod_api_key=os.environ["RUNPOD_API_KEY"],
            runpod_classify_endpoint=os.environ["RUNPOD_CLASSIFY_ENDPOINT"],
            runpod_extract_endpoint=os.environ["RUNPOD_EXTRACT_ENDPOINT"],
            runpod_timeout=int(os.environ.get("RUNPOD_TIMEOUT", "120")),
            runpod_max_retries=int(os.environ.get("RUNPOD_MAX_RETRIES", "3")),
            rate_limit_requests=int(os.environ.get("RATE_LIMIT_REQUESTS", "100")),
            rate_limit_window=int(os.environ.get("RATE_LIMIT_WINDOW", "60")),
            log_level=os.environ.get("LOG_LEVEL", "INFO")
        )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings.from_env()
```

### 6.3 RunPod Client Service

```python
# gateway/app/services/runpod_client.py
"""RunPod Serverless API client."""

import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)

RUNPOD_API_BASE = "https://api.runpod.ai/v2"


@dataclass
class RunPodResponse:
    """Response from RunPod endpoint."""
    success: bool
    data: dict | None
    error: str | None
    latency_ms: int
    job_id: str | None = None


class RunPodClient:
    """
    Async client for RunPod Serverless endpoints.
    
    Handles authentication, retries, and error handling.
    """
    
    def __init__(
        self,
        api_key: str,
        classify_endpoint: str,
        extract_endpoint: str,
        timeout: int = 120,
        max_retries: int = 3
    ):
        self.api_key = api_key
        self.classify_endpoint = classify_endpoint
        self.extract_endpoint = extract_endpoint
        self.timeout = timeout
        self.max_retries = max_retries
        
        self._client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            timeout=httpx.Timeout(timeout, connect=10)
        )
    
    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()
    
    async def classify(
        self,
        image_urls: list[str],
        prompt: str | None = None
    ) -> RunPodResponse:
        """
        Call classification endpoint.
        
        Args:
            image_urls: ShareFile pre-signed URLs
            prompt: Optional custom prompt
            
        Returns:
            RunPodResponse with classification result
        """
        return await self._call(
            endpoint_id=self.classify_endpoint,
            payload={
                "image_urls": image_urls,
                "prompt": prompt
            }
        )
    
    async def extract(
        self,
        image_urls: list[str],
        doc_type: str,
        prompt: str | None = None
    ) -> RunPodResponse:
        """
        Call extraction endpoint.
        
        Args:
            image_urls: ShareFile pre-signed URLs
            doc_type: Document type for prompt selection
            prompt: Optional custom prompt
            
        Returns:
            RunPodResponse with extraction result
        """
        return await self._call(
            endpoint_id=self.extract_endpoint,
            payload={
                "image_urls": image_urls,
                "doc_type": doc_type,
                "prompt": prompt
            }
        )
    
    async def health_check(self) -> dict:
        """Check health of both endpoints."""
        results = {}
        
        for name, endpoint_id in [
            ("classify", self.classify_endpoint),
            ("extract", self.extract_endpoint)
        ]:
            try:
                url = f"{RUNPOD_API_BASE}/{endpoint_id}/health"
                response = await self._client.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    results[name] = {
                        "status": "healthy",
                        "workers": data.get("workers", {})
                    }
                elif response.status_code == 204:
                    results[name] = {
                        "status": "idle",
                        "workers": {"idle": 0, "running": 0}
                    }
                else:
                    results[name] = {
                        "status": "unhealthy",
                        "error": f"HTTP {response.status_code}"
                    }
            except Exception as e:
                results[name] = {
                    "status": "unreachable",
                    "error": str(e)
                }
        
        return results
    
    async def _call(
        self,
        endpoint_id: str,
        payload: dict
    ) -> RunPodResponse:
        """
        Make a synchronous call to a RunPod endpoint.
        
        Uses /runsync for immediate response.
        """
        url = f"{RUNPOD_API_BASE}/{endpoint_id}/runsync"
        
        last_error = None
        base_delay = 5.0
        
        for attempt in range(self.max_retries):
            start_time = time.time()
            
            try:
                response = await self._client.post(
                    url,
                    json={"input": payload}
                )
                
                latency_ms = int((time.time() - start_time) * 1000)
                data = response.json()
                
                status = data.get("status")
                job_id = data.get("id")
                
                # Success
                if status == "COMPLETED":
                    return RunPodResponse(
                        success=True,
                        data=data.get("output", {}),
                        error=None,
                        latency_ms=data.get("executionTime", latency_ms),
                        job_id=job_id
                    )
                
                # Failed
                if status == "FAILED":
                    error_msg = data.get("error", "Unknown error")
                    return RunPodResponse(
                        success=False,
                        data=None,
                        error=error_msg,
                        latency_ms=latency_ms,
                        job_id=job_id
                    )
                
                # Timeout/in-progress
                if status in ("IN_QUEUE", "IN_PROGRESS"):
                    last_error = f"Job timed out in status: {status}"
                    logger.warning(f"RunPod job {job_id} timed out: {status}")
                
                # HTTP errors
                if response.status_code == 401:
                    return RunPodResponse(
                        success=False,
                        data=None,
                        error="Authentication failed",
                        latency_ms=latency_ms
                    )
                
                if response.status_code == 400:
                    return RunPodResponse(
                        success=False,
                        data=None,
                        error=f"Bad request: {data.get('error', 'Unknown')}",
                        latency_ms=latency_ms
                    )
                
                # Retry on server errors
                if response.status_code in (429, 500, 502, 503, 504):
                    last_error = f"HTTP {response.status_code}"
                    logger.warning(
                        f"RunPod error {response.status_code}, "
                        f"attempt {attempt + 1}/{self.max_retries}"
                    )
                    
            except httpx.TimeoutException:
                last_error = "Request timeout"
                logger.warning(
                    f"RunPod timeout, attempt {attempt + 1}/{self.max_retries}"
                )
                
            except httpx.RequestError as e:
                last_error = str(e)
                logger.warning(
                    f"RunPod request error: {e}, "
                    f"attempt {attempt + 1}/{self.max_retries}"
                )
            
            # Exponential backoff
            if attempt < self.max_retries - 1:
                delay = min(base_delay * (2 ** attempt), 60)
                logger.info(f"Retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)
        
        # All retries exhausted
        return RunPodResponse(
            success=False,
            data=None,
            error=f"Max retries exceeded: {last_error}",
            latency_ms=0
        )


import asyncio  # Import at top in real code
```

### 6.4 Authentication Middleware

```python
# gateway/app/middleware/auth.py
"""API key authentication middleware."""

import logging
import secrets
from typing import Callable

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

logger = logging.getLogger(__name__)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def create_auth_dependency(valid_api_key: str) -> Callable:
    """
    Create an API key validation dependency.
    
    Args:
        valid_api_key: The expected API key
        
    Returns:
        FastAPI dependency function
    """
    async def verify_api_key(
        api_key: str = Security(api_key_header)
    ) -> str:
        if api_key is None:
            logger.warning("Missing API key in request")
            raise HTTPException(
                status_code=401,
                detail="Missing API key"
            )
        
        # Constant-time comparison to prevent timing attacks
        if not secrets.compare_digest(api_key, valid_api_key):
            logger.warning("Invalid API key attempt")
            raise HTTPException(
                status_code=401,
                detail="Invalid API key"
            )
        
        return api_key
    
    return verify_api_key
```

### 6.5 Logging Middleware

```python
# gateway/app/middleware/logging.py
"""Request logging middleware."""

import logging
import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all requests with timing and request ID."""
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        # Generate request ID
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        
        # Log request
        logger.info(
            f"[{request_id}] {request.method} {request.url.path} "
            f"from {request.client.host if request.client else 'unknown'}"
        )
        
        # Process request
        start_time = time.time()
        
        try:
            response = await call_next(request)
            
            # Log response
            duration_ms = int((time.time() - start_time) * 1000)
            logger.info(
                f"[{request_id}] {response.status_code} "
                f"in {duration_ms}ms"
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.error(
                f"[{request_id}] Error after {duration_ms}ms: {e}"
            )
            raise
```

### 6.6 Route: Classification

```python
# gateway/app/routes/classify.py
"""Classification endpoint."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..config import Settings, get_settings
from ..middleware.auth import create_auth_dependency
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
    settings: Annotated[Settings, Depends(get_settings)],
    _api_key: Annotated[str, Depends(create_auth_dependency)]
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


# Need to inject the actual API key at startup
def configure_router(api_key: str):
    """Configure the router with the API key for auth."""
    router.dependencies = [Depends(create_auth_dependency(api_key))]
```

### 6.7 Route: Extraction

```python
# gateway/app/routes/extract.py
"""Extraction endpoint."""

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..config import Settings, get_settings
from ..middleware.auth import create_auth_dependency
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
    confidence: dict[str, Any] = Field(..., description="Per-field confidence scores")
    doc_type: str = Field(..., description="Document type used for extraction")
    page_count: int = Field(..., description="Number of pages processed")
    latency_ms: int = Field(..., description="Inference latency in milliseconds")


@router.post("/extract", response_model=ExtractResponse)
async def extract_document(
    request: ExtractRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    _api_key: Annotated[str, Depends(create_auth_dependency)]
) -> ExtractResponse:
    """
    Extract structured data from document images.
    
    Returns extracted fields with per-field confidence scores.
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
        # Call extraction endpoint
        response = await client.extract(
            image_urls=request.image_urls,
            doc_type=request.doc_type,
            prompt=request.prompt
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
            doc_type=data.get("doc_type", request.doc_type),
            page_count=data.get("page_count", len(request.image_urls)),
            latency_ms=response.latency_ms
        )
        
    finally:
        await client.close()
```

### 6.8 Route: Health

```python
# gateway/app/routes/health.py
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
```

### 6.9 Main Application

```python
# gateway/app/main.py
"""Gateway server main application."""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .middleware.logging import RequestLoggingMiddleware
from .middleware.auth import create_auth_dependency
from .routes import classify, extract, health

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title="Document Processing Inference Gateway",
        description="Gateway for document classification and extraction services",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Add CORS middleware (configure for your needs)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Restrict in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add request logging
    app.add_middleware(RequestLoggingMiddleware)
    
    # Create auth dependency with configured API key
    auth_dependency = create_auth_dependency(settings.api_key)
    
    # Include routers
    # Health routes (no auth)
    app.include_router(health.router)
    
    # Inference routes (with auth)
    classify.router.dependencies = [auth_dependency]
    extract.router.dependencies = [auth_dependency]
    app.include_router(classify.router)
    app.include_router(extract.router)
    
    @app.on_event("startup")
    async def startup():
        logger.info("Gateway server starting...")
        logger.info(f"Classify endpoint: {settings.runpod_classify_endpoint}")
        logger.info(f"Extract endpoint: {settings.runpod_extract_endpoint}")
    
    @app.on_event("shutdown")
    async def shutdown():
        logger.info("Gateway server shutting down...")
    
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 6.10 Dockerfile

```dockerfile
# gateway/Dockerfile

FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Create non-root user
RUN useradd -m -u 1000 appuser
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/live || exit 1

# Run with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 6.11 Requirements

```
# gateway/requirements.txt
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
httpx>=0.26.0
pydantic>=2.5.0
python-multipart>=0.0.6
```

---

## 7. Deployment

### 7.1 Build Handler Images

```bash
# Navigate to handlers directory
cd inference/handlers

# Build classification handler
cd classify
docker build -t your-registry/docproc-classify:v1 .
docker push your-registry/docproc-classify:v1

# Build extraction handler
cd ../extract
docker build -t your-registry/docproc-extract:v1 .
docker push your-registry/docproc-extract:v1
```

### 7.2 Deploy to RunPod

1. **Go to RunPod Console**: https://www.runpod.io/console/serverless

2. **Create Classification Endpoint**:
   - Click "New Endpoint"
   - Name: `docproc-classify`
   - Docker Image: `your-registry/docproc-classify:v1`
   - GPU Type: L4 or A10 (16GB+ VRAM)
   - Max Workers: 3
   - Idle Timeout: 60s
   - Note the **Endpoint ID**

3. **Create Extraction Endpoint**:
   - Click "New Endpoint"
   - Name: `docproc-extract`
   - Docker Image: `your-registry/docproc-extract:v1`
   - GPU Type: L4 or A10 (16GB+ VRAM)
   - Max Workers: 3
   - Idle Timeout: 60s
   - Note the **Endpoint ID**

4. **Get API Key**: Settings → API Keys

### 7.3 Deploy Gateway

**Option A: Docker Compose (recommended for pilot)**

```yaml
# inference/docker-compose.yml
version: '3.8'

services:
  gateway:
    build: ./gateway
    ports:
      - "8000:8000"
    environment:
      GATEWAY_API_KEY: ${GATEWAY_API_KEY}
      RUNPOD_API_KEY: ${RUNPOD_API_KEY}
      RUNPOD_CLASSIFY_ENDPOINT: ${RUNPOD_CLASSIFY_ENDPOINT}
      RUNPOD_EXTRACT_ENDPOINT: ${RUNPOD_EXTRACT_ENDPOINT}
      RUNPOD_TIMEOUT: "120"
      LOG_LEVEL: "INFO"
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/live"]
      interval: 30s
      timeout: 10s
      retries: 3
```

**Option B: Cloud Run / Railway / Fly.io**

The gateway is stateless and can deploy to any container platform.

### 7.4 Environment Variables

```bash
# inference/.env.example

# Gateway Authentication
GATEWAY_API_KEY=your-secure-api-key-here

# RunPod Configuration
RUNPOD_API_KEY=your-runpod-api-key
RUNPOD_CLASSIFY_ENDPOINT=abc123xyz
RUNPOD_EXTRACT_ENDPOINT=def456uvw
RUNPOD_TIMEOUT=120

# Optional
LOG_LEVEL=INFO
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60
```

### 7.5 Update Document Pipeline

Update your main document processing config to use the gateway:

```yaml
# config/config.yaml
inference:
  gateway_url: "http://localhost:8000"  # Or your deployed URL
  api_key: ${GATEWAY_API_KEY}
  timeout_seconds: 130  # Slightly more than RunPod timeout
  retry:
    max_attempts: 2  # Gateway handles internal retries
    base_delay_seconds: 5
```

---

## 8. Testing

### 8.1 Test Gateway Locally

```bash
# Start gateway
cd inference/gateway
uvicorn app.main:app --reload

# Test health
curl http://localhost:8000/health

# Test classification (with test image URL)
curl -X POST http://localhost:8000/v1/classify \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "image_urls": ["https://example.com/test-w2.png"]
  }'

# Test extraction
curl -X POST http://localhost:8000/v1/extract \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "image_urls": ["https://example.com/test-w2.png"],
    "doc_type": "W2"
  }'
```

### 8.2 Test RunPod Endpoints Directly

```bash
# Test classification endpoint
curl -X POST "https://api.runpod.ai/v2/${ENDPOINT_ID}/runsync" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "image_urls": ["https://example.com/test.png"]
    }
  }'
```

### 8.3 Integration Test Script

```python
# inference/tests/test_integration.py
"""Integration tests for inference pipeline."""

import os
import httpx
import pytest

GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://localhost:8000")
API_KEY = os.environ.get("GATEWAY_API_KEY", "test-key")

# Test image URLs (replace with real ShareFile URLs)
TEST_W2_URL = "https://your-sharefile.com/test-w2.png"
TEST_BANK_URL = "https://your-sharefile.com/test-bank.png"


@pytest.fixture
def client():
    return httpx.Client(
        base_url=GATEWAY_URL,
        headers={"X-API-Key": API_KEY},
        timeout=180
    )


def test_health(client):
    """Test health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["gateway"] == "healthy"


def test_classify_w2(client):
    """Test W2 classification."""
    response = client.post("/v1/classify", json={
        "image_urls": [TEST_W2_URL]
    })
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "W2"
    assert data["confidence"] >= 0.8


def test_extract_w2(client):
    """Test W2 extraction."""
    response = client.post("/v1/extract", json={
        "image_urls": [TEST_W2_URL],
        "doc_type": "W2"
    })
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert "boxes" in data["data"]
    assert "box1_wages" in data["data"]["boxes"]


def test_extract_bank_statement(client):
    """Test bank statement extraction."""
    response = client.post("/v1/extract", json={
        "image_urls": [TEST_BANK_URL],
        "doc_type": "bank_statement"
    })
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert "transactions" in data["data"]


def test_invalid_api_key(client):
    """Test authentication rejection."""
    bad_client = httpx.Client(
        base_url=GATEWAY_URL,
        headers={"X-API-Key": "wrong-key"}
    )
    response = bad_client.post("/v1/classify", json={
        "image_urls": ["https://example.com/test.png"]
    })
    assert response.status_code == 401


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

---

## 9. Cost Optimization

### 9.1 Expected Costs

| Component | Estimated Cost |
|-----------|----------------|
| RunPod L4 inference | ~$0.20/hr per worker |
| Gateway (Cloud Run) | ~$5-10/month |
| ShareFile | Existing subscription |

**Per-document cost estimate:**
- Classification: ~3-8s = $0.0002-0.0004
- Extraction: ~8-20s = $0.0004-0.0011
- **Total per doc: ~$0.001-0.002**

For 2,500 pages: **~$2.50-5.00**

### 9.2 Optimization Strategies

1. **Batch Classification**: For bulk imports, batch classify first pages only
2. **Idle Timeout**: Set to 30-60s to balance cold starts vs idle cost
3. **Right-size Workers**: Start with max_workers=2, increase if queue builds
4. **FlashBoot**: Enable for faster cold starts (trades startup time for network load)

### 9.3 Monitoring

Track these metrics:

```python
# Add to gateway for monitoring
from prometheus_client import Counter, Histogram

REQUESTS = Counter(
    'inference_requests_total',
    'Total inference requests',
    ['endpoint', 'status']
)

LATENCY = Histogram(
    'inference_latency_seconds',
    'Inference latency',
    ['endpoint'],
    buckets=[1, 2, 5, 10, 20, 30, 60, 120]
)
```

---

## Appendix A: Troubleshooting

### Cold Start Too Slow

**Symptoms:** First request takes 60-120s

**Solutions:**
1. Enable FlashBoot in RunPod endpoint settings
2. Keep min_workers=1 during business hours
3. Pre-warm with scheduled health checks

### Out of Memory

**Symptoms:** Worker crashes, "CUDA out of memory"

**Solutions:**
1. Use smaller batch size (process 1-2 pages at a time)
2. Upgrade to larger GPU (A40 instead of L4)
3. Enable model quantization (AWQ/GPTQ)

### Image Fetch Failures

**Symptoms:** "Failed to fetch image" errors

**Solutions:**
1. Verify ShareFile URL expiry (increase url_expiry_minutes)
2. Check network connectivity from RunPod
3. Add retry logic for transient failures

### JSON Parse Errors

**Symptoms:** "Could not parse JSON" in logs

**Solutions:**
1. Review prompt to emphasize JSON-only output
2. Add more robust parsing patterns
3. Lower temperature further (0.05)

---

## Appendix B: Quick Reference

### API Endpoints

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| GET | `/health` | Full health check | No |
| GET | `/ready` | Readiness probe | No |
| GET | `/live` | Liveness probe | No |
| POST | `/v1/classify` | Classify document | Yes |
| POST | `/v1/extract` | Extract data | Yes |

### Request Headers

```
X-API-Key: your-gateway-api-key
Content-Type: application/json
```

### Response Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 401 | Invalid or missing API key |
| 400 | Invalid request body |
| 502 | Inference endpoint failure |
| 504 | Inference timeout |

---

*End of Implementation Guide*
, '').replace(',', ''))
                result["boxes"][box_name] = value
            except ValueError:
                continue
    
    return result


def transform_1099(elements: list, raw_text: str, form_type: str) -> dict:
    """Transform dots.ocr output to 1099 schema."""
    result = {
        "tax_year": None,
        "payer": {
            "name": None,
            "tin": None,
            "address": None
        },
        "recipient": {
            "name": None,
            "tin": None,
            "address": None
        },
        "amounts": {}
    }
    
    # Extract tax year
    year_match = re.search(r'20\d{2}', raw_text)
    if year_match:
        result["tax_year"] = int(year_match.group())
    
    # Extract TIN/EIN
    tin_matches = re.findall(r'(\d{2}-\d{7})', raw_text)
    if len(tin_matches) >= 1:
        result["payer"]["tin"] = tin_matches[0]
    if len(tin_matches) >= 2:
        result["recipient"]["tin"] = f"***-**-{tin_matches[1][-4:]}"
    
    # Extract all amounts
    amounts = extract_amounts(raw_text)
    for i, amt in enumerate(amounts[:10]):  # Limit to first 10
        result["amounts"][f"amount_{i+1}"] = amt["value"]
    
    return result


def transform_invoice(elements: list, raw_text: str) -> dict:
    """Transform dots.ocr output to invoice schema."""
    result = {
        "vendor": {
            "name": None,
            "address": None,
            "phone": None,
            "email": None
        },
        "invoice_number": None,
        "invoice_date": None,
        "due_date": None,
        "line_items": [],
        "subtotal": None,
        "tax": None,
        "total": None
    }
    
    # Extract invoice number
    inv_match = re.search(r'(?:Invoice|Inv)[#:\s]*(\w+)', raw_text, re.IGNORECASE)
    if inv_match:
        result["invoice_number"] = inv_match.group(1)
    
    # Extract dates
    dates = extract_dates(raw_text)
    if len(dates) >= 1:
        result["invoice_date"] = dates[0]
    if len(dates) >= 2:
        result["due_date"] = dates[1]
    
    # Extract email
    email_match = re.search(r'[\w.-]+@[\w.-]+\.\w+', raw_text)
    if email_match:
        result["vendor"]["email"] = email_match.group()
    
    # Extract phone
    phone_match = re.search(r'(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})', raw_text)
    if phone_match:
        result["vendor"]["phone"] = phone_match.group(1)
    
    # Extract total (look for "Total" label)
    total_match = re.search(r'Total[:\s]*\$?([\d,]+\.?\d*)', raw_text, re.IGNORECASE)
    if total_match:
        try:
            result["total"] = float(total_match.group(1).replace(',', ''))
        except ValueError:
            pass
    
    return result


def transform_default(elements: list, raw_text: str) -> dict:
    """Default transformation for unknown document types."""
    return {
        "document_info": {
            "title": None,
            "date": extract_dates(raw_text)[0] if extract_dates(raw_text) else None,
            "type": "unknown"
        },
        "entities": {
            "names": [],
            "addresses": [],
            "dates": extract_dates(raw_text)
        },
        "amounts": extract_amounts(raw_text),
        "raw_text": raw_text[:5000]  # Truncate for storage
    }


def transform_to_document_schema(
    doc_type: str,
    layout_elements: list,
    raw_text: str
) -> dict:
    """
    Transform dots.ocr output to document-specific schema.
    
    Args:
        doc_type: Document type (W2, bank_statement, etc.)
        layout_elements: List of layout elements from dots.ocr
        raw_text: Combined raw text from all pages
        
    Returns:
        Structured data matching the document type schema
    """
    transformers = {
        "bank_statement": transform_bank_statement,
        "credit_card_statement": transform_bank_statement,  # Similar format
        "W2": transform_w2,
        "1099-INT": lambda e, t: transform_1099(e, t, "INT"),
        "1099-DIV": lambda e, t: transform_1099(e, t, "DIV"),
        "1099-MISC": lambda e, t: transform_1099(e, t, "MISC"),
        "1099-NEC": lambda e, t: transform_1099(e, t, "NEC"),
        "1099-R": lambda e, t: transform_1099(e, t, "R"),
        "invoice": transform_invoice,
    }
    
    transformer = transformers.get(doc_type, transform_default)
    return transformer(layout_elements, raw_text)
```

### 5.6 Requirements

```
# handlers/extract/requirements.txt
runpod>=1.6.0
httpx>=0.26.0
requests>=2.31.0
pillow>=10.0.0
```

### 5.7 Local Testing

```python
# handlers/extract/test_handler.py
"""Local tests for dots.ocr extraction handler."""

import json
from unittest.mock import AsyncMock, patch, MagicMock


def test_parse_dots_ocr_output():
    """Test parsing of dots.ocr output formats."""
    from handler import parse_dots_ocr_output
    
    # Test JSON format
    json_output = '{"layout_elements": [{"text": "Hello", "bbox": [0,0,100,50]}]}'
    result = parse_dots_ocr_output(json_output)
    assert "layout_elements" in result
    
    # Test bbox format
    bbox_output = "[10, 20, 100, 50] Hello World\n[10, 60, 100, 90] Second line"
    result = parse_dots_ocr_output(bbox_output)
    assert len(result["layout_elements"]) == 2
    
    # Test plain text
    plain_output = "Just some plain text\nWith multiple lines"
    result = parse_dots_ocr_output(plain_output)
    assert "raw_text" in result


def test_transform_bank_statement():
    """Test bank statement transformation."""
    from transformers import transform_bank_statement
    
    elements = [
        {"text": "Beginning Balance: $1,234.56", "bbox": [0, 0, 100, 50]},
        {"text": "Ending Balance: $2,345.67", "bbox": [0, 50, 100, 100]},
    ]
    raw_text = """
    Chase Bank
    Account: ****1234
    Statement Period: 01/01/2024 - 01/31/2024
    Beginning Balance: $1,234.56
    01/15 Direct Deposit $500.00
    01/20 Grocery Store -$45.67
    Ending Balance: $2,345.67
    """
    
    result = transform_bank_statement(elements, raw_text)
    
    assert result["header"]["bank_name"] == "Chase"
    assert result["header"]["account_number"] == "****1234"
    assert result["header"]["beginning_balance"] == 1234.56
    assert result["header"]["ending_balance"] == 2345.67


def test_transform_w2():
    """Test W2 transformation."""
    from transformers import transform_w2
    
    elements = []
    raw_text = """
    2024 W-2 Wage and Tax Statement
    Employer: 12-3456789
    Employee SSN: 123-45-6789
    Box 1 Wages: $75,000.00
    Box 2 Federal Tax Withheld: $12,500.00
    Box 3 Social security wages: $75,000.00
    Box 4 Social security tax: $4,650.00
    """
    
    result = transform_w2(elements, raw_text)
    
    assert result["tax_year"] == 2024
    assert result["employer"]["ein"] == "12-3456789"
    assert result["employee"]["ssn"] == "***-**-6789"
    assert result["boxes"]["box1_wages"] == 75000.00
    assert result["boxes"]["box2_federal_withheld"] == 12500.00


def test_extract_amounts():
    """Test amount extraction."""
    from transformers import extract_amounts
    
    text = "Total: $1,234.56 and another $500"
    amounts = extract_amounts(text)
    
    assert len(amounts) == 2
    assert amounts[0]["value"] == 1234.56
    assert amounts[1]["value"] == 500


def test_extract_dates():
    """Test date extraction."""
    from transformers import extract_dates
    
    text = "Date: 01/15/2024 and 2024-01-31 and Jan 15, 2024"
    dates = extract_dates(text)
    
    assert len(dates) >= 2


if __name__ == "__main__":
    test_parse_dots_ocr_output()
    test_transform_bank_statement()
    test_transform_w2()
    test_extract_amounts()
    test_extract_dates()
    print("All tests passed!")
```

### 5.8 dots.ocr Output Format Reference

dots.ocr returns structured output with layout elements. Here's the typical format:

```json
{
  "layout_elements": [
    {
      "bbox": [x1, y1, x2, y2],
      "category": "Text|Table|Title|Figure|Formula|Header|Footer|Caption|Footnote",
      "text": "extracted content",
      "confidence": 0.95
    }
  ],
  "tables": [
    {
      "bbox": [x1, y1, x2, y2],
      "cells": [
        {"row": 0, "col": 0, "text": "Header 1"},
        {"row": 0, "col": 1, "text": "Header 2"},
        {"row": 1, "col": 0, "text": "Value 1"},
        {"row": 1, "col": 1, "text": "Value 2"}
      ]
    }
  ]
}
```

**Category Types:**
- `Text`: Regular paragraph text
- `Title`: Document titles and headings
- `Table`: Tabular data (with cell structure)
- `Figure`: Images and diagrams
- `Formula`: Mathematical formulas
- `Header`/`Footer`: Page headers and footers
- `Caption`: Figure/table captions
- `Footnote`: Footnotes and references

### 5.9 Performance Characteristics

| Metric | Value |
|--------|-------|
| Model Size | 1.7B parameters |
| VRAM Usage | ~8GB |
| Inference Time | 2-8 seconds per page |
| Max Resolution | ~11M pixels |
| Max Tokens | 24,000 |
| Cold Start | ~30-45 seconds |

**Optimization Tips:**
- Use 200 DPI for PDF conversion (optimal for dots.ocr)
- Process pages in parallel when possible
- Cache vLLM server with min_workers=1 for production

---

## 6. Gateway Server

### 6.1 Directory Structure

```
gateway/
├── Dockerfile
├── requirements.txt
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── classify.py
│   │   ├── extract.py
│   │   └── health.py
│   ├── services/
│   │   ├── __init__.py
│   │   └── runpod_client.py
│   └── middleware/
│       ├── __init__.py
│       ├── auth.py
│       └── logging.py
└── tests/
    └── test_routes.py
```

### 6.2 Configuration

```python
# gateway/app/config.py
"""Gateway configuration from environment variables."""

import os
from dataclasses import dataclass
from functools import lru_cache


@dataclass
class Settings:
    """Application settings."""
    
    # API Authentication
    api_key: str
    
    # RunPod Configuration
    runpod_api_key: str
    runpod_classify_endpoint: str
    runpod_extract_endpoint: str
    runpod_timeout: int = 120
    runpod_max_retries: int = 3
    
    # Rate Limiting
    rate_limit_requests: int = 100  # per minute
    rate_limit_window: int = 60     # seconds
    
    # Logging
    log_level: str = "INFO"
    
    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from environment variables."""
        return cls(
            api_key=os.environ["GATEWAY_API_KEY"],
            runpod_api_key=os.environ["RUNPOD_API_KEY"],
            runpod_classify_endpoint=os.environ["RUNPOD_CLASSIFY_ENDPOINT"],
            runpod_extract_endpoint=os.environ["RUNPOD_EXTRACT_ENDPOINT"],
            runpod_timeout=int(os.environ.get("RUNPOD_TIMEOUT", "120")),
            runpod_max_retries=int(os.environ.get("RUNPOD_MAX_RETRIES", "3")),
            rate_limit_requests=int(os.environ.get("RATE_LIMIT_REQUESTS", "100")),
            rate_limit_window=int(os.environ.get("RATE_LIMIT_WINDOW", "60")),
            log_level=os.environ.get("LOG_LEVEL", "INFO")
        )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings.from_env()
```

### 6.3 RunPod Client Service

```python
# gateway/app/services/runpod_client.py
"""RunPod Serverless API client."""

import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)

RUNPOD_API_BASE = "https://api.runpod.ai/v2"


@dataclass
class RunPodResponse:
    """Response from RunPod endpoint."""
    success: bool
    data: dict | None
    error: str | None
    latency_ms: int
    job_id: str | None = None


class RunPodClient:
    """
    Async client for RunPod Serverless endpoints.
    
    Handles authentication, retries, and error handling.
    """
    
    def __init__(
        self,
        api_key: str,
        classify_endpoint: str,
        extract_endpoint: str,
        timeout: int = 120,
        max_retries: int = 3
    ):
        self.api_key = api_key
        self.classify_endpoint = classify_endpoint
        self.extract_endpoint = extract_endpoint
        self.timeout = timeout
        self.max_retries = max_retries
        
        self._client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            timeout=httpx.Timeout(timeout, connect=10)
        )
    
    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()
    
    async def classify(
        self,
        image_urls: list[str],
        prompt: str | None = None
    ) -> RunPodResponse:
        """
        Call classification endpoint.
        
        Args:
            image_urls: ShareFile pre-signed URLs
            prompt: Optional custom prompt
            
        Returns:
            RunPodResponse with classification result
        """
        return await self._call(
            endpoint_id=self.classify_endpoint,
            payload={
                "image_urls": image_urls,
                "prompt": prompt
            }
        )
    
    async def extract(
        self,
        image_urls: list[str],
        doc_type: str,
        prompt: str | None = None
    ) -> RunPodResponse:
        """
        Call extraction endpoint.
        
        Args:
            image_urls: ShareFile pre-signed URLs
            doc_type: Document type for prompt selection
            prompt: Optional custom prompt
            
        Returns:
            RunPodResponse with extraction result
        """
        return await self._call(
            endpoint_id=self.extract_endpoint,
            payload={
                "image_urls": image_urls,
                "doc_type": doc_type,
                "prompt": prompt
            }
        )
    
    async def health_check(self) -> dict:
        """Check health of both endpoints."""
        results = {}
        
        for name, endpoint_id in [
            ("classify", self.classify_endpoint),
            ("extract", self.extract_endpoint)
        ]:
            try:
                url = f"{RUNPOD_API_BASE}/{endpoint_id}/health"
                response = await self._client.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    results[name] = {
                        "status": "healthy",
                        "workers": data.get("workers", {})
                    }
                elif response.status_code == 204:
                    results[name] = {
                        "status": "idle",
                        "workers": {"idle": 0, "running": 0}
                    }
                else:
                    results[name] = {
                        "status": "unhealthy",
                        "error": f"HTTP {response.status_code}"
                    }
            except Exception as e:
                results[name] = {
                    "status": "unreachable",
                    "error": str(e)
                }
        
        return results
    
    async def _call(
        self,
        endpoint_id: str,
        payload: dict
    ) -> RunPodResponse:
        """
        Make a synchronous call to a RunPod endpoint.
        
        Uses /runsync for immediate response.
        """
        url = f"{RUNPOD_API_BASE}/{endpoint_id}/runsync"
        
        last_error = None
        base_delay = 5.0
        
        for attempt in range(self.max_retries):
            start_time = time.time()
            
            try:
                response = await self._client.post(
                    url,
                    json={"input": payload}
                )
                
                latency_ms = int((time.time() - start_time) * 1000)
                data = response.json()
                
                status = data.get("status")
                job_id = data.get("id")
                
                # Success
                if status == "COMPLETED":
                    return RunPodResponse(
                        success=True,
                        data=data.get("output", {}),
                        error=None,
                        latency_ms=data.get("executionTime", latency_ms),
                        job_id=job_id
                    )
                
                # Failed
                if status == "FAILED":
                    error_msg = data.get("error", "Unknown error")
                    return RunPodResponse(
                        success=False,
                        data=None,
                        error=error_msg,
                        latency_ms=latency_ms,
                        job_id=job_id
                    )
                
                # Timeout/in-progress
                if status in ("IN_QUEUE", "IN_PROGRESS"):
                    last_error = f"Job timed out in status: {status}"
                    logger.warning(f"RunPod job {job_id} timed out: {status}")
                
                # HTTP errors
                if response.status_code == 401:
                    return RunPodResponse(
                        success=False,
                        data=None,
                        error="Authentication failed",
                        latency_ms=latency_ms
                    )
                
                if response.status_code == 400:
                    return RunPodResponse(
                        success=False,
                        data=None,
                        error=f"Bad request: {data.get('error', 'Unknown')}",
                        latency_ms=latency_ms
                    )
                
                # Retry on server errors
                if response.status_code in (429, 500, 502, 503, 504):
                    last_error = f"HTTP {response.status_code}"
                    logger.warning(
                        f"RunPod error {response.status_code}, "
                        f"attempt {attempt + 1}/{self.max_retries}"
                    )
                    
            except httpx.TimeoutException:
                last_error = "Request timeout"
                logger.warning(
                    f"RunPod timeout, attempt {attempt + 1}/{self.max_retries}"
                )
                
            except httpx.RequestError as e:
                last_error = str(e)
                logger.warning(
                    f"RunPod request error: {e}, "
                    f"attempt {attempt + 1}/{self.max_retries}"
                )
            
            # Exponential backoff
            if attempt < self.max_retries - 1:
                delay = min(base_delay * (2 ** attempt), 60)
                logger.info(f"Retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)
        
        # All retries exhausted
        return RunPodResponse(
            success=False,
            data=None,
            error=f"Max retries exceeded: {last_error}",
            latency_ms=0
        )


import asyncio  # Import at top in real code
```

### 6.4 Authentication Middleware

```python
# gateway/app/middleware/auth.py
"""API key authentication middleware."""

import logging
import secrets
from typing import Callable

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

logger = logging.getLogger(__name__)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def create_auth_dependency(valid_api_key: str) -> Callable:
    """
    Create an API key validation dependency.
    
    Args:
        valid_api_key: The expected API key
        
    Returns:
        FastAPI dependency function
    """
    async def verify_api_key(
        api_key: str = Security(api_key_header)
    ) -> str:
        if api_key is None:
            logger.warning("Missing API key in request")
            raise HTTPException(
                status_code=401,
                detail="Missing API key"
            )
        
        # Constant-time comparison to prevent timing attacks
        if not secrets.compare_digest(api_key, valid_api_key):
            logger.warning("Invalid API key attempt")
            raise HTTPException(
                status_code=401,
                detail="Invalid API key"
            )
        
        return api_key
    
    return verify_api_key
```

### 6.5 Logging Middleware

```python
# gateway/app/middleware/logging.py
"""Request logging middleware."""

import logging
import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all requests with timing and request ID."""
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        # Generate request ID
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        
        # Log request
        logger.info(
            f"[{request_id}] {request.method} {request.url.path} "
            f"from {request.client.host if request.client else 'unknown'}"
        )
        
        # Process request
        start_time = time.time()
        
        try:
            response = await call_next(request)
            
            # Log response
            duration_ms = int((time.time() - start_time) * 1000)
            logger.info(
                f"[{request_id}] {response.status_code} "
                f"in {duration_ms}ms"
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.error(
                f"[{request_id}] Error after {duration_ms}ms: {e}"
            )
            raise
```

### 6.6 Route: Classification

```python
# gateway/app/routes/classify.py
"""Classification endpoint."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..config import Settings, get_settings
from ..middleware.auth import create_auth_dependency
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
    settings: Annotated[Settings, Depends(get_settings)],
    _api_key: Annotated[str, Depends(create_auth_dependency)]
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


# Need to inject the actual API key at startup
def configure_router(api_key: str):
    """Configure the router with the API key for auth."""
    router.dependencies = [Depends(create_auth_dependency(api_key))]
```

### 6.7 Route: Extraction

```python
# gateway/app/routes/extract.py
"""Extraction endpoint."""

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..config import Settings, get_settings
from ..middleware.auth import create_auth_dependency
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
    confidence: dict[str, Any] = Field(..., description="Per-field confidence scores")
    doc_type: str = Field(..., description="Document type used for extraction")
    page_count: int = Field(..., description="Number of pages processed")
    latency_ms: int = Field(..., description="Inference latency in milliseconds")


@router.post("/extract", response_model=ExtractResponse)
async def extract_document(
    request: ExtractRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    _api_key: Annotated[str, Depends(create_auth_dependency)]
) -> ExtractResponse:
    """
    Extract structured data from document images.
    
    Returns extracted fields with per-field confidence scores.
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
        # Call extraction endpoint
        response = await client.extract(
            image_urls=request.image_urls,
            doc_type=request.doc_type,
            prompt=request.prompt
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
            doc_type=data.get("doc_type", request.doc_type),
            page_count=data.get("page_count", len(request.image_urls)),
            latency_ms=response.latency_ms
        )
        
    finally:
        await client.close()
```

### 6.8 Route: Health

```python
# gateway/app/routes/health.py
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
```

### 6.9 Main Application

```python
# gateway/app/main.py
"""Gateway server main application."""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .middleware.logging import RequestLoggingMiddleware
from .middleware.auth import create_auth_dependency
from .routes import classify, extract, health

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title="Document Processing Inference Gateway",
        description="Gateway for document classification and extraction services",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Add CORS middleware (configure for your needs)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Restrict in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add request logging
    app.add_middleware(RequestLoggingMiddleware)
    
    # Create auth dependency with configured API key
    auth_dependency = create_auth_dependency(settings.api_key)
    
    # Include routers
    # Health routes (no auth)
    app.include_router(health.router)
    
    # Inference routes (with auth)
    classify.router.dependencies = [auth_dependency]
    extract.router.dependencies = [auth_dependency]
    app.include_router(classify.router)
    app.include_router(extract.router)
    
    @app.on_event("startup")
    async def startup():
        logger.info("Gateway server starting...")
        logger.info(f"Classify endpoint: {settings.runpod_classify_endpoint}")
        logger.info(f"Extract endpoint: {settings.runpod_extract_endpoint}")
    
    @app.on_event("shutdown")
    async def shutdown():
        logger.info("Gateway server shutting down...")
    
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 6.10 Dockerfile

```dockerfile
# gateway/Dockerfile

FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Create non-root user
RUN useradd -m -u 1000 appuser
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/live || exit 1

# Run with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 6.11 Requirements

```
# gateway/requirements.txt
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
httpx>=0.26.0
pydantic>=2.5.0
python-multipart>=0.0.6
```

---

## 7. Deployment

### 7.1 Build Handler Images

```bash
# Navigate to handlers directory
cd inference/handlers

# Build classification handler
cd classify
docker build -t your-registry/docproc-classify:v1 .
docker push your-registry/docproc-classify:v1

# Build extraction handler
cd ../extract
docker build -t your-registry/docproc-extract:v1 .
docker push your-registry/docproc-extract:v1
```

### 7.2 Deploy to RunPod

1. **Go to RunPod Console**: https://www.runpod.io/console/serverless

2. **Create Classification Endpoint**:
   - Click "New Endpoint"
   - Name: `docproc-classify`
   - Docker Image: `your-registry/docproc-classify:v1`
   - GPU Type: L4 or A10 (16GB+ VRAM)
   - Max Workers: 3
   - Idle Timeout: 60s
   - Note the **Endpoint ID**

3. **Create Extraction Endpoint**:
   - Click "New Endpoint"
   - Name: `docproc-extract`
   - Docker Image: `your-registry/docproc-extract:v1`
   - GPU Type: L4 or A10 (16GB+ VRAM)
   - Max Workers: 3
   - Idle Timeout: 60s
   - Note the **Endpoint ID**

4. **Get API Key**: Settings → API Keys

### 7.3 Deploy Gateway

**Option A: Docker Compose (recommended for pilot)**

```yaml
# inference/docker-compose.yml
version: '3.8'

services:
  gateway:
    build: ./gateway
    ports:
      - "8000:8000"
    environment:
      GATEWAY_API_KEY: ${GATEWAY_API_KEY}
      RUNPOD_API_KEY: ${RUNPOD_API_KEY}
      RUNPOD_CLASSIFY_ENDPOINT: ${RUNPOD_CLASSIFY_ENDPOINT}
      RUNPOD_EXTRACT_ENDPOINT: ${RUNPOD_EXTRACT_ENDPOINT}
      RUNPOD_TIMEOUT: "120"
      LOG_LEVEL: "INFO"
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/live"]
      interval: 30s
      timeout: 10s
      retries: 3
```

**Option B: Cloud Run / Railway / Fly.io**

The gateway is stateless and can deploy to any container platform.

### 7.4 Environment Variables

```bash
# inference/.env.example

# Gateway Authentication
GATEWAY_API_KEY=your-secure-api-key-here

# RunPod Configuration
RUNPOD_API_KEY=your-runpod-api-key
RUNPOD_CLASSIFY_ENDPOINT=abc123xyz
RUNPOD_EXTRACT_ENDPOINT=def456uvw
RUNPOD_TIMEOUT=120

# Optional
LOG_LEVEL=INFO
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60
```

### 7.5 Update Document Pipeline

Update your main document processing config to use the gateway:

```yaml
# config/config.yaml
inference:
  gateway_url: "http://localhost:8000"  # Or your deployed URL
  api_key: ${GATEWAY_API_KEY}
  timeout_seconds: 130  # Slightly more than RunPod timeout
  retry:
    max_attempts: 2  # Gateway handles internal retries
    base_delay_seconds: 5
```

---

## 8. Testing

### 8.1 Test Gateway Locally

```bash
# Start gateway
cd inference/gateway
uvicorn app.main:app --reload

# Test health
curl http://localhost:8000/health

# Test classification (with test image URL)
curl -X POST http://localhost:8000/v1/classify \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "image_urls": ["https://example.com/test-w2.png"]
  }'

# Test extraction
curl -X POST http://localhost:8000/v1/extract \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "image_urls": ["https://example.com/test-w2.png"],
    "doc_type": "W2"
  }'
```

### 8.2 Test RunPod Endpoints Directly

```bash
# Test classification endpoint
curl -X POST "https://api.runpod.ai/v2/${ENDPOINT_ID}/runsync" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "image_urls": ["https://example.com/test.png"]
    }
  }'
```

### 8.3 Integration Test Script

```python
# inference/tests/test_integration.py
"""Integration tests for inference pipeline."""

import os
import httpx
import pytest

GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://localhost:8000")
API_KEY = os.environ.get("GATEWAY_API_KEY", "test-key")

# Test image URLs (replace with real ShareFile URLs)
TEST_W2_URL = "https://your-sharefile.com/test-w2.png"
TEST_BANK_URL = "https://your-sharefile.com/test-bank.png"


@pytest.fixture
def client():
    return httpx.Client(
        base_url=GATEWAY_URL,
        headers={"X-API-Key": API_KEY},
        timeout=180
    )


def test_health(client):
    """Test health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["gateway"] == "healthy"


def test_classify_w2(client):
    """Test W2 classification."""
    response = client.post("/v1/classify", json={
        "image_urls": [TEST_W2_URL]
    })
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "W2"
    assert data["confidence"] >= 0.8


def test_extract_w2(client):
    """Test W2 extraction."""
    response = client.post("/v1/extract", json={
        "image_urls": [TEST_W2_URL],
        "doc_type": "W2"
    })
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert "boxes" in data["data"]
    assert "box1_wages" in data["data"]["boxes"]


def test_extract_bank_statement(client):
    """Test bank statement extraction."""
    response = client.post("/v1/extract", json={
        "image_urls": [TEST_BANK_URL],
        "doc_type": "bank_statement"
    })
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert "transactions" in data["data"]


def test_invalid_api_key(client):
    """Test authentication rejection."""
    bad_client = httpx.Client(
        base_url=GATEWAY_URL,
        headers={"X-API-Key": "wrong-key"}
    )
    response = bad_client.post("/v1/classify", json={
        "image_urls": ["https://example.com/test.png"]
    })
    assert response.status_code == 401


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

---

## 9. Cost Optimization

### 9.1 Expected Costs

| Component | Estimated Cost |
|-----------|----------------|
| RunPod L4 inference | ~$0.20/hr per worker |
| Gateway (Cloud Run) | ~$5-10/month |
| ShareFile | Existing subscription |

**Per-document cost estimate:**
- Classification: ~3-8s = $0.0002-0.0004
- Extraction: ~8-20s = $0.0004-0.0011
- **Total per doc: ~$0.001-0.002**

For 2,500 pages: **~$2.50-5.00**

### 9.2 Optimization Strategies

1. **Batch Classification**: For bulk imports, batch classify first pages only
2. **Idle Timeout**: Set to 30-60s to balance cold starts vs idle cost
3. **Right-size Workers**: Start with max_workers=2, increase if queue builds
4. **FlashBoot**: Enable for faster cold starts (trades startup time for network load)

### 9.3 Monitoring

Track these metrics:

```python
# Add to gateway for monitoring
from prometheus_client import Counter, Histogram

REQUESTS = Counter(
    'inference_requests_total',
    'Total inference requests',
    ['endpoint', 'status']
)

LATENCY = Histogram(
    'inference_latency_seconds',
    'Inference latency',
    ['endpoint'],
    buckets=[1, 2, 5, 10, 20, 30, 60, 120]
)
```

---

## Appendix A: Troubleshooting

### Cold Start Too Slow

**Symptoms:** First request takes 60-120s

**Solutions:**
1. Enable FlashBoot in RunPod endpoint settings
2. Keep min_workers=1 during business hours
3. Pre-warm with scheduled health checks

### Out of Memory

**Symptoms:** Worker crashes, "CUDA out of memory"

**Solutions:**
1. Use smaller batch size (process 1-2 pages at a time)
2. Upgrade to larger GPU (A40 instead of L4)
3. Enable model quantization (AWQ/GPTQ)

### Image Fetch Failures

**Symptoms:** "Failed to fetch image" errors

**Solutions:**
1. Verify ShareFile URL expiry (increase url_expiry_minutes)
2. Check network connectivity from RunPod
3. Add retry logic for transient failures

### JSON Parse Errors

**Symptoms:** "Could not parse JSON" in logs

**Solutions:**
1. Review prompt to emphasize JSON-only output
2. Add more robust parsing patterns
3. Lower temperature further (0.05)

---

## Appendix B: Quick Reference

### API Endpoints

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| GET | `/health` | Full health check | No |
| GET | `/ready` | Readiness probe | No |
| GET | `/live` | Liveness probe | No |
| POST | `/v1/classify` | Classify document | Yes |
| POST | `/v1/extract` | Extract data | Yes |

### Request Headers

```
X-API-Key: your-gateway-api-key
Content-Type: application/json
```

### Response Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 401 | Invalid or missing API key |
| 400 | Invalid request body |
| 502 | Inference endpoint failure |
| 504 | Inference timeout |

---

*End of Implementation Guide*
