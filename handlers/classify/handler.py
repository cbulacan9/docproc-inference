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
