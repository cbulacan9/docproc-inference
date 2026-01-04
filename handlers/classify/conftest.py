"""Pytest fixtures for classification handler tests."""

import pytest
from unittest.mock import MagicMock
from io import BytesIO
from PIL import Image


def _create_image_bytes(mode: str, color) -> bytes:
    """Create image bytes for a given mode and color."""
    if mode == 'P':
        # Palette mode needs special handling
        img = Image.new('P', (1, 1))
        img.putpalette([255, 0, 0] * 256)  # Red palette
        img.putpixel((0, 0), 0)
    else:
        img = Image.new(mode, (1, 1), color=color)
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    return buffer.getvalue()


# Pre-generate image bytes for different modes
VALID_PNG_BYTES = _create_image_bytes('RGB', 'red')
RGBA_PNG_BYTES = _create_image_bytes('RGBA', (255, 0, 0, 255))
GRAYSCALE_PNG_BYTES = _create_image_bytes('L', 128)
PALETTE_PNG_BYTES = _create_image_bytes('P', None)


@pytest.fixture
def valid_image_bytes():
    """Return valid RGB PNG image bytes."""
    return VALID_PNG_BYTES


@pytest.fixture
def rgba_image_bytes():
    """Return RGBA PNG image bytes (with alpha channel)."""
    return RGBA_PNG_BYTES


@pytest.fixture
def grayscale_image_bytes():
    """Return grayscale (L mode) PNG image bytes."""
    return GRAYSCALE_PNG_BYTES


@pytest.fixture
def palette_image_bytes():
    """Return palette (P mode) PNG image bytes."""
    return PALETTE_PNG_BYTES


@pytest.fixture
def mock_successful_response(valid_image_bytes):
    """Return a mock successful HTTP response with image data."""
    mock_response = MagicMock()
    mock_response.content = valid_image_bytes
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    return mock_response


@pytest.fixture
def mock_http_error_response():
    """Return a mock HTTP error response."""
    import requests
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
    return mock_response


@pytest.fixture
def sample_job():
    """Return a sample RunPod job input."""
    return {
        "input": {
            "image_urls": ["https://example.com/document.png"]
        }
    }


@pytest.fixture
def sample_job_with_prompt():
    """Return a sample RunPod job with custom prompt."""
    return {
        "input": {
            "image_urls": ["https://example.com/document.png"],
            "prompt": "Custom classification prompt"
        }
    }


@pytest.fixture
def sample_classification_result():
    """Return a sample classification result."""
    return {
        "type": "W2",
        "confidence": 0.95,
        "reasoning": "Document contains W-2 form header and wage information"
    }
