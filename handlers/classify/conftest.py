"""Pytest fixtures for classification handler tests."""

import pytest
from unittest.mock import MagicMock, patch
from io import BytesIO

# Create a valid PNG image using PIL to ensure it's properly formatted
def _create_valid_png():
    """Create a valid 1x1 red PNG image."""
    from PIL import Image
    from io import BytesIO
    img = Image.new('RGB', (1, 1), color='red')
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    return buffer.getvalue()

VALID_PNG_BYTES = _create_valid_png()


@pytest.fixture
def valid_image_bytes():
    """Return valid PNG image bytes."""
    return VALID_PNG_BYTES


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


@pytest.fixture
def mock_model_and_processor():
    """Mock the MODEL and PROCESSOR globals."""
    with patch('handler.MODEL') as mock_model, \
         patch('handler.PROCESSOR') as mock_processor:

        # Setup mock model
        mock_model.device = 'cpu'
        mock_model.generate.return_value = MagicMock()

        # Setup mock processor
        mock_processor.apply_chat_template.return_value = "processed template"
        mock_processor.return_value = MagicMock()
        mock_processor.return_value.to.return_value = {
            'input_ids': MagicMock(shape=(1, 10))
        }
        mock_processor.batch_decode.return_value = ['{"type": "W2", "confidence": 0.95, "reasoning": "test"}']
        mock_processor.tokenizer.pad_token_id = 0

        yield mock_model, mock_processor
