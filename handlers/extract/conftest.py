"""
Pytest fixtures for extraction handler tests.
"""

import pytest
from io import BytesIO
from PIL import Image
from unittest.mock import MagicMock


def _create_image_bytes(mode: str = 'RGB', color: tuple = (255, 0, 0)) -> bytes:
    """Create PNG image bytes for testing."""
    if mode == 'P':
        img = Image.new('P', (100, 100))
        img.putpalette([i for i in range(256)] * 3)
    else:
        img = Image.new(mode, (100, 100), color=color)
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    return buffer.getvalue()


# Pre-generate image bytes at module load
_RGB_BYTES = _create_image_bytes('RGB', (255, 0, 0))
_RGBA_BYTES = _create_image_bytes('RGBA', (0, 255, 0, 128))
_GRAYSCALE_BYTES = _create_image_bytes('L', 128)
_PALETTE_BYTES = _create_image_bytes('P')


@pytest.fixture
def valid_image_bytes() -> bytes:
    """Return valid RGB PNG image bytes."""
    return _RGB_BYTES


@pytest.fixture
def rgba_image_bytes() -> bytes:
    """Return RGBA PNG image bytes."""
    return _RGBA_BYTES


@pytest.fixture
def grayscale_image_bytes() -> bytes:
    """Return grayscale PNG image bytes."""
    return _GRAYSCALE_BYTES


@pytest.fixture
def palette_image_bytes() -> bytes:
    """Return palette-mode PNG image bytes."""
    return _PALETTE_BYTES


@pytest.fixture
def mock_successful_response(valid_image_bytes):
    """Create a mock successful HTTP response with image data."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = valid_image_bytes
    mock_response.raise_for_status = MagicMock()
    return mock_response


@pytest.fixture
def mock_http_error_response():
    """Create a mock HTTP error response."""
    from requests.exceptions import HTTPError
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.raise_for_status.side_effect = HTTPError("404 Not Found")
    return mock_response


@pytest.fixture
def sample_job():
    """Return a minimal sample job input."""
    return {
        "input": {
            "image_urls": ["https://example.com/doc1.png"],
            "doc_type": "bank_statement"
        }
    }


@pytest.fixture
def sample_job_multi_page():
    """Return a sample job with multiple pages."""
    return {
        "input": {
            "image_urls": [
                "https://example.com/doc1_page1.png",
                "https://example.com/doc1_page2.png"
            ],
            "doc_type": "W2"
        }
    }


@pytest.fixture
def sample_job_with_prompt_mode():
    """Return a sample job with custom prompt mode."""
    return {
        "input": {
            "image_urls": ["https://example.com/doc.png"],
            "doc_type": "other",
            "prompt_mode": "ocr"
        }
    }


@pytest.fixture
def sample_vllm_response():
    """Return a sample vLLM API response."""
    return {
        "choices": [
            {
                "message": {
                    "content": '[100, 50, 400, 80] Account Statement\n[100, 100, 400, 130] Balance: $1,234.56'
                }
            }
        ],
        "usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 50,
            "total_tokens": 1050
        }
    }


@pytest.fixture
def sample_vllm_json_response():
    """Return a sample vLLM response with JSON format."""
    import json
    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps({
                        "layout_elements": [
                            {
                                "bbox": [100, 50, 400, 80],
                                "category": "Title",
                                "text": "Bank Statement",
                                "confidence": 0.98
                            },
                            {
                                "bbox": [100, 100, 400, 130],
                                "category": "Text",
                                "text": "Account: ****1234",
                                "confidence": 0.95
                            }
                        ]
                    })
                }
            }
        ],
        "usage": {}
    }


@pytest.fixture
def sample_extraction_result():
    """Return a sample extraction result."""
    return {
        "data": {
            "header": {
                "bank_name": "Chase",
                "account_number": "****1234",
                "statement_period": "01/01/2024 - 01/31/2024"
            },
            "transactions": [],
            "summary": {}
        },
        "raw_ocr": {
            "pages": [{"page": 1, "layout_elements": []}],
            "combined_text": "Chase Bank Statement"
        },
        "confidence": {
            "overall": 0.95,
            "page_count": 1,
            "element_count": 5
        },
        "page_count": 1
    }


@pytest.fixture
def sample_layout_elements():
    """Return sample layout elements for transformer tests."""
    return [
        {
            "bbox": [100, 50, 400, 80],
            "category": "Title",
            "text": "Chase Bank Statement",
            "confidence": 0.98
        },
        {
            "bbox": [100, 100, 300, 130],
            "category": "Text",
            "text": "Account: ****1234",
            "confidence": 0.95
        },
        {
            "bbox": [100, 150, 400, 180],
            "category": "Text",
            "text": "Beginning Balance: $5,000.00",
            "confidence": 0.96
        },
        {
            "bbox": [100, 200, 400, 230],
            "category": "Text",
            "text": "Ending Balance: $4,500.00",
            "confidence": 0.94
        }
    ]


@pytest.fixture
def sample_bank_statement_text():
    """Return sample bank statement raw text."""
    return """
    Chase Bank
    Account Statement
    Account: ****1234
    Statement Period: 01/01/2024 - 01/31/2024

    Beginning Balance: $5,000.00

    01/05 Direct Deposit $2,000.00
    01/10 Electric Bill -$150.00
    01/15 Grocery Store -$75.50

    Ending Balance: $6,774.50

    Total Credits: $2,000.00
    Total Debits: $225.50
    """


@pytest.fixture
def sample_w2_text():
    """Return sample W2 form raw text."""
    return """
    W-2 Wage and Tax Statement 2023

    Employee SSN: XXX-XX-1234
    Employer EIN: 12-3456789

    Box 1 Wages, tips: $75,000.00
    Box 2 Federal income tax withheld: $12,000.00
    Box 3 Social security wages: $75,000.00
    Box 4 Social security tax withheld: $4,650.00
    Box 5 Medicare wages: $75,000.00
    Box 6 Medicare tax withheld: $1,087.50
    """
