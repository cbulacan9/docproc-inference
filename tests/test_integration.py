"""Integration tests for the document processing inference service.

These tests run against a live gateway with real RunPod endpoints.
Skip tests if INTEGRATION_TEST_URL is not set.

Usage:
    export INTEGRATION_TEST_URL=http://localhost:8000
    export GATEWAY_API_KEY=your-test-key
    pytest tests/test_integration.py -v -m integration
"""

import os

import httpx
import pytest

GATEWAY_URL = os.getenv("INTEGRATION_TEST_URL", "")
API_KEY = os.getenv("GATEWAY_API_KEY", "")

# Sample test image URL (public domain W-2 form example)
# TODO: Replace with actual test document URLs in your environment
SAMPLE_IMAGE_URL = os.getenv(
    "TEST_IMAGE_URL",
    "https://www.irs.gov/pub/irs-pdf/fw2.pdf"
)

pytestmark = pytest.mark.integration


def skip_if_no_gateway():
    """Skip test if gateway URL is not configured."""
    if not GATEWAY_URL:
        pytest.skip("INTEGRATION_TEST_URL not set")


@pytest.fixture
def client():
    """HTTP client without authentication."""
    skip_if_no_gateway()
    return httpx.Client(base_url=GATEWAY_URL, timeout=30.0)


@pytest.fixture
def auth_client():
    """HTTP client with authentication header."""
    skip_if_no_gateway()
    if not API_KEY:
        pytest.skip("GATEWAY_API_KEY not set")
    return httpx.Client(
        base_url=GATEWAY_URL,
        headers={"X-API-Key": API_KEY},
        timeout=120.0
    )


# =============================================================================
# Health Endpoint Tests
# =============================================================================


def test_live_endpoint(client):
    """GET /live returns 200 with status."""
    response = client.get("/live")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "live"


def test_ready_endpoint(client):
    """GET /ready returns 200 with status."""
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"


def test_health_endpoint(client):
    """GET /health returns 200 with endpoint status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "gateway" in data
    assert "endpoints" in data
    assert data["gateway"] == "healthy"


# =============================================================================
# Authentication Tests
# =============================================================================


def test_classify_requires_auth(client):
    """POST /v1/classify without API key returns 401."""
    response = client.post(
        "/v1/classify",
        json={"image_urls": ["https://example.com/test.png"]}
    )
    assert response.status_code == 401


def test_extract_requires_auth(client):
    """POST /v1/extract without API key returns 401."""
    response = client.post(
        "/v1/extract",
        json={
            "image_urls": ["https://example.com/test.png"],
            "doc_type": "W2"
        }
    )
    assert response.status_code == 401


def test_invalid_api_key_rejected(client):
    """POST /v1/classify with invalid API key returns 401."""
    response = client.post(
        "/v1/classify",
        json={"image_urls": ["https://example.com/test.png"]},
        headers={"X-API-Key": "invalid-key-12345"}
    )
    assert response.status_code == 401


# =============================================================================
# Classification Endpoint Tests
# =============================================================================


def test_classify_with_valid_image(auth_client):
    """POST /v1/classify returns classification result."""
    response = auth_client.post(
        "/v1/classify",
        json={"image_urls": [SAMPLE_IMAGE_URL]}
    )
    assert response.status_code == 200
    data = response.json()

    # Verify response structure
    assert "type" in data
    assert "confidence" in data
    assert "latency_ms" in data

    # Verify data types
    assert isinstance(data["type"], str)
    assert isinstance(data["confidence"], (int, float))
    assert 0 <= data["confidence"] <= 1
    assert isinstance(data["latency_ms"], int)


def test_classify_validation_error(auth_client):
    """POST /v1/classify with invalid request returns 422."""
    response = auth_client.post(
        "/v1/classify",
        json={"image_urls": []}  # Empty list should fail validation
    )
    assert response.status_code == 422


# =============================================================================
# Extraction Endpoint Tests
# =============================================================================


def test_extract_with_valid_image(auth_client):
    """POST /v1/extract returns extracted data."""
    response = auth_client.post(
        "/v1/extract",
        json={
            "image_urls": [SAMPLE_IMAGE_URL],
            "doc_type": "W2"
        }
    )
    assert response.status_code == 200
    data = response.json()

    # Verify response structure
    assert "data" in data
    assert "confidence" in data
    assert "doc_type" in data
    assert "page_count" in data
    assert "latency_ms" in data

    # Verify data types
    assert isinstance(data["data"], dict)
    assert isinstance(data["confidence"], dict)
    assert isinstance(data["doc_type"], str)
    assert isinstance(data["page_count"], int)
    assert isinstance(data["latency_ms"], int)


def test_extract_with_bank_statement(auth_client):
    """POST /v1/extract with bank_statement type returns data."""
    response = auth_client.post(
        "/v1/extract",
        json={
            "image_urls": [SAMPLE_IMAGE_URL],
            "doc_type": "bank_statement"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["doc_type"] == "bank_statement"


def test_extract_validation_error(auth_client):
    """POST /v1/extract with missing doc_type returns 422."""
    response = auth_client.post(
        "/v1/extract",
        json={"image_urls": [SAMPLE_IMAGE_URL]}  # Missing doc_type
    )
    assert response.status_code == 422
