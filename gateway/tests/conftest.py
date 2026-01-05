"""Pytest fixtures for gateway tests."""

import os
import pytest
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.services.runpod_client import RunPodResponse


# =============================================================================
# Environment fixtures
# =============================================================================

@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set up mock environment variables for testing."""
    monkeypatch.setenv("GATEWAY_API_KEY", "test-api-key-12345")
    monkeypatch.setenv("RUNPOD_API_KEY", "test-runpod-key")
    monkeypatch.setenv("RUNPOD_CLASSIFY_ENDPOINT", "classify-endpoint-id")
    monkeypatch.setenv("RUNPOD_EXTRACT_ENDPOINT", "extract-endpoint-id")
    monkeypatch.setenv("RUNPOD_TIMEOUT", "60")
    monkeypatch.setenv("RUNPOD_MAX_RETRIES", "2")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")


@pytest.fixture
def test_settings(mock_env_vars):
    """Create test Settings instance."""
    # Clear lru_cache before creating settings
    from app.config import get_settings, Settings
    get_settings.cache_clear()
    return Settings.from_env()


# =============================================================================
# Mock RunPod Client fixture
# =============================================================================

@pytest.fixture
def mock_runpod_client():
    """
    Create a mock RunPodClient for injection into app.state.

    Tests can configure return values via the mock's methods:
        mock_runpod_client.classify.return_value = RunPodResponse(...)
        mock_runpod_client.extract.return_value = RunPodResponse(...)
        mock_runpod_client.health_check.return_value = {...}
    """
    mock = AsyncMock()
    mock.classify = AsyncMock(return_value=RunPodResponse(
        success=True,
        data={"type": "W2", "confidence": 0.95, "reasoning": "Default mock response"},
        error=None,
        latency_ms=1000,
        job_id="mock-job-123"
    ))
    mock.extract = AsyncMock(return_value=RunPodResponse(
        success=True,
        data={"data": {}, "confidence": {}, "doc_type": "W2", "page_count": 1},
        error=None,
        latency_ms=1000,
        job_id="mock-job-456"
    ))
    mock.health_check = AsyncMock(return_value={
        "classify": {"status": "healthy", "workers": {"idle": 1}},
        "extract": {"status": "healthy", "workers": {"idle": 1}}
    })
    mock.close = AsyncMock()
    return mock


# =============================================================================
# Client fixtures
# =============================================================================

@pytest.fixture
def test_client(mock_env_vars, mock_runpod_client):
    """Create FastAPI test client with mocked RunPod client."""
    from app.config import get_settings
    get_settings.cache_clear()

    # Reset router dependencies before creating app
    # This is needed because routers are shared module-level objects
    from app.routes import classify, extract
    classify.router.dependencies = []
    extract.router.dependencies = []

    # Patch RunPodClient to return our mock instead of creating a real client
    with patch('app.main.RunPodClient', return_value=mock_runpod_client):
        from app.main import create_app
        app = create_app()

        with TestClient(app) as client:
            yield client


@pytest.fixture
def auth_headers():
    """Headers with valid API key."""
    return {"X-API-Key": "test-api-key-12345"}


@pytest.fixture
def invalid_auth_headers():
    """Headers with invalid API key."""
    return {"X-API-Key": "wrong-api-key"}


# =============================================================================
# RunPod response fixtures
# =============================================================================

@pytest.fixture
def mock_runpod_classify_success():
    """Mock successful classification response."""
    return {
        "id": "job-123",
        "status": "COMPLETED",
        "output": {
            "type": "W2",
            "confidence": 0.95,
            "reasoning": "Document contains W-2 form elements"
        },
        "executionTime": 1500
    }


@pytest.fixture
def mock_runpod_extract_success():
    """Mock successful extraction response."""
    return {
        "id": "job-456",
        "status": "COMPLETED",
        "output": {
            "data": {
                "employee": {"name": "John Doe", "ssn": "XXX-XX-1234"},
                "wages": {"box1_wages": 75000.00}
            },
            "confidence": {"overall": 0.92},
            "doc_type": "W2",
            "page_count": 1
        },
        "executionTime": 2500
    }


@pytest.fixture
def mock_runpod_failure():
    """Mock failed RunPod response."""
    return {
        "id": "job-789",
        "status": "FAILED",
        "error": "Model inference failed"
    }


@pytest.fixture
def mock_runpod_health_success():
    """Mock healthy RunPod endpoints."""
    return {
        "workers": {
            "idle": 1,
            "running": 0
        }
    }


# =============================================================================
# HTTP mock fixtures
# =============================================================================

@pytest.fixture
def mock_httpx_client():
    """Create a mock httpx.AsyncClient."""
    mock = AsyncMock()
    mock.aclose = AsyncMock()
    return mock


@pytest.fixture
def sample_classify_request():
    """Sample classification request body."""
    return {
        "image_urls": ["https://example.com/doc1.png"]
    }


@pytest.fixture
def sample_extract_request():
    """Sample extraction request body."""
    return {
        "image_urls": ["https://example.com/doc1.png"],
        "doc_type": "W2"
    }
