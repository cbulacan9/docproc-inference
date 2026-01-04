"""Tests for authentication middleware."""

import pytest
from unittest.mock import AsyncMock, patch
from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient

from app.middleware.auth import create_auth_dependency, api_key_header


class TestCreateAuthDependency:
    """Tests for create_auth_dependency function."""

    def test_valid_api_key_passes(self):
        """Should pass through with valid API key."""
        app = FastAPI()
        auth_dep = create_auth_dependency("valid-key-123")

        @app.get("/protected")
        async def protected_route(api_key: str = Depends(auth_dep)):
            return {"status": "authenticated", "key": api_key}

        client = TestClient(app)
        response = client.get("/protected", headers={"X-API-Key": "valid-key-123"})

        assert response.status_code == 200
        assert response.json()["status"] == "authenticated"

    def test_missing_api_key_returns_401(self):
        """Should return 401 when API key is missing."""
        app = FastAPI()
        auth_dep = create_auth_dependency("valid-key-123")

        @app.get("/protected")
        async def protected_route(api_key: str = Depends(auth_dep)):
            return {"status": "authenticated"}

        client = TestClient(app)
        response = client.get("/protected")

        assert response.status_code == 401
        assert "Missing API key" in response.json()["detail"]

    def test_invalid_api_key_returns_401(self):
        """Should return 401 when API key is invalid."""
        app = FastAPI()
        auth_dep = create_auth_dependency("valid-key-123")

        @app.get("/protected")
        async def protected_route(api_key: str = Depends(auth_dep)):
            return {"status": "authenticated"}

        client = TestClient(app)
        response = client.get("/protected", headers={"X-API-Key": "wrong-key"})

        assert response.status_code == 401
        assert "Invalid API key" in response.json()["detail"]

    def test_empty_api_key_returns_401(self):
        """Should return 401 when API key is empty string."""
        app = FastAPI()
        auth_dep = create_auth_dependency("valid-key-123")

        @app.get("/protected")
        async def protected_route(api_key: str = Depends(auth_dep)):
            return {"status": "authenticated"}

        client = TestClient(app)
        response = client.get("/protected", headers={"X-API-Key": ""})

        assert response.status_code == 401

    def test_uses_constant_time_comparison(self):
        """Should use secrets.compare_digest for timing-safe comparison."""
        app = FastAPI()
        auth_dep = create_auth_dependency("valid-key-123")

        @app.get("/protected")
        async def protected_route(api_key: str = Depends(auth_dep)):
            return {"status": "authenticated"}

        client = TestClient(app)

        # Mock secrets.compare_digest and verify it's called
        with patch('app.middleware.auth.secrets.compare_digest', return_value=True) as mock_compare:
            response = client.get("/protected", headers={"X-API-Key": "valid-key-123"})

            assert response.status_code == 200
            mock_compare.assert_called_once_with("valid-key-123", "valid-key-123")


class TestApiKeyHeader:
    """Tests for API key header configuration."""

    def test_uses_x_api_key_header(self):
        """Should expect X-API-Key header."""
        assert api_key_header.model.name == "X-API-Key"

    def test_auto_error_is_false(self):
        """Should not auto-error on missing key (we handle it)."""
        assert api_key_header.auto_error is False


class TestAuthIntegration:
    """Integration tests for auth with real app."""

    def _create_fresh_app(self, api_key="test-api-key-12345"):
        """Create a fresh app with clean routers for testing."""
        import os
        from fastapi import Depends, FastAPI
        from app.config import Settings
        from app.middleware.auth import create_auth_dependency
        from app.middleware.logging import RequestLoggingMiddleware
        from app.routes import classify, extract, health

        # Create settings directly
        settings = Settings(
            api_key=api_key,
            runpod_api_key="test-runpod-key",
            runpod_classify_endpoint="classify-endpoint-id",
            runpod_extract_endpoint="extract-endpoint-id"
        )

        app = FastAPI()
        app.add_middleware(RequestLoggingMiddleware)

        auth_dep = Depends(create_auth_dependency(settings.api_key))

        # Include health router (no auth)
        app.include_router(health.router)

        # Include routes with auth dependencies
        app.include_router(
            classify.router,
            dependencies=[auth_dep]
        )
        app.include_router(
            extract.router,
            dependencies=[auth_dep]
        )

        return app, settings

    def test_health_routes_no_auth_required(self, test_client):
        """Health routes should not require authentication."""
        # Mock the RunPod client for /health endpoint
        with patch('app.routes.health.RunPodClient') as MockClient:
            mock_instance = AsyncMock()
            mock_instance.health_check = AsyncMock(return_value={
                "classify": {"status": "healthy"},
                "extract": {"status": "healthy"}
            })
            mock_instance.close = AsyncMock()
            MockClient.return_value = mock_instance

            response = test_client.get("/health")
            assert response.status_code != 401

        response = test_client.get("/ready")
        assert response.status_code == 200

        response = test_client.get("/live")
        assert response.status_code == 200

    def test_classify_requires_auth_missing_key(self, sample_classify_request):
        """POST /v1/classify should return 401 when API key is missing."""
        app, _ = self._create_fresh_app()

        with patch('app.routes.classify.RunPodClient') as MockClient:
            mock_instance = AsyncMock()
            MockClient.return_value = mock_instance

            with TestClient(app) as client:
                response = client.post("/v1/classify", json=sample_classify_request)

            assert response.status_code == 401
            assert "Missing API key" in response.json()["detail"]

    def test_extract_requires_auth_missing_key(self, sample_extract_request):
        """POST /v1/extract should return 401 when API key is missing."""
        app, _ = self._create_fresh_app()

        with patch('app.routes.extract.RunPodClient') as MockClient:
            mock_instance = AsyncMock()
            MockClient.return_value = mock_instance

            with TestClient(app) as client:
                response = client.post("/v1/extract", json=sample_extract_request)

            assert response.status_code == 401
            assert "Missing API key" in response.json()["detail"]

    def test_classify_with_valid_auth_passes(self, sample_classify_request):
        """POST /v1/classify should pass auth with valid API key."""
        from app.services.runpod_client import RunPodResponse

        app, _ = self._create_fresh_app()

        mock_response = RunPodResponse(
            success=True,
            data={"type": "W2", "confidence": 0.9, "reasoning": "test"},
            error=None,
            latency_ms=1000
        )

        with patch('app.routes.classify.RunPodClient') as MockClient:
            mock_instance = AsyncMock()
            mock_instance.classify = AsyncMock(return_value=mock_response)
            mock_instance.close = AsyncMock()
            MockClient.return_value = mock_instance

            with TestClient(app) as client:
                response = client.post(
                    "/v1/classify",
                    json=sample_classify_request,
                    headers={"X-API-Key": "test-api-key-12345"}
                )

            assert response.status_code == 200
            MockClient.assert_called_once()

    def test_classify_with_invalid_auth(self, sample_classify_request):
        """POST /v1/classify should return 401 for invalid API key."""
        app, _ = self._create_fresh_app()

        with patch('app.routes.classify.RunPodClient') as MockClient:
            mock_instance = AsyncMock()
            MockClient.return_value = mock_instance

            with TestClient(app) as client:
                response = client.post(
                    "/v1/classify",
                    json=sample_classify_request,
                    headers={"X-API-Key": "wrong-key-12345"}
                )

            assert response.status_code == 401
            assert "Invalid API key" in response.json()["detail"]

    def test_extract_with_valid_auth_passes(self, sample_extract_request):
        """POST /v1/extract should pass auth with valid API key."""
        from app.services.runpod_client import RunPodResponse

        app, _ = self._create_fresh_app()

        mock_response = RunPodResponse(
            success=True,
            data={"data": {}, "confidence": {}, "doc_type": "W2", "page_count": 1},
            error=None,
            latency_ms=1000
        )

        with patch('app.routes.extract.RunPodClient') as MockClient:
            mock_instance = AsyncMock()
            mock_instance.extract = AsyncMock(return_value=mock_response)
            mock_instance.close = AsyncMock()
            MockClient.return_value = mock_instance

            with TestClient(app) as client:
                response = client.post(
                    "/v1/extract",
                    json=sample_extract_request,
                    headers={"X-API-Key": "test-api-key-12345"}
                )

            assert response.status_code == 200
            MockClient.assert_called_once()
