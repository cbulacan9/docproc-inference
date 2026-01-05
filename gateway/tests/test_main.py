"""Tests for main application module."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient


class TestCreateApp:
    """Tests for create_app factory function."""

    def test_create_app_returns_fastapi_instance(self, mock_env_vars):
        """Should return a FastAPI application instance."""
        from app.config import get_settings
        get_settings.cache_clear()

        with patch('app.main.RunPodClient'):
            from app.main import create_app
            app = create_app()

            assert isinstance(app, FastAPI)

    def test_create_app_has_correct_metadata(self, mock_env_vars):
        """Should configure app with correct title, version, and docs."""
        from app.config import get_settings
        get_settings.cache_clear()

        with patch('app.main.RunPodClient'):
            from app.main import create_app
            app = create_app()

            assert app.title == "Document Processing Inference Gateway"
            assert app.version == "1.0.0"
            assert app.docs_url == "/docs"
            assert app.redoc_url == "/redoc"

    def test_create_app_registers_health_routes(self, mock_env_vars, mock_runpod_client):
        """Should register health check routes without auth."""
        from app.config import get_settings
        get_settings.cache_clear()

        with patch('app.main.RunPodClient', return_value=mock_runpod_client):
            from app.main import create_app
            app = create_app()

            with TestClient(app) as client:
                # Health routes should be accessible without auth
                response = client.get("/ready")
                assert response.status_code == 200

                response = client.get("/live")
                assert response.status_code == 200

    def test_create_app_registers_classify_route(self, mock_env_vars, mock_runpod_client):
        """Should register /v1/classify route."""
        from app.config import get_settings
        get_settings.cache_clear()

        with patch('app.main.RunPodClient', return_value=mock_runpod_client):
            from app.main import create_app
            app = create_app()

            # Check route exists in app routes
            route_paths = [route.path for route in app.routes]
            assert "/v1/classify" in route_paths

    def test_create_app_registers_extract_route(self, mock_env_vars, mock_runpod_client):
        """Should register /v1/extract route."""
        from app.config import get_settings
        get_settings.cache_clear()

        with patch('app.main.RunPodClient', return_value=mock_runpod_client):
            from app.main import create_app
            app = create_app()

            # Check route exists in app routes
            route_paths = [route.path for route in app.routes]
            assert "/v1/extract" in route_paths


class TestCORSMiddleware:
    """Tests for CORS middleware configuration."""

    def test_cors_allows_all_origins(self, mock_env_vars, mock_runpod_client):
        """Should allow requests from any origin (dev config)."""
        from app.config import get_settings
        get_settings.cache_clear()

        with patch('app.main.RunPodClient', return_value=mock_runpod_client):
            from app.main import create_app
            app = create_app()

            with TestClient(app) as client:
                # Preflight request with Origin header
                response = client.options(
                    "/ready",
                    headers={
                        "Origin": "http://example.com",
                        "Access-Control-Request-Method": "GET"
                    }
                )

                # CORS headers should be present
                # When allow_credentials=True, the origin is echoed back instead of "*"
                assert response.headers.get("access-control-allow-origin") == "http://example.com"

    def test_cors_allows_credentials(self, mock_env_vars, mock_runpod_client):
        """Should allow credentials in CORS requests."""
        from app.config import get_settings
        get_settings.cache_clear()

        with patch('app.main.RunPodClient', return_value=mock_runpod_client):
            from app.main import create_app
            app = create_app()

            with TestClient(app) as client:
                response = client.options(
                    "/ready",
                    headers={
                        "Origin": "http://example.com",
                        "Access-Control-Request-Method": "GET"
                    }
                )

                assert response.headers.get("access-control-allow-credentials") == "true"

    def test_cors_allows_all_methods(self, mock_env_vars, mock_runpod_client):
        """Should allow all HTTP methods."""
        from app.config import get_settings
        get_settings.cache_clear()

        with patch('app.main.RunPodClient', return_value=mock_runpod_client):
            from app.main import create_app
            app = create_app()

            with TestClient(app) as client:
                response = client.options(
                    "/ready",
                    headers={
                        "Origin": "http://example.com",
                        "Access-Control-Request-Method": "POST"
                    }
                )

                allowed_methods = response.headers.get("access-control-allow-methods", "")
                # Should allow POST (and typically others)
                assert "POST" in allowed_methods or "*" in allowed_methods


class TestLifespan:
    """Tests for application lifespan management."""

    def test_lifespan_creates_runpod_client(self, mock_env_vars):
        """Should create RunPod client during startup."""
        from app.config import get_settings
        get_settings.cache_clear()

        mock_client = AsyncMock()
        mock_client.close = AsyncMock()

        with patch('app.main.RunPodClient', return_value=mock_client) as MockRunPod:
            from app.main import create_app
            app = create_app()

            with TestClient(app):
                # Client should be created with correct parameters
                MockRunPod.assert_called_once()
                call_kwargs = MockRunPod.call_args.kwargs
                assert "api_key" in call_kwargs
                assert "classify_endpoint" in call_kwargs
                assert "extract_endpoint" in call_kwargs

    def test_lifespan_closes_client_on_shutdown(self, mock_env_vars):
        """Should close RunPod client during shutdown."""
        from app.config import get_settings
        get_settings.cache_clear()

        mock_client = AsyncMock()
        mock_client.close = AsyncMock()

        with patch('app.main.RunPodClient', return_value=mock_client):
            from app.main import create_app
            app = create_app()

            with TestClient(app):
                pass  # Enter and exit context

            # Client should be closed after context exit
            mock_client.close.assert_called_once()

    def test_lifespan_sets_runpod_client_on_app_state(self, mock_env_vars):
        """Should set runpod_client on app.state."""
        from app.config import get_settings
        get_settings.cache_clear()

        mock_client = AsyncMock()
        mock_client.close = AsyncMock()

        with patch('app.main.RunPodClient', return_value=mock_client):
            from app.main import create_app
            app = create_app()

            with TestClient(app):
                # Client should be accessible on app.state
                assert app.state.runpod_client is mock_client


class TestAppRouteProtection:
    """Tests for route authentication requirements."""

    def test_classify_requires_auth(self, mock_env_vars, mock_runpod_client):
        """POST /v1/classify should require authentication."""
        from app.config import get_settings
        get_settings.cache_clear()

        with patch('app.main.RunPodClient', return_value=mock_runpod_client):
            from app.main import create_app
            app = create_app()

            with TestClient(app) as client:
                response = client.post(
                    "/v1/classify",
                    json={"image_urls": ["https://example.com/doc.png"]}
                )

                assert response.status_code == 401

    def test_extract_requires_auth(self, mock_env_vars, mock_runpod_client):
        """POST /v1/extract should require authentication."""
        from app.config import get_settings
        get_settings.cache_clear()

        with patch('app.main.RunPodClient', return_value=mock_runpod_client):
            from app.main import create_app
            app = create_app()

            with TestClient(app) as client:
                response = client.post(
                    "/v1/extract",
                    json={"image_urls": ["https://example.com/doc.png"], "doc_type": "W2"}
                )

                assert response.status_code == 401

    def test_health_does_not_require_auth(self, mock_env_vars, mock_runpod_client):
        """GET /health should not require authentication."""
        from app.config import get_settings
        get_settings.cache_clear()

        with patch('app.main.RunPodClient', return_value=mock_runpod_client):
            from app.main import create_app
            app = create_app()

            with TestClient(app) as client:
                response = client.get("/health")

                # Should not be 401
                assert response.status_code != 401
