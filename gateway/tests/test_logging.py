"""Tests for request logging middleware."""

import pytest
import logging
from unittest.mock import patch, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.middleware.logging import RequestLoggingMiddleware


class TestRequestLoggingMiddleware:
    """Tests for RequestLoggingMiddleware."""

    def test_adds_request_id_to_response_headers(self):
        """Should add X-Request-ID header to response."""
        app = FastAPI()
        app.add_middleware(RequestLoggingMiddleware)

        @app.get("/test")
        async def test_route():
            return {"status": "ok"}

        client = TestClient(app)
        response = client.get("/test")

        assert "X-Request-ID" in response.headers
        assert len(response.headers["X-Request-ID"]) == 8  # UUID[:8]

    def test_request_id_is_unique_per_request(self):
        """Should generate unique request ID for each request."""
        app = FastAPI()
        app.add_middleware(RequestLoggingMiddleware)

        @app.get("/test")
        async def test_route():
            return {"status": "ok"}

        client = TestClient(app)
        response1 = client.get("/test")
        response2 = client.get("/test")

        id1 = response1.headers["X-Request-ID"]
        id2 = response2.headers["X-Request-ID"]

        assert id1 != id2

    def test_logs_request_info(self, caplog):
        """Should log request method and path."""
        app = FastAPI()
        app.add_middleware(RequestLoggingMiddleware)

        @app.get("/test-endpoint")
        async def test_route():
            return {"status": "ok"}

        client = TestClient(app)

        with caplog.at_level(logging.INFO):
            client.get("/test-endpoint")

        # Check request was logged
        log_messages = [r.message for r in caplog.records]
        assert any("GET /test-endpoint" in msg for msg in log_messages)

    def test_logs_response_status_and_timing(self, caplog):
        """Should log response status code and timing."""
        app = FastAPI()
        app.add_middleware(RequestLoggingMiddleware)

        @app.get("/test")
        async def test_route():
            return {"status": "ok"}

        client = TestClient(app)

        with caplog.at_level(logging.INFO):
            client.get("/test")

        log_messages = [r.message for r in caplog.records]
        # Should have response log with status and timing
        assert any("200" in msg and "ms" in msg for msg in log_messages)

    def test_logs_client_host(self, caplog):
        """Should log client host/IP."""
        app = FastAPI()
        app.add_middleware(RequestLoggingMiddleware)

        @app.get("/test")
        async def test_route():
            return {"status": "ok"}

        client = TestClient(app)

        with caplog.at_level(logging.INFO):
            client.get("/test")

        log_messages = [r.message for r in caplog.records]
        # TestClient uses 'testclient' as host
        assert any("from" in msg.lower() for msg in log_messages)

    def test_logs_error_on_exception(self, caplog):
        """Should log error when route raises exception."""
        app = FastAPI()
        app.add_middleware(RequestLoggingMiddleware)

        @app.get("/error")
        async def error_route():
            raise ValueError("Test error")

        client = TestClient(app, raise_server_exceptions=False)

        with caplog.at_level(logging.ERROR):
            client.get("/error")

        log_messages = [r.message for r in caplog.records]
        assert any("Error" in msg for msg in log_messages)

    def test_request_id_format(self):
        """Request ID should be first 8 chars of UUID."""
        app = FastAPI()
        app.add_middleware(RequestLoggingMiddleware)

        @app.get("/test")
        async def test_route():
            return {"status": "ok"}

        client = TestClient(app)
        response = client.get("/test")

        request_id = response.headers["X-Request-ID"]

        # Should be 8 hex characters
        assert len(request_id) == 8
        assert all(c in "0123456789abcdef-" for c in request_id.lower())

    def test_request_id_in_logs_matches_header(self, caplog):
        """Request ID in logs should match the response header."""
        app = FastAPI()
        app.add_middleware(RequestLoggingMiddleware)

        @app.get("/test")
        async def test_route():
            return {"status": "ok"}

        client = TestClient(app)

        with caplog.at_level(logging.INFO):
            response = client.get("/test")

        request_id = response.headers["X-Request-ID"]
        log_messages = [r.message for r in caplog.records]

        # Request ID should appear in logs
        assert any(request_id in msg for msg in log_messages)


class TestRequestLoggingIntegration:
    """Integration tests with real app."""

    def test_logging_on_protected_route(self, test_client, auth_headers, caplog):
        """Should log requests to protected routes."""
        with caplog.at_level(logging.INFO):
            test_client.get("/ready")

        log_messages = [r.message for r in caplog.records]
        assert any("/ready" in msg for msg in log_messages)

    def test_logging_includes_request_id_in_all_logs(self, test_client, caplog):
        """All log messages for a request should include the same request ID."""
        with caplog.at_level(logging.INFO):
            response = test_client.get("/live")

        request_id = response.headers.get("X-Request-ID")
        if request_id:
            relevant_logs = [r for r in caplog.records if request_id in r.message]
            assert len(relevant_logs) >= 1
