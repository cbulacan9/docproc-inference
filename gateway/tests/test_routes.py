"""Tests for gateway routes."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.runpod_client import RunPodResponse


# =============================================================================
# Health Routes Tests
# =============================================================================

class TestHealthRoutes:
    """Tests for health check endpoints."""

    def test_ready_returns_200(self, test_client):
        """GET /ready should return 200 with status."""
        response = test_client.get("/ready")

        assert response.status_code == 200
        assert response.json()["status"] == "ready"

    def test_live_returns_200(self, test_client):
        """GET /live should return 200 with status."""
        response = test_client.get("/live")

        assert response.status_code == 200
        assert response.json()["status"] == "live"

    def test_health_returns_healthy_when_endpoints_up(self, test_client):
        """GET /health should return healthy when all endpoints healthy."""
        mock_health_result = {
            "classify": {"status": "healthy", "workers": {"idle": 1}},
            "extract": {"status": "healthy", "workers": {"idle": 1}}
        }

        with patch('app.routes.health.RunPodClient') as MockClient:
            mock_instance = AsyncMock()
            mock_instance.health_check = AsyncMock(return_value=mock_health_result)
            mock_instance.close = AsyncMock()
            MockClient.return_value = mock_instance

            response = test_client.get("/health")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["gateway"] == "healthy"
            assert data["endpoints"]["classify"]["status"] == "healthy"
            assert data["endpoints"]["extract"]["status"] == "healthy"

    def test_health_returns_degraded_when_endpoint_unhealthy(self, test_client):
        """GET /health should return degraded when one endpoint unhealthy."""
        mock_health_result = {
            "classify": {"status": "healthy", "workers": {"idle": 1}},
            "extract": {"status": "unhealthy", "error": "HTTP 500"}
        }

        with patch('app.routes.health.RunPodClient') as MockClient:
            mock_instance = AsyncMock()
            mock_instance.health_check = AsyncMock(return_value=mock_health_result)
            mock_instance.close = AsyncMock()
            MockClient.return_value = mock_instance

            response = test_client.get("/health")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "degraded"
            assert data["endpoints"]["extract"]["status"] == "unhealthy"

    def test_health_returns_degraded_when_endpoint_unreachable(self, test_client):
        """GET /health should return degraded when endpoint unreachable."""
        mock_health_result = {
            "classify": {"status": "unreachable", "error": "Connection refused"},
            "extract": {"status": "healthy", "workers": {}}
        }

        with patch('app.routes.health.RunPodClient') as MockClient:
            mock_instance = AsyncMock()
            mock_instance.health_check = AsyncMock(return_value=mock_health_result)
            mock_instance.close = AsyncMock()
            MockClient.return_value = mock_instance

            response = test_client.get("/health")

            data = response.json()
            assert data["status"] == "degraded"

    def test_health_returns_healthy_for_idle_endpoints(self, test_client):
        """GET /health should consider idle endpoints as healthy."""
        mock_health_result = {
            "classify": {"status": "idle", "workers": {"idle": 0}},
            "extract": {"status": "idle", "workers": {"idle": 0}}
        }

        with patch('app.routes.health.RunPodClient') as MockClient:
            mock_instance = AsyncMock()
            mock_instance.health_check = AsyncMock(return_value=mock_health_result)
            mock_instance.close = AsyncMock()
            MockClient.return_value = mock_instance

            response = test_client.get("/health")

            data = response.json()
            assert data["status"] == "healthy"


# =============================================================================
# Classify Routes Tests
# =============================================================================

class TestClassifyRoute:
    """Tests for POST /v1/classify endpoint."""

    def test_classify_success(self, test_client, auth_headers, sample_classify_request):
        """Should return classification result on success."""
        mock_response = RunPodResponse(
            success=True,
            data={"type": "W2", "confidence": 0.95, "reasoning": "W-2 form detected"},
            error=None,
            latency_ms=1500,
            job_id="job-123"
        )

        with patch('app.routes.classify.RunPodClient') as MockClient:
            mock_instance = AsyncMock()
            mock_instance.classify = AsyncMock(return_value=mock_response)
            mock_instance.close = AsyncMock()
            MockClient.return_value = mock_instance

            response = test_client.post(
                "/v1/classify",
                json=sample_classify_request,
                headers=auth_headers
            )

            assert response.status_code == 200
            data = response.json()
            assert data["type"] == "W2"
            assert data["confidence"] == 0.95
            assert data["reasoning"] == "W-2 form detected"
            assert data["latency_ms"] == 1500

    def test_classify_returns_502_on_failure(self, test_client, auth_headers, sample_classify_request):
        """Should return 502 when RunPod endpoint fails."""
        mock_response = RunPodResponse(
            success=False,
            data=None,
            error="Model inference failed",
            latency_ms=0
        )

        with patch('app.routes.classify.RunPodClient') as MockClient:
            mock_instance = AsyncMock()
            mock_instance.classify = AsyncMock(return_value=mock_response)
            mock_instance.close = AsyncMock()
            MockClient.return_value = mock_instance

            response = test_client.post(
                "/v1/classify",
                json=sample_classify_request,
                headers=auth_headers
            )

            assert response.status_code == 502
            assert "Inference failed" in response.json()["detail"]

    def test_classify_validates_empty_image_urls(self, test_client, auth_headers):
        """Should return 422 when image_urls is empty."""
        response = test_client.post(
            "/v1/classify",
            json={"image_urls": []},
            headers=auth_headers
        )

        assert response.status_code == 422

    def test_classify_validates_missing_image_urls(self, test_client, auth_headers):
        """Should return 422 when image_urls is missing."""
        response = test_client.post(
            "/v1/classify",
            json={},
            headers=auth_headers
        )

        assert response.status_code == 422

    def test_classify_accepts_custom_prompt(self, test_client, auth_headers):
        """Should accept and pass custom prompt."""
        mock_response = RunPodResponse(
            success=True,
            data={"type": "invoice", "confidence": 0.8, "reasoning": ""},
            error=None,
            latency_ms=1000
        )

        with patch('app.routes.classify.RunPodClient') as MockClient:
            mock_instance = AsyncMock()
            mock_instance.classify = AsyncMock(return_value=mock_response)
            mock_instance.close = AsyncMock()
            MockClient.return_value = mock_instance

            response = test_client.post(
                "/v1/classify",
                json={
                    "image_urls": ["https://example.com/doc.png"],
                    "prompt": "Custom classification prompt"
                },
                headers=auth_headers
            )

            assert response.status_code == 200

            # Verify prompt was passed
            mock_instance.classify.assert_called_once()
            call_args = mock_instance.classify.call_args
            assert call_args.kwargs.get("prompt") == "Custom classification prompt"

    def test_classify_defaults_type_to_other(self, test_client, auth_headers, sample_classify_request):
        """Should default type to 'other' if not in response."""
        mock_response = RunPodResponse(
            success=True,
            data={},  # No type in response
            error=None,
            latency_ms=1000
        )

        with patch('app.routes.classify.RunPodClient') as MockClient:
            mock_instance = AsyncMock()
            mock_instance.classify = AsyncMock(return_value=mock_response)
            mock_instance.close = AsyncMock()
            MockClient.return_value = mock_instance

            response = test_client.post(
                "/v1/classify",
                json=sample_classify_request,
                headers=auth_headers
            )

            assert response.status_code == 200
            assert response.json()["type"] == "other"


# =============================================================================
# Extract Routes Tests
# =============================================================================

class TestExtractRoute:
    """Tests for POST /v1/extract endpoint."""

    def test_extract_success(self, test_client, auth_headers, sample_extract_request):
        """Should return extraction result on success."""
        mock_response = RunPodResponse(
            success=True,
            data={
                "data": {"employee": {"name": "John Doe"}},
                "confidence": {"overall": 0.92},
                "doc_type": "W2",
                "page_count": 1
            },
            error=None,
            latency_ms=2500,
            job_id="job-456"
        )

        with patch('app.routes.extract.RunPodClient') as MockClient:
            mock_instance = AsyncMock()
            mock_instance.extract = AsyncMock(return_value=mock_response)
            mock_instance.close = AsyncMock()
            MockClient.return_value = mock_instance

            response = test_client.post(
                "/v1/extract",
                json=sample_extract_request,
                headers=auth_headers
            )

            assert response.status_code == 200
            data = response.json()
            assert data["data"]["employee"]["name"] == "John Doe"
            assert data["confidence"]["overall"] == 0.92
            assert data["doc_type"] == "W2"
            assert data["page_count"] == 1
            assert data["latency_ms"] == 2500

    def test_extract_returns_502_on_failure(self, test_client, auth_headers, sample_extract_request):
        """Should return 502 when RunPod endpoint fails."""
        mock_response = RunPodResponse(
            success=False,
            data=None,
            error="Extraction failed",
            latency_ms=0
        )

        with patch('app.routes.extract.RunPodClient') as MockClient:
            mock_instance = AsyncMock()
            mock_instance.extract = AsyncMock(return_value=mock_response)
            mock_instance.close = AsyncMock()
            MockClient.return_value = mock_instance

            response = test_client.post(
                "/v1/extract",
                json=sample_extract_request,
                headers=auth_headers
            )

            assert response.status_code == 502
            assert "Inference failed" in response.json()["detail"]

    def test_extract_validates_missing_doc_type(self, test_client, auth_headers):
        """Should return 422 when doc_type is missing."""
        response = test_client.post(
            "/v1/extract",
            json={"image_urls": ["https://example.com/doc.png"]},
            headers=auth_headers
        )

        assert response.status_code == 422

    def test_extract_validates_empty_image_urls(self, test_client, auth_headers):
        """Should return 422 when image_urls is empty."""
        response = test_client.post(
            "/v1/extract",
            json={"image_urls": [], "doc_type": "W2"},
            headers=auth_headers
        )

        assert response.status_code == 422

    def test_extract_passes_doc_type_to_client(self, test_client, auth_headers):
        """Should pass doc_type to RunPod client."""
        mock_response = RunPodResponse(
            success=True,
            data={"data": {}, "confidence": {}, "doc_type": "bank_statement", "page_count": 1},
            error=None,
            latency_ms=1000
        )

        with patch('app.routes.extract.RunPodClient') as MockClient:
            mock_instance = AsyncMock()
            mock_instance.extract = AsyncMock(return_value=mock_response)
            mock_instance.close = AsyncMock()
            MockClient.return_value = mock_instance

            response = test_client.post(
                "/v1/extract",
                json={
                    "image_urls": ["https://example.com/doc.png"],
                    "doc_type": "bank_statement"
                },
                headers=auth_headers
            )

            assert response.status_code == 200
            mock_instance.extract.assert_called_once()
            call_args = mock_instance.extract.call_args
            assert call_args.kwargs.get("doc_type") == "bank_statement"

    def test_extract_uses_request_doc_type_as_fallback(self, test_client, auth_headers, sample_extract_request):
        """Should use request doc_type if not in response."""
        mock_response = RunPodResponse(
            success=True,
            data={"data": {}, "confidence": {}},  # No doc_type in output
            error=None,
            latency_ms=1000
        )

        with patch('app.routes.extract.RunPodClient') as MockClient:
            mock_instance = AsyncMock()
            mock_instance.extract = AsyncMock(return_value=mock_response)
            mock_instance.close = AsyncMock()
            MockClient.return_value = mock_instance

            response = test_client.post(
                "/v1/extract",
                json=sample_extract_request,
                headers=auth_headers
            )

            assert response.status_code == 200
            # Should use the doc_type from request as fallback
            assert response.json()["doc_type"] == sample_extract_request["doc_type"]


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for error handling across routes."""

    def test_invalid_json_returns_422(self, test_client, auth_headers):
        """Should return 422 for invalid JSON body."""
        response = test_client.post(
            "/v1/classify",
            content="not json",
            headers={**auth_headers, "Content-Type": "application/json"}
        )

        assert response.status_code == 422

    def test_wrong_content_type_returns_422(self, test_client, auth_headers):
        """Should return 422 for wrong content type."""
        response = test_client.post(
            "/v1/classify",
            content="image_urls=test",
            headers={**auth_headers, "Content-Type": "application/x-www-form-urlencoded"}
        )

        assert response.status_code == 422

    def test_extra_fields_are_ignored(self, test_client, auth_headers):
        """Should ignore extra fields in request."""
        mock_response = RunPodResponse(
            success=True,
            data={"type": "W2", "confidence": 0.9},
            error=None,
            latency_ms=1000
        )

        with patch('app.routes.classify.RunPodClient') as MockClient:
            mock_instance = AsyncMock()
            mock_instance.classify = AsyncMock(return_value=mock_response)
            mock_instance.close = AsyncMock()
            MockClient.return_value = mock_instance

            response = test_client.post(
                "/v1/classify",
                json={
                    "image_urls": ["https://example.com/doc.png"],
                    "extra_field": "should be ignored"
                },
                headers=auth_headers
            )

            assert response.status_code == 200
