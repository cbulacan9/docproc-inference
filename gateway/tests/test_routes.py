"""Tests for gateway routes."""

import pytest
from unittest.mock import AsyncMock

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

    def test_health_returns_healthy_when_endpoints_up(self, test_client, mock_runpod_client):
        """GET /health should return healthy when all endpoints healthy."""
        mock_runpod_client.health_check.return_value = {
            "classify": {"status": "healthy", "workers": {"idle": 1}},
            "extract": {"status": "healthy", "workers": {"idle": 1}}
        }

        response = test_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["gateway"] == "healthy"
        assert data["endpoints"]["classify"]["status"] == "healthy"
        assert data["endpoints"]["extract"]["status"] == "healthy"

    def test_health_returns_degraded_when_endpoint_unhealthy(self, test_client, mock_runpod_client):
        """GET /health should return degraded when one endpoint unhealthy."""
        mock_runpod_client.health_check.return_value = {
            "classify": {"status": "healthy", "workers": {"idle": 1}},
            "extract": {"status": "unhealthy", "error": "HTTP 500"}
        }

        response = test_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["endpoints"]["extract"]["status"] == "unhealthy"

    def test_health_returns_degraded_when_endpoint_unreachable(self, test_client, mock_runpod_client):
        """GET /health should return degraded when endpoint unreachable."""
        mock_runpod_client.health_check.return_value = {
            "classify": {"status": "unreachable", "error": "Connection refused"},
            "extract": {"status": "healthy", "workers": {}}
        }

        response = test_client.get("/health")

        data = response.json()
        assert data["status"] == "degraded"

    def test_health_returns_healthy_for_idle_endpoints(self, test_client, mock_runpod_client):
        """GET /health should consider idle endpoints as healthy."""
        mock_runpod_client.health_check.return_value = {
            "classify": {"status": "idle", "workers": {"idle": 0}},
            "extract": {"status": "idle", "workers": {"idle": 0}}
        }

        response = test_client.get("/health")

        data = response.json()
        assert data["status"] == "healthy"


# =============================================================================
# Classify Routes Tests
# =============================================================================

class TestClassifyRoute:
    """Tests for POST /v1/classify endpoint."""

    def test_classify_success(self, test_client, auth_headers, sample_classify_request, mock_runpod_client):
        """Should return classification result on success."""
        mock_runpod_client.classify.return_value = RunPodResponse(
            success=True,
            data={"type": "W2", "confidence": 0.95, "reasoning": "W-2 form detected"},
            error=None,
            latency_ms=1500,
            job_id="job-123"
        )

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

    def test_classify_returns_502_on_failure(self, test_client, auth_headers, sample_classify_request, mock_runpod_client):
        """Should return 502 when RunPod endpoint fails."""
        mock_runpod_client.classify.return_value = RunPodResponse(
            success=False,
            data=None,
            error="Model inference failed",
            latency_ms=0
        )

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

    def test_classify_accepts_custom_prompt(self, test_client, auth_headers, mock_runpod_client):
        """Should accept and pass custom prompt."""
        mock_runpod_client.classify.return_value = RunPodResponse(
            success=True,
            data={"type": "invoice", "confidence": 0.8, "reasoning": ""},
            error=None,
            latency_ms=1000
        )

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
        mock_runpod_client.classify.assert_called_once()
        call_args = mock_runpod_client.classify.call_args
        assert call_args.kwargs.get("prompt") == "Custom classification prompt"

    def test_classify_defaults_type_to_other(self, test_client, auth_headers, sample_classify_request, mock_runpod_client):
        """Should default type to 'other' if not in response."""
        mock_runpod_client.classify.return_value = RunPodResponse(
            success=True,
            data={},  # No type in response
            error=None,
            latency_ms=1000
        )

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

    def test_extract_success(self, test_client, auth_headers, sample_extract_request, mock_runpod_client):
        """Should return extraction result on success."""
        mock_runpod_client.extract.return_value = RunPodResponse(
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

    def test_extract_returns_502_on_failure(self, test_client, auth_headers, sample_extract_request, mock_runpod_client):
        """Should return 502 when RunPod endpoint fails."""
        mock_runpod_client.extract.return_value = RunPodResponse(
            success=False,
            data=None,
            error="Extraction failed",
            latency_ms=0
        )

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

    def test_extract_passes_doc_type_to_client(self, test_client, auth_headers, mock_runpod_client):
        """Should pass doc_type to RunPod client."""
        mock_runpod_client.extract.return_value = RunPodResponse(
            success=True,
            data={"data": {}, "confidence": {}, "doc_type": "bank_statement", "page_count": 1},
            error=None,
            latency_ms=1000
        )

        response = test_client.post(
            "/v1/extract",
            json={
                "image_urls": ["https://example.com/doc.png"],
                "doc_type": "bank_statement"
            },
            headers=auth_headers
        )

        assert response.status_code == 200
        mock_runpod_client.extract.assert_called_once()
        call_args = mock_runpod_client.extract.call_args
        assert call_args.kwargs.get("doc_type") == "bank_statement"

    def test_extract_uses_request_doc_type_as_fallback(self, test_client, auth_headers, sample_extract_request, mock_runpod_client):
        """Should use request doc_type if not in response."""
        mock_runpod_client.extract.return_value = RunPodResponse(
            success=True,
            data={"data": {}, "confidence": {}},  # No doc_type in output
            error=None,
            latency_ms=1000
        )

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

    def test_extra_fields_are_ignored(self, test_client, auth_headers, mock_runpod_client):
        """Should ignore extra fields in request."""
        mock_runpod_client.classify.return_value = RunPodResponse(
            success=True,
            data={"type": "W2", "confidence": 0.9},
            error=None,
            latency_ms=1000
        )

        response = test_client.post(
            "/v1/classify",
            json={
                "image_urls": ["https://example.com/doc.png"],
                "extra_field": "should be ignored"
            },
            headers=auth_headers
        )

        assert response.status_code == 200


# =============================================================================
# Extract Per-Field Confidence Tests
# =============================================================================

class TestExtractPerFieldConfidence:
    """Tests for per-field confidence in /v1/extract endpoint.

    Per Testing Plan:
    - Per-field confidence returned for all doc types
    - Missing fields return confidence 0.0
    - Response includes confidence.overall and confidence.fields
    - Model name is included in response
    """

    def test_extract_returns_per_field_confidence_structure(
        self, test_client, auth_headers, mock_runpod_client
    ):
        """
        Behavior: Response includes confidence.fields dict with per-field scores.
        Assumptions: RunPod returns properly structured confidence data.
        Failure criteria: Missing confidence.fields in response.
        """
        mock_runpod_client.extract.return_value = RunPodResponse(
            success=True,
            data={
                "data": {
                    "header": {"bank_name": "Chase", "beginning_balance": 1000.00}
                },
                "confidence": {
                    "overall": 0.92,
                    "fields": {
                        "header.bank_name": 0.95,
                        "header.beginning_balance": 0.90
                    }
                },
                "doc_type": "bank_statement",
                "page_count": 1,
                "model": "rednote-hilab/dots.ocr"
            },
            error=None,
            latency_ms=2000,
            job_id="job-123"
        )

        response = test_client.post(
            "/v1/extract",
            json={
                "image_urls": ["https://example.com/statement.png"],
                "doc_type": "bank_statement"
            },
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()

        # Verify confidence structure
        assert "confidence" in data
        assert "overall" in data["confidence"]
        assert "fields" in data["confidence"]
        assert isinstance(data["confidence"]["fields"], dict)

        # Verify field confidences present
        assert data["confidence"]["overall"] == 0.92
        assert data["confidence"]["fields"]["header.bank_name"] == 0.95
        assert data["confidence"]["fields"]["header.beginning_balance"] == 0.90

    def test_extract_returns_model_name(
        self, test_client, auth_headers, mock_runpod_client
    ):
        """
        Behavior: Response includes model name used for extraction.
        Assumptions: RunPod returns model in output.
        Failure criteria: Missing model field in response.
        """
        mock_runpod_client.extract.return_value = RunPodResponse(
            success=True,
            data={
                "data": {},
                "confidence": {"overall": 0.85, "fields": {}},
                "doc_type": "W2",
                "page_count": 1,
                "model": "rednote-hilab/dots.ocr"
            },
            error=None,
            latency_ms=1500
        )

        response = test_client.post(
            "/v1/extract",
            json={
                "image_urls": ["https://example.com/w2.png"],
                "doc_type": "W2"
            },
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert "model" in data
        assert data["model"] == "rednote-hilab/dots.ocr"

    def test_extract_bank_statement_returns_field_confidences(
        self, test_client, auth_headers, mock_runpod_client
    ):
        """
        Behavior: Bank statement extraction returns confidence for expected fields.
        Assumptions: Handler properly extracts bank statement fields.
        Failure criteria: Missing expected field confidence keys.
        """
        mock_runpod_client.extract.return_value = RunPodResponse(
            success=True,
            data={
                "data": {
                    "header": {
                        "bank_name": "Chase",
                        "account_number": "****1234",
                        "beginning_balance": 5000.00,
                        "ending_balance": 4500.00
                    },
                    "transactions": [
                        {"date": "01/15", "description": "ATM", "amount": -500.00}
                    ]
                },
                "confidence": {
                    "overall": 0.89,
                    "fields": {
                        "header.bank_name": 0.95,
                        "header.account_number": 0.90,
                        "header.beginning_balance": 0.92,
                        "header.ending_balance": 0.92,
                        "transactions": [
                            {"date": 0.88, "description": 0.85, "amount": 0.90}
                        ]
                    }
                },
                "doc_type": "bank_statement",
                "page_count": 1,
                "model": "dots-ocr"
            },
            error=None,
            latency_ms=2500
        )

        response = test_client.post(
            "/v1/extract",
            json={
                "image_urls": ["https://example.com/statement.png"],
                "doc_type": "bank_statement"
            },
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()

        fields = data["confidence"]["fields"]
        assert "header.bank_name" in fields
        assert "header.beginning_balance" in fields
        assert "header.ending_balance" in fields
        assert "transactions" in fields
        assert isinstance(fields["transactions"], list)

    def test_extract_w2_returns_field_confidences(
        self, test_client, auth_headers, mock_runpod_client
    ):
        """
        Behavior: W2 extraction returns confidence for expected fields.
        Assumptions: Handler properly extracts W2 fields.
        Failure criteria: Missing expected field confidence keys.
        """
        mock_runpod_client.extract.return_value = RunPodResponse(
            success=True,
            data={
                "data": {
                    "employee": {"ssn": "XXX-XX-1234", "name": "John Doe"},
                    "employer": {"ein": "12-3456789"},
                    "boxes": {"box1_wages": 75000.00, "box2_federal_withheld": 12000.00}
                },
                "confidence": {
                    "overall": 0.91,
                    "fields": {
                        "employee.ssn": 0.95,
                        "employee.name": 0.0,
                        "employer.ein": 0.93,
                        "boxes.box1_wages": 0.90,
                        "boxes.box2_federal_withheld": 0.88
                    }
                },
                "doc_type": "W2",
                "page_count": 1,
                "model": "dots-ocr"
            },
            error=None,
            latency_ms=2000
        )

        response = test_client.post(
            "/v1/extract",
            json={
                "image_urls": ["https://example.com/w2.png"],
                "doc_type": "W2"
            },
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()

        fields = data["confidence"]["fields"]
        assert "employee.ssn" in fields
        assert "employer.ein" in fields
        assert "boxes.box1_wages" in fields
        assert "boxes.box2_federal_withheld" in fields

    def test_extract_missing_field_returns_zero_confidence(
        self, test_client, auth_headers, mock_runpod_client
    ):
        """
        Behavior: Missing/null fields should have confidence 0.0.
        Assumptions: Handler sets 0.0 confidence for fields it cannot extract.
        Failure criteria: Missing field has non-zero confidence.
        """
        mock_runpod_client.extract.return_value = RunPodResponse(
            success=True,
            data={
                "data": {
                    "employee": {"ssn": "XXX-XX-1234", "name": None},  # name is null
                    "employer": {"ein": None},  # EIN is null
                    "boxes": {"box1_wages": 75000.00}
                },
                "confidence": {
                    "overall": 0.75,
                    "fields": {
                        "employee.ssn": 0.95,
                        "employee.name": 0.0,  # 0.0 for null field
                        "employer.ein": 0.0,   # 0.0 for null field
                        "boxes.box1_wages": 0.90
                    }
                },
                "doc_type": "W2",
                "page_count": 1,
                "model": "dots-ocr"
            },
            error=None,
            latency_ms=1800
        )

        response = test_client.post(
            "/v1/extract",
            json={
                "image_urls": ["https://example.com/w2.png"],
                "doc_type": "W2"
            },
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()

        # Verify null fields have 0.0 confidence
        fields = data["confidence"]["fields"]
        assert fields["employee.name"] == 0.0
        assert fields["employer.ein"] == 0.0

        # Verify extracted fields have non-zero confidence
        assert fields["employee.ssn"] > 0
        assert fields["boxes.box1_wages"] > 0

    def test_extract_1099_returns_field_confidences(
        self, test_client, auth_headers, mock_runpod_client
    ):
        """
        Behavior: 1099 extraction returns confidence for expected fields.
        Assumptions: Handler properly extracts 1099 fields.
        Failure criteria: Missing expected field confidence keys.
        """
        mock_runpod_client.extract.return_value = RunPodResponse(
            success=True,
            data={
                "data": {
                    "recipient": {"tin": "XXX-XX-5678"},
                    "payer": {"tin": "12-7654321"},
                    "boxes": {"box1": 1234.56},
                    "form_type": "1099-INT"
                },
                "confidence": {
                    "overall": 0.88,
                    "fields": {
                        "recipient.tin": 0.92,
                        "payer.tin": 0.90,
                        "boxes.box1": 0.85,
                        "form_type": 0.95
                    }
                },
                "doc_type": "1099-INT",
                "page_count": 1,
                "model": "dots-ocr"
            },
            error=None,
            latency_ms=1600
        )

        response = test_client.post(
            "/v1/extract",
            json={
                "image_urls": ["https://example.com/1099.png"],
                "doc_type": "1099-INT"
            },
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()

        fields = data["confidence"]["fields"]
        assert "recipient.tin" in fields
        assert "form_type" in fields

    def test_extract_overall_confidence_within_valid_range(
        self, test_client, auth_headers, mock_runpod_client
    ):
        """
        Behavior: Overall confidence should be between 0.0 and 1.0.
        Assumptions: Confidence calculation produces valid range.
        Failure criteria: Overall confidence outside [0.0, 1.0].
        """
        mock_runpod_client.extract.return_value = RunPodResponse(
            success=True,
            data={
                "data": {},
                "confidence": {"overall": 0.87, "fields": {}},
                "doc_type": "other",
                "page_count": 1,
                "model": "dots-ocr"
            },
            error=None,
            latency_ms=1000
        )

        response = test_client.post(
            "/v1/extract",
            json={
                "image_urls": ["https://example.com/doc.png"],
                "doc_type": "other"
            },
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()

        overall = data["confidence"]["overall"]
        assert 0.0 <= overall <= 1.0

    def test_extract_defaults_confidence_when_missing(
        self, test_client, auth_headers, mock_runpod_client
    ):
        """
        Behavior: Default confidence values when not provided by handler.
        Assumptions: Endpoint provides sensible defaults.
        Failure criteria: Crash or missing confidence fields.
        """
        mock_runpod_client.extract.return_value = RunPodResponse(
            success=True,
            data={
                "data": {"some": "data"},
                "confidence": {},  # Empty confidence dict
                "doc_type": "other",
                "page_count": 1
            },
            error=None,
            latency_ms=1000
        )

        response = test_client.post(
            "/v1/extract",
            json={
                "image_urls": ["https://example.com/doc.png"],
                "doc_type": "other"
            },
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()

        # Should have default overall confidence
        assert "confidence" in data
        assert "overall" in data["confidence"]
        assert data["confidence"]["overall"] == 0.5  # Default

    def test_extract_multi_page_returns_combined_confidence(
        self, test_client, auth_headers, mock_runpod_client
    ):
        """
        Behavior: Multi-page documents return combined confidence.
        Assumptions: Handler aggregates confidence across pages.
        Failure criteria: Confidence doesn't reflect multi-page processing.
        """
        mock_runpod_client.extract.return_value = RunPodResponse(
            success=True,
            data={
                "data": {
                    "header": {"bank_name": "Chase"},
                    "transactions": [
                        {"amount": 100.00},
                        {"amount": 200.00},
                        {"amount": 300.00}
                    ]
                },
                "confidence": {
                    "overall": 0.88,
                    "fields": {
                        "header.bank_name": 0.95,
                        "transactions": [
                            {"amount": 0.90},
                            {"amount": 0.85},
                            {"amount": 0.88}
                        ]
                    }
                },
                "doc_type": "bank_statement",
                "page_count": 3,  # Multi-page
                "model": "dots-ocr"
            },
            error=None,
            latency_ms=4500
        )

        response = test_client.post(
            "/v1/extract",
            json={
                "image_urls": [
                    "https://example.com/page1.png",
                    "https://example.com/page2.png",
                    "https://example.com/page3.png"
                ],
                "doc_type": "bank_statement"
            },
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()

        assert data["page_count"] == 3
        assert len(data["confidence"]["fields"]["transactions"]) == 3

    def test_extract_passes_custom_prompt(
        self, test_client, auth_headers, mock_runpod_client
    ):
        """
        Behavior: Custom prompt is passed to extraction handler.
        Assumptions: Handler accepts optional prompt parameter.
        Failure criteria: Prompt not passed to RunPod client.
        """
        mock_runpod_client.extract.return_value = RunPodResponse(
            success=True,
            data={
                "data": {},
                "confidence": {"overall": 0.8, "fields": {}},
                "doc_type": "other",
                "page_count": 1,
                "model": "dots-ocr"
            },
            error=None,
            latency_ms=1000
        )

        response = test_client.post(
            "/v1/extract",
            json={
                "image_urls": ["https://example.com/doc.png"],
                "doc_type": "invoice",
                "prompt": "Extract invoice number and total amount"
            },
            headers=auth_headers
        )

        assert response.status_code == 200

        # Verify prompt was passed to client
        mock_runpod_client.extract.assert_called_once()
        call_args = mock_runpod_client.extract.call_args
        assert call_args.kwargs.get("prompt") == "Extract invoice number and total amount"

    def test_extract_response_includes_latency(
        self, test_client, auth_headers, mock_runpod_client
    ):
        """
        Behavior: Response includes latency_ms from inference.
        Assumptions: Handler tracks and returns latency.
        Failure criteria: Missing or incorrect latency_ms.
        """
        mock_runpod_client.extract.return_value = RunPodResponse(
            success=True,
            data={
                "data": {},
                "confidence": {"overall": 0.85, "fields": {}},
                "doc_type": "W2",
                "page_count": 1,
                "model": "dots-ocr"
            },
            error=None,
            latency_ms=3456,
            job_id="job-789"
        )

        response = test_client.post(
            "/v1/extract",
            json={
                "image_urls": ["https://example.com/w2.png"],
                "doc_type": "W2"
            },
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()

        assert "latency_ms" in data
        assert data["latency_ms"] == 3456
