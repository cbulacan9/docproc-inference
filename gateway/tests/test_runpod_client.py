"""Tests for RunPod client service."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx

from app.services.runpod_client import RunPodClient, RunPodResponse


class TestRunPodClientInit:
    """Tests for RunPodClient initialization."""

    def test_creates_client_with_correct_headers(self):
        """Should create HTTP client with auth headers."""
        client = RunPodClient(
            api_key="test-key",
            classify_endpoint="classify-id",
            extract_endpoint="extract-id"
        )

        assert client.api_key == "test-key"
        assert client.classify_endpoint == "classify-id"
        assert client.extract_endpoint == "extract-id"

    def test_uses_default_timeout_and_retries(self):
        """Should use default timeout and max_retries."""
        client = RunPodClient(
            api_key="test-key",
            classify_endpoint="classify-id",
            extract_endpoint="extract-id"
        )

        assert client.timeout == 120
        assert client.max_retries == 3

    def test_accepts_custom_timeout_and_retries(self):
        """Should accept custom timeout and max_retries."""
        client = RunPodClient(
            api_key="test-key",
            classify_endpoint="classify-id",
            extract_endpoint="extract-id",
            timeout=60,
            max_retries=5
        )

        assert client.timeout == 60
        assert client.max_retries == 5


class TestRunPodClientClassify:
    """Tests for classify method."""

    @pytest.mark.asyncio
    async def test_classify_success(self, mock_runpod_classify_success):
        """Should return success response on COMPLETED status."""
        with patch('httpx.AsyncClient') as MockClient:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_runpod_classify_success
            mock_response.status_code = 200

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.aclose = AsyncMock()
            MockClient.return_value = mock_client

            client = RunPodClient(
                api_key="test-key",
                classify_endpoint="classify-id",
                extract_endpoint="extract-id"
            )
            client._client = mock_client

            result = await client.classify(["https://example.com/doc.png"])

            assert result.success is True
            assert result.data["type"] == "W2"
            assert result.data["confidence"] == 0.95
            assert result.job_id == "job-123"

    @pytest.mark.asyncio
    async def test_classify_with_custom_prompt(self):
        """Should pass custom prompt to endpoint."""
        with patch('httpx.AsyncClient') as MockClient:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "id": "job-123",
                "status": "COMPLETED",
                "output": {"type": "invoice", "confidence": 0.8},
                "executionTime": 1000
            }

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.aclose = AsyncMock()
            MockClient.return_value = mock_client

            client = RunPodClient(
                api_key="test-key",
                classify_endpoint="classify-id",
                extract_endpoint="extract-id"
            )
            client._client = mock_client

            await client.classify(
                ["https://example.com/doc.png"],
                prompt="Custom classification prompt"
            )

            # Verify prompt was passed
            call_args = mock_client.post.call_args
            payload = call_args.kwargs.get('json') or call_args[1].get('json')
            assert payload["input"]["prompt"] == "Custom classification prompt"

    @pytest.mark.asyncio
    async def test_classify_failed_status(self, mock_runpod_failure):
        """Should return error on FAILED status."""
        with patch('httpx.AsyncClient') as MockClient:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_runpod_failure
            mock_response.status_code = 200

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            MockClient.return_value = mock_client

            client = RunPodClient(
                api_key="test-key",
                classify_endpoint="classify-id",
                extract_endpoint="extract-id"
            )
            client._client = mock_client

            result = await client.classify(["https://example.com/doc.png"])

            assert result.success is False
            assert "Model inference failed" in result.error


class TestRunPodClientExtract:
    """Tests for extract method."""

    @pytest.mark.asyncio
    async def test_extract_success(self, mock_runpod_extract_success):
        """Should return success response on COMPLETED status."""
        with patch('httpx.AsyncClient') as MockClient:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_runpod_extract_success

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            MockClient.return_value = mock_client

            client = RunPodClient(
                api_key="test-key",
                classify_endpoint="classify-id",
                extract_endpoint="extract-id"
            )
            client._client = mock_client

            result = await client.extract(
                ["https://example.com/doc.png"],
                doc_type="W2"
            )

            assert result.success is True
            assert result.data["doc_type"] == "W2"
            assert result.job_id == "job-456"

    @pytest.mark.asyncio
    async def test_extract_passes_doc_type(self):
        """Should pass doc_type to endpoint."""
        with patch('httpx.AsyncClient') as MockClient:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "id": "job-123",
                "status": "COMPLETED",
                "output": {},
                "executionTime": 1000
            }

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            MockClient.return_value = mock_client

            client = RunPodClient(
                api_key="test-key",
                classify_endpoint="classify-id",
                extract_endpoint="extract-id"
            )
            client._client = mock_client

            await client.extract(
                ["https://example.com/doc.png"],
                doc_type="bank_statement"
            )

            # Verify doc_type was passed
            call_args = mock_client.post.call_args
            payload = call_args.kwargs.get('json') or call_args[1].get('json')
            assert payload["input"]["doc_type"] == "bank_statement"


class TestRunPodClientHealthCheck:
    """Tests for health_check method."""

    @pytest.mark.asyncio
    async def test_health_check_both_healthy(self, mock_runpod_health_success):
        """Should return healthy status for both endpoints."""
        with patch('httpx.AsyncClient') as MockClient:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_runpod_health_success
            mock_response.status_code = 200

            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            MockClient.return_value = mock_client

            client = RunPodClient(
                api_key="test-key",
                classify_endpoint="classify-id",
                extract_endpoint="extract-id"
            )
            client._client = mock_client

            result = await client.health_check()

            assert result["classify"]["status"] == "healthy"
            assert result["extract"]["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_check_idle_endpoint(self):
        """Should return idle status for 204 response."""
        with patch('httpx.AsyncClient') as MockClient:
            mock_response = MagicMock()
            mock_response.status_code = 204

            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            MockClient.return_value = mock_client

            client = RunPodClient(
                api_key="test-key",
                classify_endpoint="classify-id",
                extract_endpoint="extract-id"
            )
            client._client = mock_client

            result = await client.health_check()

            assert result["classify"]["status"] == "idle"

    @pytest.mark.asyncio
    async def test_health_check_unhealthy_endpoint(self):
        """Should return unhealthy status for error response."""
        with patch('httpx.AsyncClient') as MockClient:
            mock_response = MagicMock()
            mock_response.status_code = 500

            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            MockClient.return_value = mock_client

            client = RunPodClient(
                api_key="test-key",
                classify_endpoint="classify-id",
                extract_endpoint="extract-id"
            )
            client._client = mock_client

            result = await client.health_check()

            assert result["classify"]["status"] == "unhealthy"

    @pytest.mark.asyncio
    async def test_health_check_unreachable_endpoint(self):
        """Should return unreachable status on connection error."""
        with patch('httpx.AsyncClient') as MockClient:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
            MockClient.return_value = mock_client

            client = RunPodClient(
                api_key="test-key",
                classify_endpoint="classify-id",
                extract_endpoint="extract-id"
            )
            client._client = mock_client

            result = await client.health_check()

            assert result["classify"]["status"] == "unreachable"
            assert "Connection refused" in result["classify"]["error"]


class TestRunPodClientRetryLogic:
    """Tests for retry logic in _call method."""

    @pytest.mark.asyncio
    async def test_retries_on_timeout(self):
        """Should retry on timeout and eventually fail."""
        with patch('httpx.AsyncClient') as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
            MockClient.return_value = mock_client

            client = RunPodClient(
                api_key="test-key",
                classify_endpoint="classify-id",
                extract_endpoint="extract-id",
                max_retries=2
            )
            client._client = mock_client

            with patch('asyncio.sleep', new_callable=AsyncMock):
                result = await client.classify(["https://example.com/doc.png"])

            assert result.success is False
            assert "Max retries exceeded" in result.error
            assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_retries_on_5xx_errors(self):
        """Should retry on 5xx HTTP errors."""
        with patch('httpx.AsyncClient') as MockClient:
            mock_response = MagicMock()
            mock_response.status_code = 503
            mock_response.json.return_value = {"status": "error"}

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            MockClient.return_value = mock_client

            client = RunPodClient(
                api_key="test-key",
                classify_endpoint="classify-id",
                extract_endpoint="extract-id",
                max_retries=2
            )
            client._client = mock_client

            with patch('asyncio.sleep', new_callable=AsyncMock):
                result = await client.classify(["https://example.com/doc.png"])

            assert result.success is False
            assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_no_retry_on_auth_failure(self):
        """Should not retry on 401 authentication error."""
        with patch('httpx.AsyncClient') as MockClient:
            mock_response = MagicMock()
            mock_response.status_code = 401
            mock_response.json.return_value = {}

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            MockClient.return_value = mock_client

            client = RunPodClient(
                api_key="test-key",
                classify_endpoint="classify-id",
                extract_endpoint="extract-id",
                max_retries=3
            )
            client._client = mock_client

            result = await client.classify(["https://example.com/doc.png"])

            assert result.success is False
            assert "Authentication failed" in result.error
            # Should not retry on 401
            assert mock_client.post.call_count == 1

    @pytest.mark.asyncio
    async def test_exponential_backoff_delays(self):
        """Should use exponential backoff between retries."""
        with patch('httpx.AsyncClient') as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
            MockClient.return_value = mock_client

            client = RunPodClient(
                api_key="test-key",
                classify_endpoint="classify-id",
                extract_endpoint="extract-id",
                max_retries=3
            )
            client._client = mock_client

            with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
                await client.classify(["https://example.com/doc.png"])

                # Check backoff delays (5, 10 seconds with base_delay=5)
                delays = [call.args[0] for call in mock_sleep.call_args_list]
                assert len(delays) == 2  # 2 sleeps for 3 attempts
                assert delays[0] == 5.0  # First backoff
                assert delays[1] == 10.0  # Second backoff (exponential)


class TestRunPodResponse:
    """Tests for RunPodResponse dataclass."""

    def test_response_fields(self):
        """Should have all expected fields."""
        response = RunPodResponse(
            success=True,
            data={"type": "W2"},
            error=None,
            latency_ms=1500,
            job_id="job-123"
        )

        assert response.success is True
        assert response.data == {"type": "W2"}
        assert response.error is None
        assert response.latency_ms == 1500
        assert response.job_id == "job-123"

    def test_job_id_defaults_to_none(self):
        """Should default job_id to None."""
        response = RunPodResponse(
            success=False,
            data=None,
            error="Error",
            latency_ms=0
        )

        assert response.job_id is None


class TestRunPodClientClose:
    """Tests for close method."""

    @pytest.mark.asyncio
    async def test_close_calls_aclose_on_http_client(self):
        """Should call aclose on the underlying HTTP client."""
        with patch('httpx.AsyncClient') as MockClient:
            mock_client = AsyncMock()
            mock_client.aclose = AsyncMock()
            MockClient.return_value = mock_client

            client = RunPodClient(
                api_key="test-key",
                classify_endpoint="classify-id",
                extract_endpoint="extract-id"
            )
            client._client = mock_client

            await client.close()

            mock_client.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_can_be_called_multiple_times(self):
        """Should handle multiple close calls gracefully."""
        with patch('httpx.AsyncClient') as MockClient:
            mock_client = AsyncMock()
            mock_client.aclose = AsyncMock()
            MockClient.return_value = mock_client

            client = RunPodClient(
                api_key="test-key",
                classify_endpoint="classify-id",
                extract_endpoint="extract-id"
            )
            client._client = mock_client

            # Should not raise on multiple calls
            await client.close()
            await client.close()

            assert mock_client.aclose.call_count == 2


class TestRunPodClientHttpErrors:
    """Tests for HTTP error handling."""

    @pytest.mark.asyncio
    async def test_bad_request_400_returns_error(self):
        """Should return error on HTTP 400 bad request."""
        with patch('httpx.AsyncClient') as MockClient:
            mock_response = MagicMock()
            mock_response.status_code = 400
            mock_response.json.return_value = {"error": "Invalid image URL format"}

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            MockClient.return_value = mock_client

            client = RunPodClient(
                api_key="test-key",
                classify_endpoint="classify-id",
                extract_endpoint="extract-id",
                max_retries=3
            )
            client._client = mock_client

            result = await client.classify(["invalid-url"])

            assert result.success is False
            assert "Bad request" in result.error
            assert "Invalid image URL format" in result.error
            # Should not retry on 400
            assert mock_client.post.call_count == 1

    @pytest.mark.asyncio
    async def test_bad_request_400_with_unknown_error(self):
        """Should handle 400 response with no error message."""
        with patch('httpx.AsyncClient') as MockClient:
            mock_response = MagicMock()
            mock_response.status_code = 400
            mock_response.json.return_value = {}  # No error field

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            MockClient.return_value = mock_client

            client = RunPodClient(
                api_key="test-key",
                classify_endpoint="classify-id",
                extract_endpoint="extract-id"
            )
            client._client = mock_client

            result = await client.classify(["https://example.com/doc.png"])

            assert result.success is False
            assert "Bad request" in result.error
            assert "Unknown" in result.error


class TestRunPodClientJobStatus:
    """Tests for job status handling (IN_QUEUE, IN_PROGRESS)."""

    @pytest.mark.asyncio
    async def test_in_queue_status_triggers_retry(self):
        """Should retry when job is IN_QUEUE (timeout scenario)."""
        with patch('httpx.AsyncClient') as MockClient:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "id": "job-123",
                "status": "IN_QUEUE"
            }

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            MockClient.return_value = mock_client

            client = RunPodClient(
                api_key="test-key",
                classify_endpoint="classify-id",
                extract_endpoint="extract-id",
                max_retries=2
            )
            client._client = mock_client

            with patch('asyncio.sleep', new_callable=AsyncMock):
                result = await client.classify(["https://example.com/doc.png"])

            assert result.success is False
            assert "Max retries exceeded" in result.error
            assert "IN_QUEUE" in result.error
            assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_in_progress_status_triggers_retry(self):
        """Should retry when job is IN_PROGRESS (timeout scenario)."""
        with patch('httpx.AsyncClient') as MockClient:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "id": "job-456",
                "status": "IN_PROGRESS"
            }

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            MockClient.return_value = mock_client

            client = RunPodClient(
                api_key="test-key",
                classify_endpoint="classify-id",
                extract_endpoint="extract-id",
                max_retries=2
            )
            client._client = mock_client

            with patch('asyncio.sleep', new_callable=AsyncMock):
                result = await client.classify(["https://example.com/doc.png"])

            assert result.success is False
            assert "Max retries exceeded" in result.error
            assert "IN_PROGRESS" in result.error
            assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_in_queue_then_completed(self):
        """Should succeed if job completes after being in queue."""
        with patch('httpx.AsyncClient') as MockClient:
            in_queue_response = MagicMock()
            in_queue_response.status_code = 200
            in_queue_response.json.return_value = {
                "id": "job-123",
                "status": "IN_QUEUE"
            }

            completed_response = MagicMock()
            completed_response.status_code = 200
            completed_response.json.return_value = {
                "id": "job-123",
                "status": "COMPLETED",
                "output": {"type": "W2", "confidence": 0.95},
                "executionTime": 2000
            }

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=[in_queue_response, completed_response])
            MockClient.return_value = mock_client

            client = RunPodClient(
                api_key="test-key",
                classify_endpoint="classify-id",
                extract_endpoint="extract-id",
                max_retries=3
            )
            client._client = mock_client

            with patch('asyncio.sleep', new_callable=AsyncMock):
                result = await client.classify(["https://example.com/doc.png"])

            assert result.success is True
            assert result.data["type"] == "W2"
            assert mock_client.post.call_count == 2


class TestRunPodClientRequestError:
    """Tests for request error handling."""

    @pytest.mark.asyncio
    async def test_request_error_triggers_retry(self):
        """Should retry on generic request errors."""
        with patch('httpx.AsyncClient') as MockClient:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=httpx.RequestError("Network error"))
            MockClient.return_value = mock_client

            client = RunPodClient(
                api_key="test-key",
                classify_endpoint="classify-id",
                extract_endpoint="extract-id",
                max_retries=2
            )
            client._client = mock_client

            with patch('asyncio.sleep', new_callable=AsyncMock):
                result = await client.classify(["https://example.com/doc.png"])

            assert result.success is False
            assert "Max retries exceeded" in result.error
            assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_rate_limit_429_triggers_retry(self):
        """Should retry on HTTP 429 rate limit."""
        with patch('httpx.AsyncClient') as MockClient:
            mock_response = MagicMock()
            mock_response.status_code = 429
            mock_response.json.return_value = {"error": "Rate limit exceeded"}

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            MockClient.return_value = mock_client

            client = RunPodClient(
                api_key="test-key",
                classify_endpoint="classify-id",
                extract_endpoint="extract-id",
                max_retries=2
            )
            client._client = mock_client

            with patch('asyncio.sleep', new_callable=AsyncMock):
                result = await client.classify(["https://example.com/doc.png"])

            assert result.success is False
            assert "Max retries exceeded" in result.error
            assert "429" in result.error
            assert mock_client.post.call_count == 2
