"""RunPod Serverless API client."""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)

RUNPOD_API_BASE = "https://api.runpod.ai/v2"


@dataclass
class RunPodResponse:
    """Response from RunPod endpoint."""
    success: bool
    data: dict | None
    error: str | None
    latency_ms: int
    job_id: str | None = None


class RunPodClient:
    """
    Async client for RunPod Serverless endpoints.

    Handles authentication, retries, and error handling.
    """

    def __init__(
        self,
        api_key: str,
        classify_endpoint: str,
        extract_endpoint: str,
        extract_chandra_endpoint: str = "",
        timeout: int = 120,
        max_retries: int = 3
    ):
        self.api_key = api_key
        self.classify_endpoint = classify_endpoint
        self.extract_endpoint = extract_endpoint
        self.extract_chandra_endpoint = extract_chandra_endpoint
        self.timeout = timeout
        self.max_retries = max_retries

        self._client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            timeout=httpx.Timeout(timeout, connect=10)
        )

    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()

    async def classify(
        self,
        image_urls: list[str],
        prompt: str | None = None
    ) -> RunPodResponse:
        """
        Call classification endpoint.

        Args:
            image_urls: ShareFile pre-signed URLs
            prompt: Optional custom prompt

        Returns:
            RunPodResponse with classification result
        """
        return await self._call(
            endpoint_id=self.classify_endpoint,
            payload={
                "image_urls": image_urls,
                "prompt": prompt
            }
        )

    async def extract(
        self,
        image_urls: list[str],
        doc_type: str,
        prompt: str | None = None
    ) -> RunPodResponse:
        """
        Call extraction endpoint.

        Args:
            image_urls: ShareFile pre-signed URLs
            doc_type: Document type for prompt selection
            prompt: Optional custom prompt

        Returns:
            RunPodResponse with extraction result
        """
        return await self._call(
            endpoint_id=self.extract_endpoint,
            payload={
                "image_urls": image_urls,
                "doc_type": doc_type,
                "prompt": prompt
            }
        )

    async def extract_chandra(
        self,
        image_urls: list[str],
        doc_type: str
    ) -> RunPodResponse:
        """
        Call Chandra extraction endpoint (evaluation).

        Args:
            image_urls: ShareFile pre-signed URLs
            doc_type: Document type for prompt selection

        Returns:
            RunPodResponse with extraction result from Chandra
        """
        if not self.extract_chandra_endpoint:
            return RunPodResponse(
                success=False,
                data=None,
                error="Chandra endpoint not configured",
                latency_ms=0
            )

        return await self._call(
            endpoint_id=self.extract_chandra_endpoint,
            payload={
                "image_urls": image_urls,
                "doc_type": doc_type
            }
        )

    async def health_check(self) -> dict:
        """Check health of all configured endpoints."""
        results = {}

        endpoints = [
            ("classify", self.classify_endpoint),
            ("extract", self.extract_endpoint)
        ]

        # Include Chandra endpoint if configured
        if self.extract_chandra_endpoint:
            endpoints.append(("extract_chandra", self.extract_chandra_endpoint))

        for name, endpoint_id in endpoints:
            try:
                url = f"{RUNPOD_API_BASE}/{endpoint_id}/health"
                response = await self._client.get(url, timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    results[name] = {
                        "status": "healthy",
                        "workers": data.get("workers", {})
                    }
                elif response.status_code == 204:
                    results[name] = {
                        "status": "idle",
                        "workers": {"idle": 0, "running": 0}
                    }
                else:
                    results[name] = {
                        "status": "unhealthy",
                        "error": f"HTTP {response.status_code}"
                    }
            except Exception as e:
                results[name] = {
                    "status": "unreachable",
                    "error": str(e)
                }

        return results

    async def _call(
        self,
        endpoint_id: str,
        payload: dict
    ) -> RunPodResponse:
        """
        Make a synchronous call to a RunPod endpoint.

        Uses /runsync for immediate response.
        """
        url = f"{RUNPOD_API_BASE}/{endpoint_id}/runsync"

        last_error = None
        base_delay = 5.0

        for attempt in range(self.max_retries):
            start_time = time.time()

            try:
                response = await self._client.post(
                    url,
                    json={"input": payload}
                )

                latency_ms = int((time.time() - start_time) * 1000)
                data = response.json()

                status = data.get("status")
                job_id = data.get("id")

                # Success
                if status == "COMPLETED":
                    return RunPodResponse(
                        success=True,
                        data=data.get("output", {}),
                        error=None,
                        latency_ms=data.get("executionTime", latency_ms),
                        job_id=job_id
                    )

                # Failed
                if status == "FAILED":
                    error_msg = data.get("error", "Unknown error")
                    return RunPodResponse(
                        success=False,
                        data=None,
                        error=error_msg,
                        latency_ms=latency_ms,
                        job_id=job_id
                    )

                # Timeout/in-progress
                if status in ("IN_QUEUE", "IN_PROGRESS"):
                    last_error = f"Job timed out in status: {status}"
                    logger.warning(f"RunPod job {job_id} timed out: {status}")

                # HTTP errors
                if response.status_code == 401:
                    return RunPodResponse(
                        success=False,
                        data=None,
                        error="Authentication failed",
                        latency_ms=latency_ms
                    )

                if response.status_code == 400:
                    return RunPodResponse(
                        success=False,
                        data=None,
                        error=f"Bad request: {data.get('error', 'Unknown')}",
                        latency_ms=latency_ms
                    )

                # Retry on server errors
                if response.status_code in (429, 500, 502, 503, 504):
                    last_error = f"HTTP {response.status_code}"
                    logger.warning(
                        f"RunPod error {response.status_code}, "
                        f"attempt {attempt + 1}/{self.max_retries}"
                    )

            except httpx.TimeoutException:
                last_error = "Request timeout"
                logger.warning(
                    f"RunPod timeout, attempt {attempt + 1}/{self.max_retries}"
                )

            except httpx.RequestError as e:
                last_error = str(e)
                logger.warning(
                    f"RunPod request error: {e}, "
                    f"attempt {attempt + 1}/{self.max_retries}"
                )

            # Exponential backoff
            if attempt < self.max_retries - 1:
                delay = min(base_delay * (2 ** attempt), 60)
                logger.info(f"Retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)

        # All retries exhausted
        return RunPodResponse(
            success=False,
            data=None,
            error=f"Max retries exceeded: {last_error}",
            latency_ms=0
        )
