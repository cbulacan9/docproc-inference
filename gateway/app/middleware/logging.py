"""Request logging middleware."""

import logging
import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all requests with timing and request ID."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        # Generate request ID
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id

        # Log request
        client_host = request.client.host if request.client else "unknown"
        logger.info(
            f"[{request_id}] {request.method} {request.url.path} "
            f"from {client_host}"
        )

        # Process request
        start_time = time.time()

        try:
            response = await call_next(request)

            # Log response
            duration_ms = int((time.time() - start_time) * 1000)
            logger.info(
                f"[{request_id}] {response.status_code} "
                f"in {duration_ms}ms"
            )

            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id

            return response

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.error(
                f"[{request_id}] Error after {duration_ms}ms: {e}"
            )
            raise
