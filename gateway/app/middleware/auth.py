"""API key authentication middleware."""

import logging
import secrets
from typing import Callable

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

logger = logging.getLogger(__name__)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def create_auth_dependency(valid_api_key: str) -> Callable:
    """
    Create an API key validation dependency.

    Args:
        valid_api_key: The expected API key

    Returns:
        FastAPI dependency function
    """
    async def verify_api_key(
        api_key: str = Security(api_key_header)
    ) -> str:
        if api_key is None:
            logger.warning("Missing API key in request")
            raise HTTPException(
                status_code=401,
                detail="Missing API key"
            )

        # Constant-time comparison to prevent timing attacks
        if not secrets.compare_digest(api_key, valid_api_key):
            logger.warning("Invalid API key attempt")
            raise HTTPException(
                status_code=401,
                detail="Invalid API key"
            )

        return api_key

    return verify_api_key
