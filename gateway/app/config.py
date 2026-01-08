"""Gateway configuration from environment variables."""

import os
from dataclasses import dataclass
from functools import lru_cache


@dataclass
class Settings:
    """Application settings."""

    # API Authentication
    api_key: str

    # RunPod Configuration
    runpod_api_key: str
    runpod_classify_endpoint: str
    runpod_extract_endpoint: str
    runpod_extract_chandra_endpoint: str = ""  # Optional: Chandra evaluation endpoint
    runpod_timeout: int = 120
    runpod_max_retries: int = 3

    # Rate Limiting
    rate_limit_requests: int = 100  # per minute
    rate_limit_window: int = 60     # seconds

    # Logging
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from environment variables."""
        return cls(
            api_key=os.environ["GATEWAY_API_KEY"],
            runpod_api_key=os.environ["RUNPOD_API_KEY"],
            runpod_classify_endpoint=os.environ["RUNPOD_CLASSIFY_ENDPOINT"],
            runpod_extract_endpoint=os.environ["RUNPOD_EXTRACT_ENDPOINT"],
            runpod_extract_chandra_endpoint=os.environ.get("RUNPOD_EXTRACT_CHANDRA_ENDPOINT", ""),
            runpod_timeout=int(os.environ.get("RUNPOD_TIMEOUT", "120")),
            runpod_max_retries=int(os.environ.get("RUNPOD_MAX_RETRIES", "3")),
            rate_limit_requests=int(os.environ.get("RATE_LIMIT_REQUESTS", "100")),
            rate_limit_window=int(os.environ.get("RATE_LIMIT_WINDOW", "60")),
            log_level=os.environ.get("LOG_LEVEL", "INFO")
        )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings.from_env()
