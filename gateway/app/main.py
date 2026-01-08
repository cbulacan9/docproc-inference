"""Gateway server main application."""

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .middleware.auth import create_auth_dependency
from .middleware.logging import RequestLoggingMiddleware
from .routes import classify, extract, extract_chandra, health
from .services.runpod_client import RunPodClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    settings = get_settings()
    logger.info("Gateway server starting...")
    logger.info(f"Classify endpoint: {settings.runpod_classify_endpoint}")
    logger.info(f"Extract endpoint: {settings.runpod_extract_endpoint}")
    if settings.runpod_extract_chandra_endpoint:
        logger.info(f"Extract Chandra endpoint: {settings.runpod_extract_chandra_endpoint}")

    # Create shared RunPod client
    app.state.runpod_client = RunPodClient(
        api_key=settings.runpod_api_key,
        classify_endpoint=settings.runpod_classify_endpoint,
        extract_endpoint=settings.runpod_extract_endpoint,
        extract_chandra_endpoint=settings.runpod_extract_chandra_endpoint,
        timeout=settings.runpod_timeout,
        max_retries=settings.runpod_max_retries
    )
    logger.info("RunPod client initialized")

    yield

    # Cleanup
    await app.state.runpod_client.close()
    logger.info("Gateway server shutting down...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Document Processing Inference Gateway",
        description="Gateway for document classification and extraction services",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )

    # Add CORS middleware
    # TODO: Restrict origins in production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add request logging
    app.add_middleware(RequestLoggingMiddleware)

    # Create auth dependency with configured API key
    auth_dependency = Depends(create_auth_dependency(settings.api_key))

    # Include routers
    # Health routes (no auth)
    app.include_router(health.router)

    # Inference routes (with auth via include_router pattern)
    app.include_router(classify.router, dependencies=[auth_dependency])
    app.include_router(extract.router, dependencies=[auth_dependency])
    app.include_router(extract_chandra.router, dependencies=[auth_dependency])

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
