"""FastAPI app factory + MCP mount."""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from config.settings import get_settings


def _setup_logging(log_level: str) -> None:
    """Configure access logger to console + file."""
    logger = logging.getLogger("rag.access")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Console
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File
    log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    os.makedirs(log_dir, exist_ok=True)
    fh = logging.FileHandler(
        os.path.join(log_dir, "access.log"), encoding="utf-8"
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)


def _configure_cache_dirs(settings) -> None:
    """Set environment variables for model cache directories."""
    if settings.model_cache_dir:
        # HuggingFace models (embedding, reranker)
        os.environ["HF_HOME"] = settings.model_cache_dir
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = os.path.join(
            settings.model_cache_dir, "sentence_transformers"
        )
    if settings.fastembed_cache_dir:
        # fastembed (BM25 sparse)
        os.environ["FASTEMBED_CACHE_PATH"] = settings.fastembed_cache_dir


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: warm up expensive singletons
    from app.dependencies import get_qdrant_client

    get_qdrant_client()
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    _configure_cache_dirs(settings)
    _setup_logging(settings.log_level)

    app = FastAPI(
        title="Advanced RAG Service",
        version="0.1.0",
        lifespan=lifespan,
    )

    # --- Logging middleware ---
    from app.middleware.logging import RequestLoggingMiddleware

    app.add_middleware(RequestLoggingMiddleware)

    # --- REST API routers ---
    from app.api.v1.health import router as health_router
    from app.api.v1.ingest import router as ingest_router
    from app.api.v1.query import router as query_router
    from app.api.v1.collections import router as collections_router

    app.include_router(health_router, prefix="/v1", tags=["health"])
    app.include_router(ingest_router, prefix="/v1", tags=["ingest"])
    app.include_router(query_router, prefix="/v1", tags=["query"])
    app.include_router(collections_router, prefix="/v1", tags=["collections"])

    # --- UI ---
    from app.ui import router as ui_router

    app.include_router(ui_router, tags=["ui"])

    # --- MCP mount ---
    from app.mcp.server import mount_mcp

    mount_mcp(app)

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    port = int(os.environ.get("PORT", settings.port))
    uvicorn.run("app.main:app", host=settings.host, port=port, reload=True)
