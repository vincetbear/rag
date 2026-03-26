"""Request/Response logging middleware."""

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("rag.access")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log API and MCP requests with timing info."""

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Skip static files, docs, UI
        if path in ("/", "/docs", "/redoc", "/openapi.json") or path.startswith("/static"):
            return await call_next(request)

        request_id = str(uuid.uuid4())[:8]
        start = time.time()
        client_ip = request.client.host if request.client else "unknown"
        method = request.method
        is_mcp = "/mcp" in path
        tag = "[MCP]" if is_mcp else "[API]"

        # Read request body
        req_body = await request.body()
        req_text = _safe_decode(req_body)

        # Log request
        logger.info(
            "%s %s REQUEST  %s %s | client=%s | body=%s",
            tag, request_id, method, path, client_ip, _truncate(req_text),
        )

        # MCP uses SSE streaming — cannot consume body_iterator
        # Only log request + status for MCP, full body for REST API
        if is_mcp:
            response = await call_next(request)
            elapsed = time.time() - start
            logger.info(
                "%s %s RESPONSE %s %s | status=%s | %.2fs",
                tag, request_id, method, path, response.status_code, elapsed,
            )
            return response

        # REST API — capture full response body
        response = await call_next(request)

        resp_body_parts = []
        async for chunk in response.body_iterator:
            if isinstance(chunk, bytes):
                resp_body_parts.append(chunk)
            else:
                resp_body_parts.append(chunk.encode("utf-8"))
        resp_bytes = b"".join(resp_body_parts)
        resp_text = _safe_decode(resp_bytes)

        elapsed = time.time() - start

        logger.info(
            "%s %s RESPONSE %s %s | status=%s | %.2fs | body=%s",
            tag, request_id, method, path, response.status_code, elapsed,
            _truncate(resp_text),
        )

        return Response(
            content=resp_bytes,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
        )


def _safe_decode(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return f"<binary {len(data)} bytes>"


def _truncate(text: str, max_len: int = 2000) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"...({len(text)} total)"
