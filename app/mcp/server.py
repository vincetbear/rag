"""Mount MCP on the FastAPI app — auto-exposes all endpoints as MCP tools."""

from fastapi import FastAPI


def mount_mcp(app: FastAPI) -> None:
    try:
        from fastapi_mcp import FastApiMCP

        mcp = FastApiMCP(
            app,
            name="RAG Service",
            description="Advanced RAG service with hybrid search, reranking, and query transformation",
        )
        mcp.mount()
    except ImportError:
        import logging

        logging.getLogger(__name__).warning(
            "fastapi-mcp not installed — MCP endpoint disabled"
        )
