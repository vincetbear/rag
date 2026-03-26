import logging

from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import (
    get_cached_settings,
    get_llm,
    get_embed_model,
    get_qdrant_client,
)
from app.api.schemas.query import QueryRequest, QueryResponse, SourceNode
from config.settings import Settings
from core.retrieval.engine import RetrievalEngine

logger = logging.getLogger("rag.access")
router = APIRouter()


def _get_engine(
    settings: Settings = Depends(get_cached_settings),
    llm=Depends(get_llm),
    embed_model=Depends(get_embed_model),
    qdrant_client=Depends(get_qdrant_client),
) -> RetrievalEngine:
    return RetrievalEngine(settings, llm, embed_model, qdrant_client)


@router.post("/query", response_model=QueryResponse, operation_id="query")
async def query(
    request: QueryRequest,
    engine: RetrievalEngine = Depends(_get_engine),
):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        result = engine.query(
            query_text=request.query,
            collection=request.collection,
            top_k=request.top_k,
            rerank_top_n=request.rerank_top_n,
            enable_hyde=request.enable_hyde,
            enable_sub_questions=request.enable_sub_questions,
            enable_multi_query=request.enable_multi_query,
            filters=request.filters,
        )
    except Exception as e:
        logger.error("Query failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    sources = [SourceNode(**s) for s in result["sources"]]
    return QueryResponse(
        answer=result["answer"],
        sources=sources,
        metadata=result["metadata"],
    )
