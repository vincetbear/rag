from fastapi import APIRouter, Depends
from qdrant_client import QdrantClient

from app.dependencies import get_qdrant_client

router = APIRouter()


@router.get("/health", operation_id="health")
async def health(client: QdrantClient = Depends(get_qdrant_client)):
    try:
        collections = client.get_collections()
        qdrant_ok = True
        collection_count = len(collections.collections)
    except Exception:
        qdrant_ok = False
        collection_count = 0

    return {
        "status": "ok" if qdrant_ok else "degraded",
        "qdrant": {"connected": qdrant_ok, "collections": collection_count},
    }
