from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams

from app.dependencies import get_index_manager, get_qdrant_client, get_cached_settings
from config.settings import Settings
from core.indexing.index_manager import IndexManager

router = APIRouter()


class CreateCollectionRequest(BaseModel):
    name: str


@router.get("/collections", operation_id="list_collections")
async def list_collections(manager: IndexManager = Depends(get_index_manager)):
    return {"collections": manager.list_collections()}


@router.get("/collections/{name}", operation_id="get_collection")
async def get_collection(name: str, manager: IndexManager = Depends(get_index_manager)):
    if not manager.collection_exists(name):
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")
    return manager.collection_info(name)


@router.post("/collections", operation_id="create_collection")
async def create_collection(
    request: CreateCollectionRequest,
    manager: IndexManager = Depends(get_index_manager),
    client=Depends(get_qdrant_client),
    settings: Settings = Depends(get_cached_settings),
):
    if manager.collection_exists(request.name):
        return {"status": "exists", "collection": request.name}

    # Explicitly create collection with proper dense + sparse config
    vectors_config = {
        "text-dense": VectorParams(size=1024, distance=Distance.COSINE),
    }
    sparse_vectors_config = None
    if settings.enable_hybrid:
        sparse_vectors_config = {
            "text-sparse": SparseVectorParams(),
        }

    client.create_collection(
        collection_name=request.name,
        vectors_config=vectors_config,
        sparse_vectors_config=sparse_vectors_config,
    )
    return {"status": "ok", "collection": request.name}


@router.delete("/collections/{name}", operation_id="delete_collection")
async def delete_collection(name: str, manager: IndexManager = Depends(get_index_manager)):
    if not manager.collection_exists(name):
        raise HTTPException(status_code=404, detail=f"Collection '{name}' not found")
    manager.delete_collection(name)
    return {"status": "deleted", "collection": name}
