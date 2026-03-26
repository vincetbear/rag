from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore

from config.settings import Settings


def create_qdrant_client(settings: Settings) -> QdrantClient:
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
    )


def create_vector_store(
    client: QdrantClient,
    collection_name: str,
    *,
    enable_hybrid: bool = True,
) -> QdrantVectorStore:
    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        enable_hybrid=enable_hybrid,
        fastembed_sparse_model="Qdrant/bm25",
    )
