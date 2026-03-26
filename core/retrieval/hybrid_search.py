"""Dense + BM25 sparse hybrid retriever via Qdrant."""

from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import BaseRetriever


def create_hybrid_retriever(
    index: VectorStoreIndex,
    top_k: int = 10,
    alpha: float = 0.5,
) -> BaseRetriever:
    return index.as_retriever(
        similarity_top_k=top_k,
        vector_store_query_mode="hybrid",
        sparse_top_k=top_k,
        alpha=alpha,
    )
