"""Chunking strategies — returns list[BaseNode]."""

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode, Document
from llama_index.core.embeddings import BaseEmbedding

from config.settings import Settings


def chunk_documents(
    documents: list[Document],
    settings: Settings,
    strategy: str = "sentence",
    embed_model: BaseEmbedding | None = None,
) -> list[BaseNode]:
    if strategy == "sentence":
        parser = SentenceSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        return parser.get_nodes_from_documents(documents)

    if strategy == "semantic":
        from core.ingestion.chunking.strategies import semantic_chunk

        if embed_model is None:
            raise ValueError("embed_model required for semantic chunking")
        return semantic_chunk(documents, embed_model)

    if strategy == "hierarchical":
        from core.ingestion.chunking.strategies import hierarchical_chunk

        return hierarchical_chunk(documents)

    raise ValueError(f"Unknown chunking strategy: {strategy}")
