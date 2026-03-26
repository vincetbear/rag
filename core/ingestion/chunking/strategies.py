"""Advanced chunking strategies: semantic and hierarchical."""

from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.node_parser import (
    SentenceSplitter,
    HierarchicalNodeParser,
    SemanticSplitterNodeParser,
)
from llama_index.core.schema import BaseNode, Document

from config.settings import Settings


def semantic_chunk(
    documents: list[Document],
    embed_model: BaseEmbedding,
    buffer_size: int = 1,
    breakpoint_percentile_threshold: int = 95,
) -> list[BaseNode]:
    parser = SemanticSplitterNodeParser(
        embed_model=embed_model,
        buffer_size=buffer_size,
        breakpoint_percentile_threshold=breakpoint_percentile_threshold,
    )
    return parser.get_nodes_from_documents(documents)


def hierarchical_chunk(
    documents: list[Document],
    chunk_sizes: list[int] | None = None,
) -> list[BaseNode]:
    # 3 levels: large (2048) → medium (512) → leaf (256)
    # leaf should be big enough to preserve context (e.g. full regulation clauses)
    chunk_sizes = chunk_sizes or [2048, 512, 256]
    parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)
    return parser.get_nodes_from_documents(documents)
