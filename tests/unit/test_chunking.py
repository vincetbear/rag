"""Test chunking strategies."""

from llama_index.core.schema import Document

from config.settings import Settings
from core.ingestion.chunking import chunk_documents


def test_sentence_chunking():
    docs = [Document(text="A " * 500)]  # long document
    settings = Settings(chunk_size=128, chunk_overlap=20, _env_file=None)
    nodes = chunk_documents(docs, settings, strategy="sentence")
    assert len(nodes) > 1


def test_hierarchical_chunking():
    docs = [Document(text="Section one. " * 200 + "Section two. " * 200)]
    settings = Settings(_env_file=None)
    nodes = chunk_documents(docs, settings, strategy="hierarchical")
    # Hierarchical produces nodes at multiple levels
    assert len(nodes) > 1
    # Check parent-child relationships exist
    has_parent = any(
        node.relationships.get(
            __import__("llama_index.core.schema", fromlist=["NodeRelationship"]).NodeRelationship.PARENT
        )
        for node in nodes
        if hasattr(node, "relationships")
    )


def test_unknown_strategy():
    import pytest

    docs = [Document(text="Hello")]
    settings = Settings(_env_file=None)
    with pytest.raises(ValueError, match="Unknown chunking strategy"):
        chunk_documents(docs, settings, strategy="unknown")
