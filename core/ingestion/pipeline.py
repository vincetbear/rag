"""IngestionPipeline — load → chunk → embed → store."""

import os
from pathlib import Path

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.node_parser import get_leaf_nodes
from llama_index.core.schema import BaseNode
from llama_index.core.storage.docstore import SimpleDocumentStore
from qdrant_client import QdrantClient

from config.settings import Settings
from core.ingestion.loaders import load_file
from core.ingestion.chunking import chunk_documents
from core.indexing.qdrant_store import create_vector_store

# Persistent docstore directory
_DOCSTORE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "docstores")


def _docstore_path(collection: str) -> str:
    path = os.path.abspath(os.path.join(_DOCSTORE_DIR, collection))
    os.makedirs(path, exist_ok=True)
    return path


def get_docstore(collection: str) -> SimpleDocumentStore | None:
    """Load docstore from disk if it exists (for AutoMergingRetriever)."""
    path = _docstore_path(collection)
    persist_file = os.path.join(path, "docstore.json")
    if os.path.exists(persist_file):
        return SimpleDocumentStore.from_persist_path(persist_file)
    return None


class IngestionPipeline:
    def __init__(
        self,
        settings: Settings,
        embed_model: BaseEmbedding,
        qdrant_client: QdrantClient,
    ):
        self._settings = settings
        self._embed_model = embed_model
        self._qdrant_client = qdrant_client

    def ingest_file(
        self,
        file_path: str | Path,
        collection: str | None = None,
        chunk_strategy: str = "sentence",
        filename: str | None = None,
    ) -> dict:
        collection = collection or self._settings.default_collection

        # 1. Load
        documents = load_file(file_path)
        if not documents:
            return {"status": "empty", "documents": 0, "nodes": 0}

        # Override source metadata with original filename
        if filename:
            for doc in documents:
                doc.metadata["source"] = filename

        # 2. Chunk
        nodes = chunk_documents(
            documents,
            self._settings,
            strategy=chunk_strategy,
            embed_model=self._embed_model if chunk_strategy == "semantic" else None,
        )

        # 3. Embed + Store
        if chunk_strategy == "hierarchical":
            self._store_hierarchical(nodes, collection)
        else:
            self._store_nodes(nodes, collection)

        return {
            "status": "ok",
            "documents": len(documents),
            "nodes": len(nodes),
            "collection": collection,
            "chunk_strategy": chunk_strategy,
        }

    def _store_nodes(self, nodes: list[BaseNode], collection: str) -> None:
        vector_store = create_vector_store(
            self._qdrant_client,
            collection,
            enable_hybrid=self._settings.enable_hybrid,
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        VectorStoreIndex(
            nodes=nodes,
            embed_model=self._embed_model,
            storage_context=storage_context,
        )

    def _store_hierarchical(self, all_nodes: list[BaseNode], collection: str) -> None:
        """Store hierarchical nodes: all nodes in docstore, only leaves in vector store."""
        leaf_nodes = get_leaf_nodes(all_nodes)

        # Build docstore with ALL nodes (parents + children) for AutoMerging
        # Load existing docstore if available (append mode)
        persist_file = os.path.join(_docstore_path(collection), "docstore.json")
        if os.path.exists(persist_file):
            docstore = SimpleDocumentStore.from_persist_path(persist_file)
        else:
            docstore = SimpleDocumentStore()
        docstore.add_documents(all_nodes)

        # Persist to disk so it survives server restarts
        docstore.persist(persist_path=persist_file)

        # Only embed and store leaf nodes in Qdrant
        vector_store = create_vector_store(
            self._qdrant_client,
            collection,
            enable_hybrid=self._settings.enable_hybrid,
        )
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            docstore=docstore,
        )
        VectorStoreIndex(
            nodes=leaf_nodes,
            embed_model=self._embed_model,
            storage_context=storage_context,
        )
