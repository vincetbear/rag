"""AutoMerging / Hierarchical retriever."""

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.retrievers import AutoMergingRetriever, BaseRetriever
from llama_index.core.storage.docstore import SimpleDocumentStore


def create_automerging_retriever(
    index: VectorStoreIndex,
    docstore: SimpleDocumentStore,
    top_k: int = 10,
) -> BaseRetriever:
    """Create an AutoMergingRetriever that merges leaf nodes back to parents.

    When >50% of a parent's children are retrieved, they get merged into
    the parent node for better context.
    """
    base_retriever = index.as_retriever(similarity_top_k=top_k)

    # Rebuild storage_context with the docstore so AutoMerging can find parents
    storage_context = StorageContext.from_defaults(
        vector_store=index.vector_store,
        docstore=docstore,
    )

    return AutoMergingRetriever(
        vector_retriever=base_retriever,
        storage_context=storage_context,
        simple_ratio_thresh=0.5,
    )
