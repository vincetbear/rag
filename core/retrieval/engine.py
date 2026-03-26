"""Main retrieval engine — combines all strategies."""

import logging
import re

from llama_index.core import VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import LLM
from llama_index.core.query_engine import RetrieverQueryEngine, SubQuestionQueryEngine
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from qdrant_client import QdrantClient

from config.settings import Settings
from core.generation.response_synth import create_response_synthesizer
from core.indexing.qdrant_store import create_vector_store
from core.retrieval.hybrid_search import create_hybrid_retriever
from core.retrieval.mmr_postprocessor import MMRPostprocessor
from core.retrieval.multi_query import retrieve_with_multi_query
from core.retrieval.reranker import create_reranker
from core.retrieval.query_transform import create_hyde_transform, create_sub_question_generator
from core.retrieval.hierarchical_retriever import create_automerging_retriever

logger = logging.getLogger("rag.retrieval")

# Pattern for queries that benefit from keyword (BM25) emphasis
_KEYWORD_PATTERN = re.compile(
    r"\d{2,}"            # numbers with 2+ digits (IDs, dates, codes)
    r"|[A-Z]{2,}"        # uppercase acronyms (API, HTTP, ISO)
    r"|[A-Z][a-z]+(?:[A-Z][a-z]+)+"  # CamelCase proper nouns
)


def _estimate_alpha(query: str, default: float = 0.5) -> float:
    """Heuristic: shift alpha toward BM25 if query has numbers/acronyms."""
    if _KEYWORD_PATTERN.search(query):
        return 0.3  # more BM25 weight
    return 0.7      # more dense weight


class RetrievalEngine:
    def __init__(
        self,
        settings: Settings,
        llm: LLM,
        embed_model: BaseEmbedding,
        qdrant_client: QdrantClient,
    ):
        self._settings = settings
        self._llm = llm
        self._embed_model = embed_model
        self._qdrant_client = qdrant_client

    def query(
        self,
        query_text: str,
        collection: str | None = None,
        top_k: int | None = None,
        rerank_top_n: int | None = None,
        enable_hyde: bool | None = None,
        enable_sub_questions: bool = False,
        enable_multi_query: bool = False,
        filters: dict | None = None,
    ) -> dict:
        collection = collection or self._settings.default_collection
        top_k = top_k or self._settings.top_k
        rerank_top_n = rerank_top_n or self._settings.rerank_top_n
        if enable_hyde is None:
            enable_hyde = self._settings.enable_hyde

        # Determine hybrid alpha dynamically
        alpha = _estimate_alpha(query_text, self._settings.hybrid_alpha)

        # Build index from existing Qdrant collection
        enable_hybrid = self._settings.enable_hybrid
        if enable_hybrid:
            try:
                col_info = self._qdrant_client.get_collection(collection)
                vectors_cfg = col_info.config.params.vectors
                if isinstance(vectors_cfg, dict) and "text-sparse" not in vectors_cfg:
                    enable_hybrid = False
            except Exception:
                enable_hybrid = False

        vector_store = create_vector_store(
            self._qdrant_client,
            collection,
            enable_hybrid=enable_hybrid,
        )
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=self._embed_model,
        )

        # --- Retriever selection ---
        retriever = self._build_retriever(index, collection, top_k, enable_hybrid, alpha)

        # --- Multi-query expansion (retrieves nodes directly) ---
        multi_query_used = False
        pre_retrieved_nodes: list[NodeWithScore] | None = None
        if enable_multi_query:
            pre_retrieved_nodes = retrieve_with_multi_query(
                retriever, self._llm, query_text,
            )
            multi_query_used = True

        # Node postprocessors
        postprocessors = [
            create_reranker(self._settings.reranker_model, top_n=rerank_top_n),
            MMRPostprocessor(similarity_threshold=0.95),
        ]

        # Response synthesizer — use configurable mode
        response_synthesizer = create_response_synthesizer(
            llm=self._llm,
            mode=self._settings.response_mode,
        )

        # Base query engine
        base_query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=postprocessors,
        )

        # --- Optional: Sub-question decomposition ---
        if enable_sub_questions:
            query_engine = self._wrap_sub_question(
                base_query_engine, collection, response_synthesizer
            )
        else:
            query_engine = base_query_engine

        # --- Optional: HyDE query transform ---
        hyde_applied = False
        if enable_hyde:
            try:
                from llama_index.core.query_engine import TransformQueryEngine

                hyde_transform = create_hyde_transform(self._llm)
                query_engine = TransformQueryEngine(
                    query_engine=query_engine,
                    query_transform=hyde_transform,
                )
                hyde_applied = True
            except Exception as e:
                logger.warning("HyDE transform failed, falling back to direct query: %s", e)

        # Execute
        try:
            if multi_query_used and pre_retrieved_nodes is not None:
                # Apply postprocessors manually, then synthesize
                nodes = pre_retrieved_nodes
                query_bundle = QueryBundle(query_str=query_text)
                for pp in postprocessors:
                    nodes = pp.postprocess_nodes(nodes, query_bundle)
                response = response_synthesizer.synthesize(query_text, nodes=nodes)
            else:
                response = query_engine.query(query_text)
        except Exception as e:
            if hyde_applied:
                # HyDE LLM call may have failed mid-stream — retry without HyDE
                logger.warning("Query with HyDE failed (%s), retrying without HyDE", e)
                try:
                    response = base_query_engine.query(query_text)
                except Exception:
                    raise
            elif enable_sub_questions and query_engine is not base_query_engine:
                logger.warning("Sub-question query failed (%s), falling back to base engine", e)
                response = base_query_engine.query(query_text)
            else:
                raise

        # Format response
        sources = []
        for node in response.source_nodes:
            sources.append(
                {
                    "text": node.node.get_content()[:500],
                    "score": node.score or 0.0,
                    "metadata": node.node.metadata,
                }
            )

        return {
            "answer": str(response),
            "sources": sources,
            "metadata": {
                "collection": collection,
                "top_k": top_k,
                "hybrid_alpha": alpha,
                "enable_hyde": enable_hyde,
                "enable_sub_questions": enable_sub_questions,
                "enable_multi_query": multi_query_used,
                "response_mode": self._settings.response_mode,
            },
        }

    def _build_retriever(
        self, index, collection: str, top_k: int,
        enable_hybrid: bool = False, alpha: float = 0.5,
    ):
        """Pick the best retriever: AutoMerging > Hybrid > Dense."""
        from core.ingestion.pipeline import get_docstore

        docstore = get_docstore(collection)
        if docstore is not None:
            return create_automerging_retriever(index, docstore, top_k=top_k)

        if enable_hybrid:
            return create_hybrid_retriever(index, top_k=top_k, alpha=alpha)

        return index.as_retriever(similarity_top_k=top_k)

    def _wrap_sub_question(self, base_engine, collection, response_synthesizer):
        """Wrap query engine with sub-question decomposition."""
        question_gen = create_sub_question_generator(self._llm)

        tool = QueryEngineTool(
            query_engine=base_engine,
            metadata=ToolMetadata(
                name=collection,
                description=f"Documents in the '{collection}' collection",
            ),
        )

        return SubQuestionQueryEngine.from_defaults(
            query_engine_tools=[tool],
            question_gen=question_gen,
            response_synthesizer=response_synthesizer,
            llm=self._llm,
            use_async=False,
        )
