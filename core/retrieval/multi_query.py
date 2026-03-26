"""Multi-query expansion: rewrite a query into variants for broader retrieval."""

from llama_index.core.llms import LLM
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle


MULTI_QUERY_PROMPT = (
    "You are an AI assistant that generates alternative search queries.\n"
    "Given the original query, generate 3 different versions that capture\n"
    "the same intent but use different wording or perspectives.\n"
    "Return ONLY the 3 queries, one per line, without numbering or bullets.\n\n"
    "Original query: {query}\n\n"
    "Alternative queries:"
)


def generate_query_variants(llm: LLM, query: str) -> list[str]:
    """Use LLM to generate 3 query variants from the original."""
    prompt = MULTI_QUERY_PROMPT.format(query=query)
    response = llm.complete(prompt)
    lines = [
        line.strip()
        for line in str(response).strip().splitlines()
        if line.strip()
    ]
    return lines[:3]


def retrieve_with_multi_query(
    retriever: BaseRetriever,
    llm: LLM,
    query: str,
) -> list[NodeWithScore]:
    """Retrieve using original + variant queries, then deduplicate by node ID."""
    variants = generate_query_variants(llm, query)
    all_queries = [query] + variants

    seen_ids: set[str] = set()
    merged: list[NodeWithScore] = []

    for q in all_queries:
        nodes = retriever.retrieve(QueryBundle(query_str=q))
        for node in nodes:
            node_id = node.node.node_id
            if node_id not in seen_ids:
                seen_ids.add(node_id)
                merged.append(node)

    # Sort by score descending so reranker gets the best candidates
    merged.sort(key=lambda n: n.score or 0.0, reverse=True)
    return merged
