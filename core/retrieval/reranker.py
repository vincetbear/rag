"""Cross-encoder reranking postprocessor."""

from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.postprocessor.types import BaseNodePostprocessor


def create_reranker(model: str, top_n: int = 5) -> BaseNodePostprocessor:
    return SentenceTransformerRerank(
        model=model,
        top_n=top_n,
    )
