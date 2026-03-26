"""Run benchmark queries and evaluate results."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import get_settings, get_secrets
from core.llm.factory import create_llm
from core.embeddings.factory import create_embed_model
from core.indexing.qdrant_store import create_qdrant_client
from core.retrieval.engine import RetrievalEngine
from evaluation.evaluator import RAGEvaluator

SAMPLE_QUERIES = [
    "What is the main topic of the document?",
    "Summarize the key findings.",
    "What are the conclusions?",
]


async def main():
    settings = get_settings()
    secrets = get_secrets()
    llm = create_llm(settings, secrets)
    embed_model = create_embed_model(settings)
    qdrant_client = create_qdrant_client(settings)

    engine = RetrievalEngine(settings, llm, embed_model, qdrant_client)
    evaluator = RAGEvaluator(llm)

    for query in SAMPLE_QUERIES:
        print(f"\nQuery: {query}")
        result = engine.query(query)
        print(f"Answer: {result['answer'][:200]}...")

        contexts = [s["text"] for s in result["sources"]]
        eval_result = await evaluator.evaluate(query, result["answer"], contexts)
        print(f"Faithfulness: {eval_result.faithfulness}")
        print(f"Relevancy: {eval_result.relevancy}")


if __name__ == "__main__":
    asyncio.run(main())
