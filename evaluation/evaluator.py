"""RAG evaluation pipeline using LlamaIndex evaluators."""

from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    CorrectnessEvaluator,
)
from llama_index.core.llms import LLM

from evaluation.metrics import EvalResult


class RAGEvaluator:
    def __init__(self, llm: LLM):
        self._faithfulness = FaithfulnessEvaluator(llm=llm)
        self._relevancy = RelevancyEvaluator(llm=llm)
        self._correctness = CorrectnessEvaluator(llm=llm)

    async def evaluate(
        self,
        query: str,
        response: str,
        contexts: list[str],
        reference: str | None = None,
    ) -> EvalResult:
        result = EvalResult(query=query, response=response)

        # Faithfulness: is the response grounded in contexts?
        faith_result = await self._faithfulness.aevaluate(
            query=query, response=response, contexts=contexts
        )
        result.faithfulness = faith_result.score

        # Relevancy: is the response relevant to the query?
        rel_result = await self._relevancy.aevaluate(
            query=query, response=response, contexts=contexts
        )
        result.relevancy = rel_result.score

        # Correctness: if reference answer provided
        if reference:
            corr_result = await self._correctness.aevaluate(
                query=query, response=response, reference=reference
            )
            result.correctness = corr_result.score

        return result
