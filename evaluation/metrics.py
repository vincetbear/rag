"""Evaluation metrics."""

from dataclasses import dataclass


@dataclass
class EvalResult:
    query: str
    response: str
    faithfulness: float | None = None
    relevancy: float | None = None
    correctness: float | None = None
