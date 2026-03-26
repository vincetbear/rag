"""Response synthesizer factory."""

from llama_index.core.llms import LLM
from llama_index.core.response_synthesizers import get_response_synthesizer, BaseSynthesizer

from core.generation.prompts import QA_PROMPT, REFINE_PROMPT


def create_response_synthesizer(
    llm: LLM,
    mode: str = "tree_summarize",
) -> BaseSynthesizer:
    kwargs = {"llm": llm, "response_mode": mode}

    if mode == "refine":
        kwargs["text_qa_template"] = QA_PROMPT
        kwargs["refine_template"] = REFINE_PROMPT

    return get_response_synthesizer(**kwargs)
