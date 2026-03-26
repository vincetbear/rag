"""Query transformations: HyDE, sub-question decomposition."""

from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.llms import LLM
from llama_index.core.output_parsers.base import OutputParserException
from llama_index.core.question_gen import LLMQuestionGenerator
from llama_index.core.question_gen.output_parser import SubQuestionOutputParser
from llama_index.core.question_gen.prompts import build_tools_text
from llama_index.core.prompts import PromptTemplate
from llama_index.core.tools import QueryEngineTool, ToolMetadata


def create_hyde_transform(llm: LLM) -> HyDEQueryTransform:
    return HyDEQueryTransform(llm=llm, include_original=True)


# Strict prompt that prevents LLM from adding extra text after JSON
_SUB_QUESTION_PROMPT_TMPL = """\
Given a user question, and a list of tools, output a list of relevant sub-questions \
in JSON format that can be answered using the tools.

Output ONLY a valid JSON object, nothing else. No explanation, no markdown fences, no extra text.

The JSON format must be:
{{"items": [{{"sub_question": "...", "tool_name": "..."}}]}}

Available tools:
{tools_str}

User question: {query_str}
"""


def create_sub_question_generator(llm: LLM) -> LLMQuestionGenerator:
    return LLMQuestionGenerator.from_defaults(
        llm=llm,
        prompt_template_str=_SUB_QUESTION_PROMPT_TMPL,
    )


def build_query_engine_tools(
    query_engines: dict[str, object],
) -> list[QueryEngineTool]:
    """Wrap multiple query engines as tools for sub-question decomposition."""
    tools = []
    for name, engine in query_engines.items():
        tools.append(
            QueryEngineTool(
                query_engine=engine,
                metadata=ToolMetadata(name=name, description=f"Query engine for {name}"),
            )
        )
    return tools
