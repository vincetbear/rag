"""Custom prompt templates for response synthesis."""

from llama_index.core.prompts import PromptTemplate

QA_PROMPT_TMPL = (
    "You are a strict question-answering assistant.\n"
    "Use ONLY the context information provided below to answer the query.\n"
    "Do NOT add any information from your own knowledge.\n"
    "Do NOT speculate, infer, or supplement beyond what is explicitly stated in the context.\n"
    "If the context does not contain enough information to answer, "
    'respond exactly: "根據提供的資料，無法回答此問題。"\n\n'
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n\n"
    "Query: {query_str}\n"
    "Answer: "
)

QA_PROMPT = PromptTemplate(QA_PROMPT_TMPL)

REFINE_PROMPT_TMPL = (
    "The original query is: {query_str}\n"
    "We have an existing answer: {existing_answer}\n\n"
    "Below is additional context that MAY help refine the answer.\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n\n"
    "Rules:\n"
    "- ONLY use information explicitly present in the context above or the existing answer.\n"
    "- Do NOT add any external knowledge or assumptions.\n"
    "- If the new context is not relevant, return the existing answer unchanged.\n"
    "- If the new context provides useful details, incorporate them into the answer.\n"
    "Refined Answer: "
)

REFINE_PROMPT = PromptTemplate(REFINE_PROMPT_TMPL)
