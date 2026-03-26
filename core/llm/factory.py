from llama_index.core.llms import LLM

from config.settings import Settings, SecretsSettings


def create_llm(settings: Settings, secrets: SecretsSettings) -> LLM:
    provider = settings.llm_provider.lower()

    if provider == "groq":
        from llama_index.llms.groq import Groq

        return Groq(
            model=settings.llm_model,
            api_key=secrets.groq_api_key,
            temperature=settings.llm_temperature,
            context_window=settings.llm_context_window,
        )

    if provider == "ollama":
        from llama_index.llms.ollama import Ollama

        return Ollama(
            model=settings.llm_model,
            base_url=settings.llm_base_url or "http://localhost:11434",
            temperature=settings.llm_temperature,
            request_timeout=120.0,
            context_window=settings.llm_context_window,
        )

    if provider == "vllm":
        from llama_index.llms.ollama import Ollama

        # vLLM exposes an OpenAI-compatible API; Ollama adapter works with base_url override
        return Ollama(
            model=settings.llm_model,
            base_url=settings.llm_base_url,
            temperature=settings.llm_temperature,
            request_timeout=120.0,
            context_window=settings.llm_context_window,
        )

    raise ValueError(f"Unknown LLM provider: {provider}")
