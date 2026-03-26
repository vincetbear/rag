from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    llm_provider: str = "groq"
    llm_model: str = "llama-3.3-70b-versatile"
    llm_base_url: str = ""
    llm_temperature: float = 0.1
    llm_context_window: int = 131072

    # Embedding
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_device: str = "cpu"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    default_collection: str = "default"

    # Retrieval
    top_k: int = 15
    rerank_top_n: int = 5
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    enable_hybrid: bool = True
    hybrid_alpha: float = 0.5
    enable_hyde: bool = False
    response_mode: str = "refine"

    # Chunking
    chunk_size: int = 1024
    chunk_overlap: int = 128

    # Cache
    model_cache_dir: str = ""
    fastembed_cache_dir: str = ""

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"


# GROQ_API_KEY is read without prefix
class SecretsSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    groq_api_key: str = ""


def get_settings() -> Settings:
    return Settings()


def get_secrets() -> SecretsSettings:
    return SecretsSettings()
