"""Dependency injection for FastAPI."""

from functools import lru_cache

from qdrant_client import QdrantClient
from llama_index.core.llms import LLM
from llama_index.core.embeddings import BaseEmbedding

from config.settings import Settings, SecretsSettings, get_settings, get_secrets
from core.llm.factory import create_llm
from core.embeddings.factory import create_embed_model
from core.indexing.qdrant_store import create_qdrant_client
from core.indexing.index_manager import IndexManager


@lru_cache
def get_cached_settings() -> Settings:
    return get_settings()


@lru_cache
def get_cached_secrets() -> SecretsSettings:
    return get_secrets()


@lru_cache
def get_qdrant_client() -> QdrantClient:
    return create_qdrant_client(get_cached_settings())


@lru_cache
def get_index_manager() -> IndexManager:
    return IndexManager(get_qdrant_client())


@lru_cache
def get_llm() -> LLM:
    return create_llm(get_cached_settings(), get_cached_secrets())


@lru_cache
def get_embed_model() -> BaseEmbedding:
    return create_embed_model(get_cached_settings())
