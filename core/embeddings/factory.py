import os

from llama_index.core.embeddings import BaseEmbedding

from config.settings import Settings


def _get_cache_dir(settings: Settings) -> str:
    if settings.model_cache_dir:
        return settings.model_cache_dir
    return os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")


def create_embed_model(settings: Settings) -> BaseEmbedding:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    return HuggingFaceEmbedding(
        model_name=settings.embedding_model,
        device=settings.embedding_device,
        trust_remote_code=True,
        cache_folder=_get_cache_dir(settings),
    )
