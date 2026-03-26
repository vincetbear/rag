"""Test configuration loading."""

from config.settings import Settings


def test_default_settings():
    s = Settings(
        _env_file=None,  # don't load .env in tests
    )
    assert s.llm_provider == "groq"
    assert s.chunk_size == 512
    assert s.enable_hybrid is True
    assert s.port == 8000


def test_custom_settings():
    s = Settings(
        llm_provider="ollama",
        embedding_model="custom/model",
        chunk_size=1024,
        _env_file=None,
    )
    assert s.llm_provider == "ollama"
    assert s.embedding_model == "custom/model"
    assert s.chunk_size == 1024
