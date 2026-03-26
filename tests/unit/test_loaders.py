"""Test document loaders."""

import tempfile
from pathlib import Path

import pytest

from core.ingestion.loaders import load_file


def test_load_text_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("Hello world. This is a test document.")
        f.flush()
        docs = load_file(f.name)

    assert len(docs) == 1
    assert "Hello world" in docs[0].text
    assert docs[0].metadata["file_type"] == "txt"
    Path(f.name).unlink()


def test_load_empty_text_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("")
        f.flush()
        docs = load_file(f.name)

    assert len(docs) == 0
    Path(f.name).unlink()


def test_load_unsupported_type():
    with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
        f.write(b"data")
        f.flush()

    with pytest.raises(ValueError, match="Unsupported file type"):
        load_file(f.name)

    Path(f.name).unlink()
