"""Test metadata filter building."""

from core.retrieval.filters import build_metadata_filters


def test_build_filters_none():
    assert build_metadata_filters(None) is None
    assert build_metadata_filters({}) is None


def test_build_filters():
    filters = build_metadata_filters({"source": "test.pdf", "page": 1})
    assert filters is not None
    assert len(filters.filters) == 2
