"""Metadata filtering utilities."""

from llama_index.core.vector_stores.types import (
    MetadataFilters,
    MetadataFilter,
    FilterOperator,
)


def build_metadata_filters(filter_dict: dict | None) -> MetadataFilters | None:
    if not filter_dict:
        return None

    filters = []
    for key, value in filter_dict.items():
        filters.append(
            MetadataFilter(key=key, value=value, operator=FilterOperator.EQ)
        )

    return MetadataFilters(filters=filters)
