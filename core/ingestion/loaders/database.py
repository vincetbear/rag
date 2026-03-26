"""Database loader — reads from SQL via connection string."""

from llama_index.core.schema import Document


def load_database(uri: str, query: str) -> list[Document]:
    from llama_index.readers.database import DatabaseReader

    reader = DatabaseReader(uri=uri)
    return reader.load_data(query=query)
