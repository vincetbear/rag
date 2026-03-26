"""Web page loader."""

from llama_index.core.schema import Document


def load_url(url: str) -> list[Document]:
    from llama_index.readers.web import SimpleWebPageReader

    reader = SimpleWebPageReader(html_to_text=True)
    return reader.load_data([url])
