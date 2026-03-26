"""Document loaders — returns list[Document]."""

from pathlib import Path

from llama_index.core.schema import Document


def load_file(file_path: str | Path) -> list[Document]:
    """Load a file based on its extension."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _load_pdf(path)
    if suffix in (".docx", ".doc"):
        return _load_docx(path)
    if suffix in (".txt", ".md", ".csv", ".json"):
        return _load_text(path)

    raise ValueError(f"Unsupported file type: {suffix}")


def _load_pdf(path: Path) -> list[Document]:
    import fitz  # pymupdf

    doc = fitz.open(str(path))
    documents = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            documents.append(
                Document(
                    text=text,
                    metadata={"source": path.name, "page": i + 1, "file_type": "pdf"},
                )
            )
    doc.close()
    return documents


def _load_docx(path: Path) -> list[Document]:
    import docx2txt

    text = docx2txt.process(str(path))
    if not text.strip():
        return []
    return [Document(text=text, metadata={"source": path.name, "file_type": "docx"})]


def _load_text(path: Path) -> list[Document]:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []
    return [
        Document(
            text=text,
            metadata={"source": path.name, "file_type": path.suffix.lstrip(".")},
        )
    ]
