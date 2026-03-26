"""Seed Qdrant with sample data for testing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import get_settings, get_secrets
from core.embeddings.factory import create_embed_model
from core.indexing.qdrant_store import create_qdrant_client
from core.ingestion.pipeline import IngestionPipeline


def main():
    settings = get_settings()
    embed_model = create_embed_model(settings)
    qdrant_client = create_qdrant_client(settings)

    pipeline = IngestionPipeline(settings, embed_model, qdrant_client)

    # Ingest all files in data/ directory
    data_dir = Path(__file__).parent.parent / "data"
    if not data_dir.exists():
        print(f"No data directory found at {data_dir}")
        return

    for file_path in data_dir.iterdir():
        if file_path.suffix.lower() in (".pdf", ".docx", ".txt", ".md"):
            print(f"Ingesting {file_path.name}...")
            result = pipeline.ingest_file(file_path)
            print(f"  → {result}")

    print("Done!")


if __name__ == "__main__":
    main()
