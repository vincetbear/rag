import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException

from app.dependencies import (
    get_cached_settings,
    get_embed_model,
    get_qdrant_client,
)
from app.api.schemas.ingest import IngestResponse
from config.settings import Settings
from core.ingestion.pipeline import IngestionPipeline

router = APIRouter()


def _get_pipeline(
    settings: Settings = Depends(get_cached_settings),
    embed_model=Depends(get_embed_model),
    qdrant_client=Depends(get_qdrant_client),
) -> IngestionPipeline:
    return IngestionPipeline(settings, embed_model, qdrant_client)


@router.post("/ingest/file", response_model=IngestResponse, operation_id="ingest_file")
async def ingest_file(
    file: UploadFile = File(...),
    collection: str | None = Form(None),
    chunk_strategy: str = Form("sentence"),
    pipeline: IngestionPipeline = Depends(_get_pipeline),
):
    suffix = Path(file.filename or "upload.bin").suffix
    if suffix.lower() not in (".pdf", ".docx", ".doc", ".txt", ".md", ".csv", ".json"):
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    original_filename = file.filename or "upload" + suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = pipeline.ingest_file(
            tmp_path,
            collection=collection,
            chunk_strategy=chunk_strategy,
            filename=original_filename,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return IngestResponse(**result)
