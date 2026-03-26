from pydantic import BaseModel


class IngestResponse(BaseModel):
    status: str
    documents: int = 0
    nodes: int = 0
    collection: str = ""
    chunk_strategy: str = ""


class IngestURLRequest(BaseModel):
    url: str
    collection: str | None = None
    chunk_strategy: str = "sentence"
