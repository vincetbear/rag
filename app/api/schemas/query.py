from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Coerce mode allows string inputs like "10" → 10, "true" → True (for MCP clients)."""
    model_config = {"coerce_numbers_to_str": False, "strict": False}

    query: str
    collection: str | None = None
    top_k: int | None = None
    rerank_top_n: int | None = None
    enable_hyde: bool = False
    enable_sub_questions: bool = False
    enable_multi_query: bool = False
    filters: dict | None = None


class SourceNode(BaseModel):
    text: str
    score: float = 0.0
    metadata: dict = Field(default_factory=dict)


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceNode] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
