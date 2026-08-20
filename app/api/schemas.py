from pydantic import BaseModel, Field


class ReindexRequest(BaseModel):
    """Запрос на переиндексацию документов."""

    clear_first: bool = False


class QueryRequest(BaseModel):
    """Запрос на поиск ответа."""

    question: str
    k: int = 4
    search_type: str = Field(default="vector", pattern="^(vector|hybrid)$")
    score_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    hybrid_search_type: str = Field(default="weighted", pattern="^(rrf|two_stage|weighted)$")
    hybrid_bm25_k: int = Field(default=20, ge=1, le=100)
    hybrid_vector_k: int = Field(default=20, ge=1, le=100)
    hybrid_rrf_k: int = Field(default=60, ge=1, le=200)


class QueryResponse(BaseModel):
    """Ответ на запрос пользователя."""

    answer: str
    sources: list[str]
    model_used: str
