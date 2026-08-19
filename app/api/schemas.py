from pydantic import BaseModel


class ReindexRequest(BaseModel):
    """Запрос на переиндексацию документов."""

    clear_first: bool = False


class QueryRequest(BaseModel):
    """Запрос на поиск ответа."""

    question: str
    k: int = 4


class QueryResponse(BaseModel):
    """Ответ на запрос пользователя."""

    answer: str
    sources: list[str]
    model_used: str
