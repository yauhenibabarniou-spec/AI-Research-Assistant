import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from langchain.chains import create_retrieval_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel
from pydantic_settings import BaseSettings as PydanticBaseSettings
from pydantic_settings import SettingsConfigDict

from vector_store import VectorStoreManager


class Settings(PydanticBaseSettings):
    """Настройки приложения."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Параметры векторного хранилища
    persist_directory: str = "./chroma_db"
    knowledge_base_dir: str = "./knowledge_base"
    chunk_size: int = 800
    chunk_overlap: int = 120
    collection_name: str = "documents"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Параметры модели
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    temperature: float = 0.7

    # Параметры API
    host: str = "0.0.0.0"
    port: int = 8000

    # Параметры CORS
    allowed_origins: list[str] = ["*"]

    # Параметры кэширования
    max_context_length: int = 4000

    # Параметры безопасности
    api_key: str | None = None

    # Параметры Rate Limiting
    rate_limit: int = 100  # запросов в минуту


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


# Загрузка настроек
settings = Settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan событий FastAPI для инициализации и очистки."""
    # Инициализация менеджеров
    vector_manager = VectorStoreManager(
        persist_directory=settings.persist_directory,
        embedding_model=settings.embedding_model,
        knowledge_base_dir=settings.knowledge_base_dir,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        collection_name=settings.collection_name,
    )

    # Проверка доступности модели
    if not _check_ollama_availability(settings.ollama_base_url):
        app.state.ollama_available = False
    else:
        app.state.ollama_available = True

    app.state.vector_manager = vector_manager

    yield

    # Очистка при завершении работы
    if hasattr(app.state.vector_manager, "vectorstore"):
        app.state.vector_manager.vectorstore = None


app = FastAPI(lifespan=lifespan)


def _check_ollama_availability(base_url: str) -> bool:
    """Проверка доступности Ollama."""
    try:
        import requests

        response = requests.get(f"{base_url}/api/tags", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def get_llm():
    """Получение LLM модели в зависимости от настроек."""

    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2")

    # Возвращаем конфиг для ленивой инициализации
    return {
        "type": "ollama",
        "base_url": ollama_url,
        "model": ollama_model,
        "temperature": 0.7,
    }


def create_rag_chain(llm):
    """Создание RAG цепочки."""
    # Промпт для генерации ответа
    template = """Ты - помощник, который отвечает на вопросы только на основе предоставленных документов.
    
Контекст из документов:
{context}

Вопрос пользователя: {question}

Если ответ не может быть найден в документах, честно скажи об этом.
Ответ должен быть точным и полезным.

Ответ:"""

    prompt = ChatPromptTemplate.from_template(template)

    # Если llm это dict (Ollama конфиг), создаём модель лениво
    if isinstance(llm, dict):
        llm_instance = ChatOllama(
            base_url=llm["base_url"],
            model=llm["model"],
            temperature=llm["temperature"],
        )
    else:
        llm_instance = llm

    # Цепочка: контекст + вопрос → промпт → LLM → ответ
    chain = prompt | llm_instance | StrOutputParser()

    return chain


@app.get("/")
def root():
    """Информация о сервисе."""
    return {
        "message": "AI Research Assistant API",
        "documents_indexed": app.state.vector_manager.get_document_count(),
        "endpoints": {
            "query": "POST /query - задать вопрос",
            "documents": "GET /documents - список документов",
            "reindex": "POST /reindex - переиндексировать",
        },
    }


@app.get("/documents")
def get_documents_info():
    """Информация о загруженных документах."""
    count = app.state.vector_manager.get_document_count()
    return {
        "document_count": count,
        "knowledge_base_dir": "./knowledge_base",
    }


@app.post("/reindex")
def reindex_documents(request: ReindexRequest):
    """Переиндексация документов."""
    try:
        if request.clear_first:
            app.state.vector_manager.clear_index()

        count = app.state.vector_manager.index_documents()
        return {
            "success": True,
            "documents_indexed": count,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """Поиск ответа на вопрос пользователя."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Вопрос не может быть пустым")

    try:
        # Поиск релевантных документов
        relevant_docs = app.state.vector_manager.get_relevant_documents(request.question, k=request.k)

        if not relevant_docs:
            return QueryResponse(
                answer="К сожалению, я не нашёл релевантных документов для ответа на ваш вопрос. Попробуйте загрузить документы в папку knowledge_base/ и выполнить переиндексацию.",
                sources=[],
                model_used="no_context",
            )

        # Формирование контекста из документов
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        sources = list(set([doc.metadata.get("source", "unknown") for doc in relevant_docs]))

        # Получение LLM и создание цепочки
        llm = get_llm()
        chain = create_rag_chain(llm)

        # Генерация ответа
        answer = chain.invoke(
            {
                "context": context,
                "question": request.question,
            }
        )

        # Определение использованной модели
        model_used = os.getenv("OLLAMA_MODEL", "llama3.2")

        return QueryResponse(
            answer=answer,
            sources=sources,
            model_used=model_used,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке запроса: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
