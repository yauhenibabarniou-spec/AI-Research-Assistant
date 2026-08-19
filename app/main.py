from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import register_routes
from app.core.config import settings
from app.rag.embeddings import Embeddings
from app.rag.stores.chroma import ChromaStore


def _check_ollama_availability(base_url: str) -> bool:
    """Проверка доступности Ollama."""
    try:
        import requests

        response = requests.get(f"{base_url}/api/tags", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan событий FastAPI для инициализации и очистки."""
    # Инициализация эмбеддингов
    embeddings = Embeddings(settings.embedding_model)

    # Инициализация векторного хранилища
    vector_manager = ChromaStore(
        persist_directory=settings.persist_directory,
        embedding_function=embeddings.get_embeddings(),
        knowledge_base_dir=settings.knowledge_base_dir,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        collection_name=settings.collection_name,
    )

    # Проверка доступности модели
    app.state.ollama_available = _check_ollama_availability(settings.ollama_base_url)
    app.state.vector_manager = vector_manager

    yield

    # Очистка при завершении работы
    if hasattr(app.state.vector_manager, "vectorstore"):
        app.state.vector_manager.vectorstore = None


app = FastAPI(lifespan=lifespan)

register_routes(app)
