import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def check_chromadb() -> bool:
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(Path("./chroma_db").resolve()))
        client.list_collections()
        return True
    except Exception as exc:
        logger.warning("ChromaDB health check failed: %s", exc)
        return False


def check_ollama(base_url: str = "http://localhost:11434") -> bool:
    try:
        import requests

        response = requests.get(f"{base_url}/api/tags", timeout=5)
        return response.status_code == 200
    except Exception as exc:
        logger.warning("Ollama health check failed: %s", exc)
        return False


def check_vector_store() -> bool:
    try:
        from app.rag.embeddings import Embeddings
        from app.rag.stores.chroma import ChromaStore
        from app.core.config import settings

        embeddings = Embeddings(settings.embedding_model)
        store = ChromaStore(
            persist_directory=settings.persist_directory,
            embedding_function=embeddings.get_embeddings(),
            knowledge_base_dir=settings.knowledge_base_dir,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            collection_name=settings.collection_name,
        )
        count = store.get_document_count()
        return count >= 0
    except Exception as exc:
        logger.warning("Vector store health check failed: %s", exc)
        return False
