import logging

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.common.utils import chunk_id
from app.rag.loaders import DocumentLoader
from app.rag.splitters import TextSplitter


class ChromaStore:
    """Управление векторным хранилищем Chroma."""

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        embedding_function=None,
        knowledge_base_dir: str = "./knowledge_base",
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        collection_name: str = "documents",
    ):
        self.logger = logging.getLogger(__name__)
        self.persist_directory = persist_directory
        self.knowledge_base_dir = knowledge_base_dir
        self.collection_name = collection_name

        self.loader = DocumentLoader(knowledge_base_dir)
        self.splitter = TextSplitter(chunk_size, chunk_overlap)

        # Инициализация векторного хранилища
        self._init_vectorstore(embedding_function)

    def _init_vectorstore(self, embedding_function):
        """Инициализация или пересоздание векторного хранилища."""
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=embedding_function,
            collection_name=self.collection_name,
        )


    def index_documents(self) -> int:
        """Индексация документов в векторном хранилище с чанкингом и идемпотентностью."""
        self.logger.info("Загрузка документов...")
        documents = self.loader.load_documents()

        if not documents:
            self.logger.info("Нет документов для индексации.")
            return 0

        self.logger.info(f"Найдено документов: {len(documents)}")

        # Чанкинг документов
        self.logger.info("Разделение документов на чанки...")
        all_chunks = self.splitter.split_documents(documents)

        self.logger.info(f"Создано чанков: {len(all_chunks)}")

        if not all_chunks:
            self.logger.warning("Нет чанков для индексации.")
            return 0

        # Очистка коллекции для идемпотентности
        try:
            self.vectorstore.delete_collection()
            self._init_vectorstore(self.vectorstore._embedding_function)
            self.logger.info("Коллекция очищена перед индексацией.")
        except Exception as e:
            self.logger.warning(f"Не удалось очистить коллекцию: {e}")

        # Генерация детерминированных ID
        chunk_ids = [chunk_id(chunk) for chunk in all_chunks]

        # Добавление чанков в векторное хранилище
        self.logger.info("Создание эмбеддингов и индексация...")
        self.vectorstore.add_documents(all_chunks, ids=chunk_ids)

        self.logger.info(f"Успешно проиндексировано {len(all_chunks)} чанков.")
        return len(all_chunks)

    def clear_index(self):
        """Очистка векторного хранилища."""
        self.vectorstore.delete_collection()
        self.logger.info("Векторное хранилище очищено.")
        self._init_vectorstore(self.vectorstore._embedding_function)

    def get_relevant_documents(
        self, query: str, k: int = 3, score_threshold: float = 0.3
    ) -> list[tuple[Document, float]]:
        """Поиск релевантных документов по запросу с порогом релевантности."""
        results = self.vectorstore.similarity_search_with_relevance_scores(query, k=k)
        # Фильтрация по порогу
        filtered = [(doc, score) for doc, score in results if score >= score_threshold]
        return filtered

    def get_document_count(self) -> int:
        """Получение количества документов в хранилище."""
        try:
            # ponytail: ChromaDB не имеет публичного API для count, приватный доступ необходим
            return self.vectorstore._collection.count()
        except Exception:
            return 0
