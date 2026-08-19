import hashlib
import logging
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredHTMLLoader,
)
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class VectorStoreManager:
    """Управление векторным хранилищем документов."""

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        knowledge_base_dir: str = "./knowledge_base",
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        collection_name: str = "documents",
    ):
        self.logger = logging.getLogger(__name__)
        self.persist_directory = persist_directory
        self.knowledge_base_dir = Path(knowledge_base_dir)
        self.collection_name = collection_name

        # Инициализация эмбеддингов (локальная модель, бесплатно)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # Текстовый сплиттер для чанкинга
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            add_start_index=True,
        )

        # Инициализация векторного хранилища
        self._init_vectorstore()

    def _init_vectorstore(self):
        """Инициализация или пересоздание векторного хранилища."""
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name=self.collection_name,
        )

    def load_documents(self) -> list[Document]:
        """Загрузка документов из папки knowledge_base."""
        documents = []

        if not self.knowledge_base_dir.exists():
            self.logger.info(f"Папка {self.knowledge_base_dir} не найдена. Создаю...")
            self.knowledge_base_dir.mkdir(parents=True, exist_ok=True)
            return documents

        # Поддерживаемые расширения файлов
        supported_extensions = {".pdf", ".txt", ".md", ".docx", ".html", ".htm"}

        for file_path in self.knowledge_base_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                try:
                    loader_map = {
                        ".pdf": lambda p: PyPDFLoader(str(p)),
                        ".txt": lambda p: TextLoader(str(p), encoding="utf-8"),
                        ".md": lambda p: UnstructuredMarkdownLoader(str(p)),
                        ".docx": lambda p: UnstructuredWordDocumentLoader(str(p)),
                        ".html": lambda p: UnstructuredHTMLLoader(str(p)),
                        ".htm": lambda p: UnstructuredHTMLLoader(str(p)),
                    }
                    suffix = file_path.suffix.lower()
                    if loader := loader_map.get(suffix):
                        documents.extend(loader(file_path).load())
                    else:
                        self.logger.warning(f"Пропущен файл {file_path.name} (не поддерживаемый формат)")
                except Exception as e:
                    self.logger.error(f"Ошибка при загрузке файла {file_path.name}: {str(e)}")

        return documents

    def _chunk_id(self, chunk: Document) -> str:
        """Генерация детерминированного ID для чанка."""
        src = chunk.metadata.get("source", "")
        start = chunk.metadata.get("start_index", 0)
        h = hashlib.sha256(chunk.page_content.encode()).hexdigest()[:16]
        return f"{src}:{start}:{h}"

    def index_documents(self) -> int:
        """Индексация документов в векторном хранилище с чанкингом и идемпотентностью."""
        self.logger.info("Загрузка документов...")
        documents = self.load_documents()

        if not documents:
            self.logger.info("Нет документов для индексации.")
            return 0

        self.logger.info(f"Найдено документов: {len(documents)}")

        # Чанкинг документов
        self.logger.info("Разделение документов на чанки...")
        all_chunks = []
        for doc in documents:
            chunks = self.text_splitter.split_documents([doc])
            all_chunks.extend(chunks)

        self.logger.info(f"Создано чанков: {len(all_chunks)}")

        if not all_chunks:
            self.logger.warning("Нет чанков для индексации.")
            return 0

        # Генерация детерминированных ID
        chunk_ids = [self._chunk_id(chunk) for chunk in all_chunks]

        # Добавление чанков в векторное хранилище
        self.logger.info("Создание эмбеддингов и индексация...")
        self.vectorstore.add_documents(all_chunks, ids=chunk_ids)

        self.logger.info(f"Успешно проиндексировано {len(all_chunks)} чанков.")
        return len(all_chunks)

    def clear_index(self):
        """Очистка векторного хранилища."""
        self.vectorstore.delete_collection()
        self.logger.info("Векторное хранилище очищено.")
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name=self.collection_name,
        )

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


if __name__ == "__main__":
    # Пример использования
    logging.basicConfig(level=logging.INFO)
    manager = VectorStoreManager()
    count = manager.index_documents()
    logging.info(f"Всего документов в хранилище: {manager.get_document_count()}")
