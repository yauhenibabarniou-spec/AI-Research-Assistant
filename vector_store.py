import os
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma


class VectorStoreManager:
    """Управление векторным хранилищем документов."""

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        knowledge_base_dir: str = "./knowledge_base",
    ):
        self.persist_directory = persist_directory
        self.knowledge_base_dir = Path(knowledge_base_dir)
        
        # Инициализация эмбеддингов (локальная модель, бесплатно)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # Инициализация векторного хранилища
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings,
        )

    def load_documents(self) -> List[Document]:
        """Загрузка документов из папки knowledge_base."""
        documents = []
        
        if not self.knowledge_base_dir.exists():
            print(f"Папка {self.knowledge_base_dir} не найдена. Создаю...")
            self.knowledge_base_dir.mkdir(parents=True, exist_ok=True)
            return documents

        for file_path in self.knowledge_base_dir.iterdir():
            if file_path.suffix == ".pdf":
                loader = PyPDFLoader(str(file_path))
                documents.extend(loader.load())
            elif file_path.suffix == ".txt":
                loader = TextLoader(str(file_path), encoding="utf-8")
                documents.extend(loader.load())
            else:
                print(f"Пропущен файл {file_path.name} (не поддерживаемый формат)")

        return documents

    def index_documents(self) -> int:
        """Индексация документов в векторном хранилище."""
        print("Загрузка документов...")
        documents = self.load_documents()

        if not documents:
            print("Нет документов для индексации.")
            return 0

        print(f"Найдено документов: {len(documents)}")
        print("Создание эмбеддингов и индексация...")

        # Добавление документов в векторное хранилище
        self.vectorstore.add_documents(documents)

        print(f"Успешно проиндексировано {len(documents)} документов.")
        return len(documents)

    def clear_index(self):
        """Очистка векторного хранилища."""
        self.vectorstore.delete_collection()
        print("Векторное хранилище очищено.")

    def get_relevant_documents(
        self, query: str, k: int = 3
    ) -> List[Document]:
        """Поиск релевантных документов по запросу."""
        return self.vectorstore.similarity_search(query, k=k)

    def get_document_count(self) -> int:
        """Получение количества документов в хранилище."""
        try:
            return self.vectorstore._collection.count()
        except Exception:
            return 0


if __name__ == "__main__":
    # Пример использования
    manager = VectorStoreManager()
    count = manager.index_documents()
    print(f"\nВсего документов в хранилище: {manager.get_document_count()}")
