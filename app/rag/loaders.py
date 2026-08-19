import logging
from pathlib import Path
from typing import Callable

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredHTMLLoader,
)
from langchain_core.documents import Document


class DocumentLoader:
    """Загрузка документов из различных форматов."""

    def __init__(self, knowledge_base_dir: str):
        self.logger = logging.getLogger(__name__)
        self.knowledge_base_dir = Path(knowledge_base_dir)

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
                    loader_map: dict[str, Callable[[Path], object]] = {
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
