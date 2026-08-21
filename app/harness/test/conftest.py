import logging
from pathlib import Path

import pytest
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def chroma_db_dir(tmp_path_factory):
    """Session-scoped temporary ChromaDB directory."""
    return tmp_path_factory.mktemp("chroma_db")


@pytest.fixture(scope="function")
def sample_documents():
    """Function-scoped sample documents for testing."""
    return [
        Document(page_content="ChromaDB is a vector database for embeddings", metadata={"source": "chromadb.txt", "start_index": 0}),
        Document(page_content="FastAPI is a modern web framework for Python", metadata={"source": "fastapi.txt", "start_index": 100}),
        Document(page_content="LangChain is a framework for building LLM applications", metadata={"source": "langchain.txt", "start_index": 200}),
        Document(page_content="Python is a programming language", metadata={"source": "python.txt", "start_index": 300}),
    ]


@pytest.fixture(scope="function")
def golden_dataset_path(tmp_path):
    """Function-scoped temporary golden dataset path."""
    dataset = [
        {
            "id": "test-1",
            "question": "What is ChromaDB?",
            "expected_answer": "A vector database",
            "expected_chunk_ids": ["kb/chromadb.txt:0:abc123"],
            "source_doc": "chromadb.txt",
            "difficulty": "easy",
            "category": "general",
        }
    ]
    path = tmp_path / "golden_dataset.json"
    import json
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dataset, f)
    return path


@pytest.fixture(scope="session", autouse=True)
def _cleanup_session():
    """Session-scoped cleanup hook."""
    yield
    logger.debug("Test session cleanup complete")
