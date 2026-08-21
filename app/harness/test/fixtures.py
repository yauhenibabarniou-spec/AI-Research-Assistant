"""Shared test fixtures for harness tests."""

from typing import Any

from langchain_core.documents import Document


def make_document(
    content: str,
    source: str = "test.txt",
    start_index: int = 0,
) -> Document:
    """Create a test document with deterministic metadata."""
    return Document(page_content=content, metadata={"source": source, "start_index": start_index})


def make_golden_item(
    id: str,
    question: str,
    expected_answer: str,
    expected_chunk_ids: list[str],
    source_doc: str = "test.txt",
    difficulty: str = "easy",
    category: str = "general",
) -> dict[str, Any]:
    """Create a golden dataset item for testing."""
    return {
        "id": id,
        "question": question,
        "expected_answer": expected_answer,
        "expected_chunk_ids": expected_chunk_ids,
        "source_doc": source_doc,
        "difficulty": difficulty,
        "category": category,
    }


def mock_llm_response(*args: Any, **kwargs: Any) -> str:
    """Mock LLM response for testing."""
    return "1.0"


def mock_chain_invoke(prompt: str) -> str:
    """Mock chain invocation for testing."""
    return "Mock answer based on context."
