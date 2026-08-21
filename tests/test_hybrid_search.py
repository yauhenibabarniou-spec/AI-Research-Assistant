import logging
import math

import pytest
from langchain_core.documents import Document

from app.rag.stores.hybrid import BM25Index, HybridStore


class TestBM25Index:
    """Tests for BM25Index."""

    def test_index_creation(self, sample_documents):
        """Test BM25 index creation."""
        index = BM25Index(sample_documents)
        assert len(index.documents) == 4
        assert len(index.doc_ids) == 4

    def test_search_returns_results(self, sample_documents):
        """Test that search returns results."""
        index = BM25Index(sample_documents)
        results = index.search("ChromaDB vector", k=2)
        assert len(results) > 0
        assert all(isinstance(r, tuple) for r in results)

    def test_search_relevance(self, sample_documents):
        """Test that search returns relevant results first."""
        index = BM25Index(sample_documents)
        results = index.search("ChromaDB", k=2)
        assert len(results) >= 1
        first_doc, _ = results[0]
        assert "ChromaDB" in first_doc.page_content

    def test_search_k_parameter(self, sample_documents):
        """Test that k parameter limits results."""
        index = BM25Index(sample_documents)
        results = index.search("vector database", k=1)
        assert len(results) == 1

    def test_search_no_results(self, sample_documents):
        """Test search with no matching terms."""
        index = BM25Index(sample_documents)
        results = index.search("zzzznonexistent", k=2)
        assert len(results) == 0


class TestHybridStore:
    """Tests for HybridStore."""

    def test_from_chroma_store_initialization(self, sample_documents, caplog):
        """Test HybridStore initialization from ChromaStore."""
        from app.rag.stores.hybrid import HybridStore

        class MockVectorStore:
            def __init__(self):
                self._embedding_function = lambda x: [0.1] * 384
                self._collection = MockCollection(sample_documents)

        class MockCollection:
            def __init__(self, docs):
                self.docs = docs

            def get(self, include=None):
                return {
                    "documents": [d.page_content for d in self.docs],
                    "metadatas": [d.metadata for d in self.docs],
                }

        class MockChromaStore:
            def __init__(self):
                self.vectorstore = MockVectorStore()

            def get_document_count(self):
                return len(sample_documents)

        mock_chroma = MockChromaStore()

        with caplog.at_level(logging.INFO):
            hybrid = HybridStore.from_chroma_store(mock_chroma)

        assert hybrid.bm25_index is not None
        assert len(hybrid.bm25_index.documents) == 4
        assert "Hybrid search initialized" in caplog.text

    def test_get_relevant_documents_fallback(self, sample_documents):
        """Test fallback to vector search when BM25 index is missing."""
        from app.rag.stores.hybrid import HybridStore

        class MockVectorStore:
            def __init__(self):
                self._embedding_function = lambda x: [0.1] * 384
                self._collection = None

        class MockChromaStore:
            def __init__(self):
                self.vectorstore = MockVectorStore()

            def get_document_count(self):
                return 0

            def get_relevant_documents(self, query, k=3, score_threshold=0.3):
                return [(sample_documents[0], 0.9)]

        mock_chroma = MockChromaStore()
        hybrid = HybridStore(chroma_store=mock_chroma, bm25_index=None)

        results = hybrid.get_relevant_documents("test query", k=2)
        assert len(results) == 1
        assert results[0][0].page_content == "ChromaDB is a vector database for embeddings"

    def test_rrf_search_returns_results(self, sample_documents, caplog):
        """Test RRF search returns results."""
        from app.rag.stores.hybrid import HybridStore

        class MockVectorStore:
            def __init__(self):
                self._embedding_function = MockEmbeddingFunction()
                self._collection = MockCollection(sample_documents)

        class MockCollection:
            def __init__(self, docs):
                self.docs = docs

            def get(self, include=None):
                return {
                    "documents": [d.page_content for d in self.docs],
                    "metadatas": [d.metadata for d in self.docs],
                }

        class MockEmbeddingFunction:
            def embed_query(self, text):
                return [0.1] * 384

            def embed_documents(self, texts):
                return [[0.1] * 384 for _ in texts]

        class MockChromaStore:
            def __init__(self):
                self.vectorstore = MockVectorStore()

            def get_document_count(self):
                return len(sample_documents)

            def get_relevant_documents(self, query, k=3, score_threshold=0.3):
                return [
                    (sample_documents[0], 0.9),
                    (sample_documents[1], 0.7),
                    (sample_documents[2], 0.5),
                ][:k]

        mock_chroma = MockChromaStore()
        hybrid = HybridStore.from_chroma_store(mock_chroma, search_type="rrf", bm25_k=10, vector_k=10)

        results = hybrid.get_relevant_documents("ChromaDB vector", k=3)
        assert len(results) > 0
        assert all(isinstance(r, tuple) for r in results)
        for _, score in results:
            assert score > 0
            assert not math.isinf(score)

    def test_two_stage_search_returns_results(self, sample_documents, caplog):
        """Test two-stage search returns results."""
        from app.rag.stores.hybrid import HybridStore

        class MockVectorStore:
            def __init__(self):
                self._embedding_function = MockEmbeddingFunction()
                self._collection = MockCollection(sample_documents)

        class MockCollection:
            def __init__(self, docs):
                self.docs = docs

            def get(self, include=None):
                return {
                    "documents": [d.page_content for d in self.docs],
                    "metadatas": [d.metadata for d in self.docs],
                }

        class MockEmbeddingFunction:
            def embed_query(self, text):
                return [0.1] * 384

            def embed_documents(self, texts):
                return [[0.1] * 384 for _ in texts]

        class MockChromaStore:
            def __init__(self):
                self.vectorstore = MockVectorStore()

            def get_document_count(self):
                return len(sample_documents)

            def get_relevant_documents(self, query, k=3, score_threshold=0.3):
                return [
                    (sample_documents[0], 0.9),
                    (sample_documents[1], 0.7),
                ][:k]

        mock_chroma = MockChromaStore()
        hybrid = HybridStore.from_chroma_store(mock_chroma, search_type="two_stage", bm25_k=10, rerank_k=5)

        results = hybrid.get_relevant_documents("ChromaDB vector", k=3)
        assert len(results) > 0
        assert all(isinstance(r, tuple) for r in results)

    def test_weighted_search_returns_results(self, sample_documents, caplog):
        """Test weighted search returns results."""
        from app.rag.stores.hybrid import HybridStore

        class MockVectorStore:
            def __init__(self):
                self._embedding_function = MockEmbeddingFunction()
                self._collection = MockCollection(sample_documents)

        class MockCollection:
            def __init__(self, docs):
                self.docs = docs

            def get(self, include=None):
                return {
                    "documents": [d.page_content for d in self.docs],
                    "metadatas": [d.metadata for d in self.docs],
                }

        class MockEmbeddingFunction:
            def embed_query(self, text):
                return [0.1] * 384

            def embed_documents(self, texts):
                return [[0.1] * 384 for _ in texts]

        class MockChromaStore:
            def __init__(self):
                self.vectorstore = MockVectorStore()

            def get_document_count(self):
                return len(sample_documents)

            def get_relevant_documents(self, query, k=3, score_threshold=0.3):
                return [
                    (sample_documents[0], 0.9),
                    (sample_documents[1], 0.7),
                ][:k]

        mock_chroma = MockChromaStore()
        hybrid = HybridStore.from_chroma_store(
            mock_chroma, search_type="weighted", bm25_k=10, vector_k=10, alpha=0.7
        )

        results = hybrid.get_relevant_documents("ChromaDB vector", k=3)
        assert len(results) > 0
        assert all(isinstance(r, tuple) for r in results)
        for _, score in results:
            assert 0.0 <= score <= 1.0