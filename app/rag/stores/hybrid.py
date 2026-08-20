import logging
from typing import Any

import rank_bm25
from langchain_core.documents import Document

from app.common.utils import chunk_id


STOPWORDS = {
    "что", "такое", "как", "для", "и", "в", "на", "с", "по", "о", "об", "от",
    "до", "из", "у", "к", "за", "под", "над", "перед", "через", "между",
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "i", "you", "he",
    "she", "it", "we", "they", "what", "which", "who", "when", "where",
    "why", "how", "all", "each", "every", "both", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same",
    "so", "than", "too", "very", "just", "because", "but", "and", "or",
    "if", "while", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up",
    "down", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all",
    "any", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than",
    "too", "very", "s", "t", "just", "don", "now",
}


class BM25Index:
    """BM25 index for keyword-based retrieval."""

    def __init__(self, documents: list[Document]):
        self.logger = logging.getLogger(__name__)
        self.documents = documents
        self.doc_ids = [chunk_id(doc) for doc in documents]
        
        # Tokenize documents for BM25
        self.corpus = [self._tokenize(doc.page_content) for doc in documents]
        self.bm25 = rank_bm25.BM25Okapi(self.corpus)

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization: lowercase, split on non-alphanumeric, remove stopwords."""
        import re
        tokens = re.findall(r"\w+", text.lower())
        return [token for token in tokens if token not in STOPWORDS and len(token) > 1]

    def search(self, query: str, k: int = 10) -> list[tuple[Document, float]]:
        """Search using BM25 scores."""
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k results
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self.documents[idx], float(scores[idx])))
        
        return results


class HybridStore:
    """Hybrid search combining BM25 and vector search."""

    def __init__(
        self,
        chroma_store: Any,
        bm25_index: BM25Index | None = None,
        search_type: str = "weighted",
        bm25_k: int = 20,
        vector_k: int = 20,
        rerank_k: int = 5,
        rrf_k: int = 60,
        alpha: float = 0.6,
    ):
        self.logger = logging.getLogger(__name__)
        self.chroma_store = chroma_store
        self.bm25_index = bm25_index
        self.search_type = search_type
        self.bm25_k = bm25_k
        self.vector_k = vector_k
        self.rerank_k = rerank_k
        self.rrf_k = rrf_k
        self.alpha = alpha

    @classmethod
    def from_chroma_store(cls, chroma_store: Any, **kwargs) -> "HybridStore":
        """Create HybridStore from existing ChromaStore, building BM25 index."""
        # Retrieve all documents from ChromaDB for BM25 indexing
        documents = cls._get_all_documents(chroma_store)
        bm25_index = BM25Index(documents) if documents else None
        
        instance = cls(chroma_store=chroma_store, bm25_index=bm25_index, **kwargs)
        
        if bm25_index:
            instance.logger.info(
                f"Hybrid search initialized with {len(documents)} documents "
                f"(type={instance.search_type}, bm25_k={instance.bm25_k}, "
                f"vector_k={instance.vector_k}, rrf_k={instance.rrf_k}, rerank_k={instance.rerank_k})"
            )
        else:
            instance.logger.warning("No documents found for BM25 indexing, hybrid search disabled")
        
        return instance

    @staticmethod
    def _get_all_documents(chroma_store: Any) -> list[Document]:
        """Retrieve all documents from ChromaDB for BM25 indexing."""
        try:
            # Access the underlying collection to get all documents
            collection = chroma_store.vectorstore._collection
            result = collection.get(include=["documents", "metadatas"])
            
            documents = []
            if result and "documents" in result:
                for i, doc_text in enumerate(result["documents"]):
                    if doc_text:
                        metadata = {}
                        if result.get("metadatas") and result["metadatas"][i]:
                            metadata = result["metadatas"][i]
                        documents.append(Document(page_content=doc_text, metadata=metadata))
            
            return documents
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to get all documents: {e}")
            return []

    def get_relevant_documents(
        self, query: str, k: int = 3, score_threshold: float = 0.3
    ) -> list[tuple[Document, float]]:
        """Hybrid search: BM25 + vector using configured strategy."""
        
        if not self.bm25_index:
            self.logger.warning("BM25 index not available, falling back to vector search")
            return self.chroma_store.get_relevant_documents(query, k=k, score_threshold=score_threshold)

        if self.search_type == "rrf":
            return self._search_rrf(query, k=k, score_threshold=score_threshold)
        elif self.search_type == "two_stage":
            return self._search_two_stage(query, k=k, score_threshold=score_threshold)
        elif self.search_type == "weighted":
            return self._search_weighted(query, k=k, score_threshold=score_threshold)
        else:
            self.logger.warning(f"Unknown search type: {self.search_type}, falling back to RRF")
            return self._search_rrf(query, k=k, score_threshold=score_threshold)

    def _search_rrf(
        self, query: str, k: int = 3, score_threshold: float = 0.3
    ) -> list[tuple[Document, float]]:
        """Reciprocal Rank Fusion: combine BM25 and vector rankings."""
        # Get BM25 results
        bm25_results = self.bm25_index.search(query, k=self.bm25_k)
        
        # Get vector results
        vector_results = self.chroma_store.get_relevant_documents(
            query, k=self.vector_k, score_threshold=0.0
        )
        
        if not bm25_results and not vector_results:
            self.logger.warning("Both BM25 and vector returned no results, returning empty")
            return []
        
        # Build RRF scores, deduplicating by content hash
        rrf_scores: dict[str, tuple[Document, float]] = {}
        
        # Add BM25 contributions
        for rank, (doc, _) in enumerate(bm25_results, start=1):
            doc_id = chunk_id(doc)
            rrf_score = 1.0 / (self.rrf_k + rank)
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = (doc, rrf_score)
            else:
                rrf_scores[doc_id] = (doc, rrf_scores[doc_id][1] + rrf_score)
        
        # Add vector contributions
        for rank, (doc, _) in enumerate(vector_results, start=1):
            doc_id = chunk_id(doc)
            rrf_score = 1.0 / (self.rrf_k + rank)
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = (doc, rrf_score)
            else:
                rrf_scores[doc_id] = (doc, rrf_scores[doc_id][1] + rrf_score)
        
        # Sort by RRF score descending
        sorted_results = sorted(rrf_scores.values(), key=lambda x: x[1], reverse=True)
        
        # Apply score threshold
        filtered = [(doc, score) for doc, score in sorted_results if score >= score_threshold]
        
        # If threshold filtering removes too many results, return top-k unfiltered
        if not filtered and sorted_results:
            filtered = sorted_results[:k]
        
        return filtered[:k]

    def _search_two_stage(
        self, query: str, k: int = 3, score_threshold: float = 0.3
    ) -> list[tuple[Document, float]]:
        """Two-stage search: BM25 retrieval + vector reranking."""
        # Stage 1: BM25 retrieval
        bm25_results = self.bm25_index.search(query, k=self.bm25_k)
        
        if not bm25_results:
            self.logger.warning("BM25 returned no results, falling back to vector search")
            return self.chroma_store.get_relevant_documents(query, k=k, score_threshold=score_threshold)

        # Stage 2: Vector reranking of BM25 candidates
        bm25_docs = [doc for doc, _ in bm25_results]
        reranked = self._rerank_with_vector(query, bm25_docs, k=self.rerank_k)
        
        # Apply score threshold
        filtered = [(doc, score) for doc, score in reranked if score >= score_threshold]
        
        # If threshold filtering removes too many results, return top-k unfiltered
        if not filtered and reranked:
            filtered = reranked[:k]
        
        return filtered[:k]

    def _search_weighted(
        self, query: str, k: int = 3, score_threshold: float = 0.3
    ) -> list[tuple[Document, float]]:
        """Weighted fusion: alpha * normalized BM25 + (1-alpha) * vector score."""
        # Get BM25 results with more candidates
        bm25_results = self.bm25_index.search(query, k=self.bm25_k)
        
        # Get vector results
        vector_results = self.chroma_store.get_relevant_documents(
            query, k=self.vector_k, score_threshold=0.0
        )
        
        if not bm25_results and not vector_results:
            return []
        
        # Normalize BM25 scores to [0, 1]
        bm25_scores = [score for _, score in bm25_results]
        bm25_max = max(bm25_scores) if bm25_scores else 1.0
        bm25_min = min(bm25_scores) if bm25_scores else 0.0
        bm25_range = bm25_max - bm25_min if bm25_max != bm25_min else 1.0
        
        # Build weighted scores, deduplicating by content hash
        weighted_scores: dict[str, tuple[Document, float]] = {}
        
        # Add BM25 contributions
        for doc, score in bm25_results:
            doc_id = chunk_id(doc)
            normalized_bm25 = (score - bm25_min) / bm25_range
            if doc_id not in weighted_scores or self.alpha * normalized_bm25 > weighted_scores[doc_id][1]:
                weighted_scores[doc_id] = (doc, self.alpha * normalized_bm25)
        
        # Add vector contributions
        for doc, score in vector_results:
            doc_id = chunk_id(doc)
            vector_score = score  # Already in [0, 1] range from ChromaDB
            if doc_id not in weighted_scores:
                weighted_scores[doc_id] = (doc, (1 - self.alpha) * vector_score)
            else:
                combined = weighted_scores[doc_id][1] + (1 - self.alpha) * vector_score
                weighted_scores[doc_id] = (doc, combined)
        
        # Sort by weighted score descending
        sorted_results = sorted(weighted_scores.values(), key=lambda x: x[1], reverse=True)
        
        # Apply score threshold
        filtered = [(doc, score) for doc, score in sorted_results if score >= score_threshold]
        
        if not filtered and sorted_results:
            filtered = sorted_results[:k]
        
        return filtered[:k]

    def _rerank_with_vector(
        self, query: str, documents: list[Document], k: int = 5
    ) -> list[tuple[Document, float]]:
        """Rerank documents using vector similarity scores."""
        # Use vectorstore to get relevance scores for reranking
        try:
            # Get the embedding function from the vectorstore
            embedding_function = self.chroma_store.vectorstore._embedding_function
            
            # Embed query
            query_embedding = embedding_function.embed_query(query)
            
            # Compute similarity scores for each document
            scored_docs = []
            for doc in documents:
                doc_embedding = embedding_function.embed_documents([doc.page_content])[0]
                # Cosine similarity
                score = self._cosine_similarity(query_embedding, doc_embedding)
                scored_docs.append((doc, score))
            
            # Sort by similarity score descending
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            return scored_docs[:k]
            
        except Exception as e:
            self.logger.warning(f"Vector reranking failed: {e}, using BM25 order")
            # Fallback: just return BM25 results with normalized scores
            return [(doc, score / 100.0) for doc, score in documents[:k]]

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        import math
        
        dot_product = sum(x * y for x, y in zip(a, b))
        magnitude_a = math.sqrt(sum(x * x for x in a))
        magnitude_b = math.sqrt(sum(x * x for x in b))
        
        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0
        
        return dot_product / (magnitude_a * magnitude_b)

    def get_document_count(self) -> int:
        """Get document count from underlying vector store."""
        return self.chroma_store.get_document_count()