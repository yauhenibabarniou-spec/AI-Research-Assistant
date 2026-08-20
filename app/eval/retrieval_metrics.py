import math
from collections.abc import Sequence

from app.common.utils import chunk_id as _chunk_id
from langchain_core.documents import Document


def hit_at_k(expected: list[str], retrieved: Sequence[Document], k: int) -> float:
    top_k = retrieved[:k]
    retrieved_ids = {_chunk_id(doc) for doc in top_k}
    return 1.0 if any(chunk_id in retrieved_ids for chunk_id in expected) else 0.0


def recall_at_k(expected: list[str], retrieved: Sequence[Document], k: int) -> float:
    if not expected:
        return 0.0
    top_k = retrieved[:k]
    retrieved_ids = {_chunk_id(doc) for doc in top_k}
    hits = sum(1 for chunk_id in expected if chunk_id in retrieved_ids)
    return hits / len(expected)


def mrr(expected: list[str], retrieved: Sequence[Document]) -> float:
    for rank, doc in enumerate(retrieved, start=1):
        if _chunk_id(doc) in expected:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(expected: list[str], retrieved: Sequence[Document], k: int) -> float:
    top_k = retrieved[:k]
    dcg = 0.0
    for rank, doc in enumerate(top_k, start=1):
        if _chunk_id(doc) in expected:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_order = min(len(expected), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_order + 1))

    if idcg == 0:
        return 0.0
    return dcg / idcg


class RetrievalMetrics:
    """Агрегированные retrieval-метрики по датасету."""

    def __init__(self) -> None:
        self.hit_scores: list[float] = []
        self.recall_scores: list[float] = []
        self.mrr_scores: list[float] = []
        self.ndcg_scores: list[float] = []

    def add(self, expected: list[str], retrieved: Sequence[Document], k: int) -> None:
        self.hit_scores.append(hit_at_k(expected, retrieved, k))
        self.recall_scores.append(recall_at_k(expected, retrieved, k))
        self.mrr_scores.append(mrr(expected, retrieved))
        self.ndcg_scores.append(ndcg_at_k(expected, retrieved, k))

    def summary(self) -> dict[str, float]:
        n = len(self.hit_scores)
        if n == 0:
            return {}
        return {
            "hit_at_k": sum(self.hit_scores) / n,
            "recall_at_k": sum(self.recall_scores) / n,
            "mrr": sum(self.mrr_scores) / n,
            "ndcg_at_k": sum(self.ndcg_scores) / n,
            "queries_evaluated": float(n),
        }
