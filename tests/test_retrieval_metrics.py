import hashlib
import os

import pytest
from langchain_core.documents import Document

from app.eval.retrieval_metrics import (
    hit_at_k,
    mrr,
    ndcg_at_k,
    recall_at_k,
    RetrievalMetrics,
)

from tests.conftest import make_document


def _expected_id(doc: Document) -> str:
    src = doc.metadata.get("source", "")
    start = doc.metadata.get("start_index", 0)
    if src and not os.path.isabs(src):
        src = os.path.abspath(src)
    h = hashlib.sha256(doc.page_content.encode()).hexdigest()[:16]
    return f"{src}:{start}:{h}"


def test_hit_at_k_hit() -> None:
    doc = make_document("alpha", source="kb/a.txt", start_index=0)
    expected = [_expected_id(doc)]
    assert hit_at_k(expected, [doc], 1) == 1.0


def test_hit_at_k_miss() -> None:
    doc = make_document("beta", source="kb/b.txt", start_index=0)
    other = make_document("alpha", source="kb/a.txt", start_index=0)
    expected = [_expected_id(other)]
    assert hit_at_k(expected, [doc], 1) == 0.0


def test_recall_at_k_partial() -> None:
    doc1 = make_document("alpha", source="kb/a.txt", start_index=0)
    doc2 = make_document("beta", source="kb/b.txt", start_index=0)
    doc3 = make_document("gamma", source="kb/c.txt", start_index=0)
    expected = [_expected_id(doc1), _expected_id(doc3)]
    assert recall_at_k(expected, [doc1, doc2], 2) == 0.5


def test_recall_at_k_empty_expected() -> None:
    doc = make_document("alpha", source="kb/a.txt", start_index=0)
    assert recall_at_k([], [doc], 1) == 0.0


def test_mrr_first_rank() -> None:
    doc = make_document("alpha", source="kb/a.txt", start_index=0)
    expected = [_expected_id(doc)]
    assert mrr(expected, [doc]) == 1.0


def test_mrr_second_rank() -> None:
    doc1 = make_document("beta", source="kb/b.txt", start_index=0)
    doc2 = make_document("alpha", source="kb/a.txt", start_index=0)
    expected = [_expected_id(doc2)]
    assert abs(mrr(expected, [doc1, doc2]) - 0.5) < 1e-9


def test_ndcg_at_k_perfect() -> None:
    doc = make_document("alpha", source="kb/a.txt", start_index=0)
    expected = [_expected_id(doc)]
    assert ndcg_at_k(expected, [doc], 1) == 1.0


def test_ndcg_at_k_zero() -> None:
    doc = make_document("beta", source="kb/b.txt", start_index=0)
    other = make_document("alpha", source="kb/a.txt", start_index=0)
    expected = [_expected_id(other)]
    assert ndcg_at_k(expected, [doc], 1) == 0.0


def test_aggregation() -> None:
    metrics = RetrievalMetrics()
    doc1 = make_document("alpha", source="kb/a.txt", start_index=0)
    metrics.add([_expected_id(doc1)], [doc1], 1)
    doc2 = make_document("beta", source="kb/b.txt", start_index=0)
    metrics.add([_expected_id(doc2)], [doc2], 1)
    summary = metrics.summary()
    assert summary["queries_evaluated"] == 2.0
    assert summary["hit_at_k"] == 1.0
    assert summary["recall_at_k"] == 1.0
