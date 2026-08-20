import hashlib

from langchain_core.documents import Document

from app.eval.retrieval_metrics import (
    hit_at_k,
    mrr,
    ndcg_at_k,
    recall_at_k,
    RetrievalMetrics,
)


def _make_doc(source: str, start: int, content: str) -> Document:
    return Document(page_content=content, metadata={"source": source, "start_index": start})


def _expected_id(doc: Document) -> str:
    src = doc.metadata.get("source", "")
    start = doc.metadata.get("start_index", 0)
    h = hashlib.sha256(doc.page_content.encode()).hexdigest()[:16]
    return f"{src}:{start}:{h}"


def test_hit_at_k_hit() -> None:
    doc = _make_doc("kb/a.txt", 0, "alpha")
    expected = [_expected_id(doc)]
    assert hit_at_k(expected, [doc], 1) == 1.0


def test_hit_at_k_miss() -> None:
    doc = _make_doc("kb/b.txt", 0, "beta")
    other = _make_doc("kb/a.txt", 0, "alpha")
    expected = [_expected_id(other)]
    assert hit_at_k(expected, [doc], 1) == 0.0


def test_recall_at_k_partial() -> None:
    doc1 = _make_doc("kb/a.txt", 0, "alpha")
    doc2 = _make_doc("kb/b.txt", 0, "beta")
    doc3 = _make_doc("kb/c.txt", 0, "gamma")
    expected = [_expected_id(doc1), _expected_id(doc3)]
    assert recall_at_k(expected, [doc1, doc2], 2) == 0.5


def test_recall_at_k_empty_expected() -> None:
    doc = _make_doc("kb/a.txt", 0, "alpha")
    assert recall_at_k([], [doc], 1) == 0.0


def test_mrr_first_rank() -> None:
    doc = _make_doc("kb/a.txt", 0, "alpha")
    expected = [_expected_id(doc)]
    assert mrr(expected, [doc]) == 1.0


def test_mrr_second_rank() -> None:
    doc1 = _make_doc("kb/b.txt", 0, "beta")
    doc2 = _make_doc("kb/a.txt", 0, "alpha")
    expected = [_expected_id(doc2)]
    assert abs(mrr(expected, [doc1, doc2]) - 0.5) < 1e-9


def test_ndcg_at_k_perfect() -> None:
    doc = _make_doc("kb/a.txt", 0, "alpha")
    expected = [_expected_id(doc)]
    assert ndcg_at_k(expected, [doc], 1) == 1.0


def test_ndcg_at_k_zero() -> None:
    doc = _make_doc("kb/b.txt", 0, "beta")
    other = _make_doc("kb/a.txt", 0, "alpha")
    expected = [_expected_id(other)]
    assert ndcg_at_k(expected, [doc], 1) == 0.0


def test_aggregation() -> None:
    metrics = RetrievalMetrics()
    doc1 = _make_doc("kb/a.txt", 0, "alpha")
    metrics.add([_expected_id(doc1)], [doc1], 1)
    doc2 = _make_doc("kb/b.txt", 0, "beta")
    metrics.add([_expected_id(doc2)], [doc2], 1)
    summary = metrics.summary()
    assert summary["queries_evaluated"] == 2.0
    assert summary["hit_at_k"] == 1.0
    assert summary["recall_at_k"] == 1.0
