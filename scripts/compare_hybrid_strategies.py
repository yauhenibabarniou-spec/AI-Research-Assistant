import argparse
import csv
import json
import logging
from pathlib import Path

from app.common.utils import chunk_id
from app.eval.golden_dataset import GoldenDataset
from app.eval.retrieval_metrics import RetrievalMetrics, hit_at_k, recall_at_k
from app.rag.embeddings import Embeddings
from app.rag.stores.chroma import ChromaStore
from app.rag.stores.hybrid import HybridStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_strategy(
    name: str,
    store,
    dataset,
    k: int,
    score_threshold: float,
) -> tuple[dict, list[dict]]:
    """Evaluate a search strategy and return metrics and rows."""
    metrics = RetrievalMetrics()
    rows = []

    for item in dataset.all():
        if item.source_doc == "multiple":
            continue
        results = store.get_relevant_documents(
            item.question,
            k=k,
            score_threshold=score_threshold,
        )
        retrieved_docs = [doc for doc, _ in results]
        metrics.add(item.expected_chunk_ids, retrieved_docs, k)
        rows.append(
            {
                "id": item.id,
                "strategy": name,
                "question": item.question,
                "expected_chunk_ids": "|".join(item.expected_chunk_ids),
                "retrieved_chunk_ids": "|".join([chunk_id(doc) for doc, _ in results]),
                "hit": hit_at_k(
                    item.expected_chunk_ids,
                    retrieved_docs,
                    k,
                ),
                "recall": recall_at_k(
                    item.expected_chunk_ids,
                    retrieved_docs,
                    k,
                ),
            }
        )

    summary = metrics.summary()
    summary["strategy"] = name
    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare hybrid search strategies")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--output", default="eval/reports/hybrid_comparison_report.csv")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    dataset = GoldenDataset(project_root / "eval" / "golden_dataset.json")

    embeddings = Embeddings()
    chroma_store = ChromaStore(
        persist_directory=str(project_root / "chroma_db"),
        embedding_function=embeddings.get_embeddings(),
        knowledge_base_dir=str(project_root / "knowledge_base"),
        chunk_size=800,
        chunk_overlap=120,
        collection_name="documents",
    )
    chroma_store.index_documents()

    # Define strategies to compare
    strategies = {
        "vector": None,  # Will use chroma_store directly
        "hybrid_rrf": {"search_type": "rrf", "bm25_k": 20, "vector_k": 20, "rrf_k": 60},
        "hybrid_two_stage": {"search_type": "two_stage", "bm25_k": 10, "rerank_k": 5},
        "hybrid_weighted": {"search_type": "weighted", "bm25_k": 20, "vector_k": 20, "alpha": 0.7},
    }

    all_summaries = []
    all_rows = []

    # Evaluate vector baseline
    logger.info("Evaluating vector search baseline...")
    metrics = RetrievalMetrics()
    for item in dataset.all():
        if item.source_doc == "multiple":
            continue
        results = chroma_store.get_relevant_documents(item.question, k=args.k, score_threshold=args.score_threshold)
        retrieved_docs = [doc for doc, _ in results]
        metrics.add(item.expected_chunk_ids, retrieved_docs, args.k)
        all_rows.append(
            {
                "id": item.id,
                "strategy": "vector",
                "question": item.question,
                "expected_chunk_ids": "|".join(item.expected_chunk_ids),
                "retrieved_chunk_ids": "|".join([chunk_id(doc) for doc, _ in results]),
                "hit": hit_at_k(item.expected_chunk_ids, retrieved_docs, args.k),
                "recall": recall_at_k(item.expected_chunk_ids, retrieved_docs, args.k),
            }
        )
    vector_summary = metrics.summary()
    vector_summary["strategy"] = "vector"
    all_summaries.append(vector_summary)

    # Evaluate hybrid strategies
    for name, config in strategies.items():
        if name == "vector":
            continue

        logger.info(f"Evaluating {name}...")
        hybrid_store = HybridStore.from_chroma_store(chroma_store, **config)
        summary, rows = evaluate_strategy(name, hybrid_store, dataset, args.k, args.score_threshold)
        all_summaries.append(summary)
        all_rows.extend(rows)

    # Print comparison
    logger.info("=== Strategy Comparison ===")
    for summary in all_summaries:
        logger.info(
            f"{summary['strategy']}: hit@k={summary['hit_at_k']:.4f}, "
            f"recall@k={summary['recall_at_k']:.4f}, "
            f"MRR={summary['mrr']:.4f}, nDCG={summary['ndcg_at_k']:.4f}"
        )

    # Save detailed report
    output_path = project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "strategy",
                "question",
                "expected_chunk_ids",
                "retrieved_chunk_ids",
                "hit",
                "recall",
            ],
        )
        writer.writeheader()
        writer.writerows(all_rows)
        for summary in all_summaries:
            writer.writerow(
                {
                    "id": f"SUMMARY_{summary['strategy'].upper()}",
                    "strategy": summary["strategy"],
                    "question": "",
                    "expected_chunk_ids": "",
                    "retrieved_chunk_ids": "",
                    "hit": summary["hit_at_k"],
                    "recall": summary["recall_at_k"],
                }
            )

    # Save summary JSON
    summary_path = output_path.with_suffix(".json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)

    print(json.dumps(all_summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
