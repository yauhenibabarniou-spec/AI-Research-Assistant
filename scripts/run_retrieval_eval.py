#!/usr/bin/env python3
"""Run retrieval evaluation against golden dataset."""

import argparse
import csv
import json
import logging
from pathlib import Path

from app.eval.golden_dataset import GoldenDataset
from app.eval.retrieval_metrics import RetrievalMetrics, hit_at_k, recall_at_k
from app.rag.embeddings import Embeddings
from app.rag.stores.chroma import ChromaStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval evaluation")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--output", default="eval/reports/retrieval_report.csv")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    dataset = GoldenDataset(project_root / "eval" / "golden_dataset.json")

    embeddings = Embeddings()
    vector_manager = ChromaStore(
        persist_directory=str(project_root / "chroma_db"),
        embedding_function=embeddings.get_embeddings(),
        knowledge_base_dir=str(project_root / "knowledge_base"),
        chunk_size=800,
        chunk_overlap=120,
        collection_name="documents",
    )
    vector_manager.index_documents()

    metrics = RetrievalMetrics()
    rows = []
    for item in dataset.all():
        if item.source_doc == "multiple":
            continue
        results = vector_manager.get_relevant_documents(
            item.question, k=args.k, score_threshold=args.score_threshold
        )
        retrieved_docs = [doc for doc, _ in results]
        metrics.add(item.expected_chunk_ids, retrieved_docs, args.k)
        rows.append(
            {
                "id": item.id,
                "question": item.question,
                "expected_chunk_ids": "|".join(item.expected_chunk_ids),
                "retrieved_chunk_ids": "|".join(
                    [
                        f"{doc.metadata.get('source', '')}:{doc.metadata.get('start_index', 0)}:{__import__('hashlib').sha256(doc.page_content.encode()).hexdigest()[:16]}"
                        for doc, _ in results
                    ]
                ),
                "hit": hit_at_k(
                    item.expected_chunk_ids,
                    retrieved_docs,
                    args.k,
                ),
                "recall": recall_at_k(
                    item.expected_chunk_ids,
                    retrieved_docs,
                    args.k,
                ),
            }
        )

    summary = metrics.summary()
    logger.info("Metrics: %s", summary)

    output_path = project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "question", "expected_chunk_ids", "retrieved_chunk_ids", "hit", "recall"],
        )
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow({"id": "SUMMARY", "question": "", "expected_chunk_ids": "", "retrieved_chunk_ids": "", "hit": summary["hit_at_k"], "recall": summary["recall_at_k"]})

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
