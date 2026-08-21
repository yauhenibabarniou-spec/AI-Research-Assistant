import csv
import json
import logging
from pathlib import Path
from typing import Any

from app.common.utils import chunk_id
from app.core.config import settings
from app.eval.golden_dataset import GoldenDataset
from app.eval.retrieval_metrics import RetrievalMetrics, hit_at_k, recall_at_k
from app.rag.embeddings import Embeddings
from app.rag.stores.chroma import ChromaStore
from app.rag.stores.hybrid import HybridStore
from app.harness.utils.storage import generate_run_id, ensure_output_dir, write_csv, write_json

logger = logging.getLogger(__name__)


def run(config: dict[str, Any], project_root: Path) -> None:
    retrieval_config = config.get("retrieval", {})
    global_config = config.get("global", {})

    k = retrieval_config.get("k", 3)
    score_threshold = retrieval_config.get("score_threshold", 0.3)
    search_type = retrieval_config.get("search_type", "weighted")
    alpha = retrieval_config.get("alpha", 0.6)
    output_dir = global_config.get("output_dir", "eval/reports")
    output_path = global_config.get("_retrieval_output")

    dataset = GoldenDataset(project_root / "eval" / "golden_dataset.json")

    embeddings = Embeddings()
    vector_manager = ChromaStore(
        persist_directory=str(project_root / "chroma_db"),
        embedding_function=embeddings.get_embeddings(),
        knowledge_base_dir=str(project_root / "knowledge_base"),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        collection_name="documents",
    )
    vector_manager.index_documents()

    hybrid_store = HybridStore.from_chroma_store(
        vector_manager,
        search_type=search_type,
        alpha=alpha,
    )

    metrics = RetrievalMetrics()
    rows = []
    for item in dataset.all():
        if item.source_doc == "multiple":
            continue
        results = hybrid_store.get_relevant_documents(
            item.question, k=k, score_threshold=score_threshold
        )
        retrieved_docs = [doc for doc, _ in results]
        metrics.add(item.expected_chunk_ids, retrieved_docs, k)
        rows.append(
            {
                "id": item.id,
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
    logger.info("Retrieval metrics: %s", summary)

    run_id = generate_run_id()
    output_dir_path = ensure_output_dir(project_root / output_dir, run_id)

    csv_path = Path(output_path) if output_path else output_dir_path / "retrieval_report.csv"
    if not Path(csv_path).is_absolute():
        csv_path = project_root / csv_path

    write_csv(
        rows + [{
            "id": "SUMMARY",
            "question": "",
            "expected_chunk_ids": "",
            "retrieved_chunk_ids": "",
            "hit": summary["hit_at_k"],
            "recall": summary["recall_at_k"],
        }],
        ["id", "question", "expected_chunk_ids", "retrieved_chunk_ids", "hit", "recall"],
        csv_path,
    )

    summary_path = csv_path.with_suffix(".json")
    write_json({"run_id": run_id, "summary": summary}, summary_path)

    print(json.dumps({"run_id": run_id, "summary": summary}, ensure_ascii=False, indent=2))
