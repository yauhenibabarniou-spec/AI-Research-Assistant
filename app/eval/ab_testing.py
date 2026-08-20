import csv
import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from app.eval.golden_dataset import GoldenDataset, GoldenItem
from app.eval.retrieval_metrics import RetrievalMetrics, hit_at_k, recall_at_k
from app.rag.embeddings import Embeddings
from app.rag.loaders import DocumentLoader
from app.rag.splitters import TextSplitter
from app.rag.stores.chroma import ChromaStore

logger = logging.getLogger(__name__)


@dataclass
class ABConfig:
    name: str
    chunk_size: int = 800
    chunk_overlap: int = 120
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    k: int = 3
    score_threshold: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedding_model": self.embedding_model,
            "k": self.k,
            "score_threshold": self.score_threshold,
        }


@dataclass
class ABResult:
    config: ABConfig
    metrics: dict[str, float]
    error: str | None = None


class ABTester:
    """A/B тестирование конфигураций RAG."""

    def __init__(
        self,
        golden_dataset_path: str | Path = "eval/golden_dataset.json",
        persist_directory: str | Path = "chroma_db",
        knowledge_base_dir: str | Path = "knowledge_base",
    ) -> None:
        self.golden_dataset_path = Path(golden_dataset_path)
        self.persist_directory = Path(persist_directory)
        self.knowledge_base_dir = Path(knowledge_base_dir)
        self.dataset = GoldenDataset(self.golden_dataset_path)

    def _build_local_golden_dataset(self, config: ABConfig) -> list[GoldenItem]:
        loader = DocumentLoader(str(self.knowledge_base_dir))
        splitter = TextSplitter(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
        documents = loader.load_documents()
        chunks = splitter.split_documents(documents)

        chunk_map: dict[str, list[str]] = {}
        for chunk in chunks:
            src = chunk.metadata.get("source", "")
            h = hashlib.sha256(chunk.page_content.encode()).hexdigest()[:16]
            chunk_map.setdefault(src, []).append(f"{src}:{chunk.metadata.get('start_index', 0)}:{h}")

        items: list[GoldenItem] = []
        for item in self.dataset.all():
            if item.source_doc == "multiple":
                continue
            expected_src = None
            for full_src in chunk_map:
                if Path(full_src).name == item.source_doc:
                    expected_src = full_src
                    break
            copied = item.model_copy()
            if expected_src:
                copied.expected_chunk_ids = chunk_map[expected_src]
            items.append(copied)
        return items

    def run_config(self, config: ABConfig) -> ABResult:
        local_dataset = self._build_local_golden_dataset(config)
        embeddings = Embeddings(model_name=config.embedding_model)
        vector_manager = ChromaStore(
            persist_directory=str(self.persist_directory / config.name),
            embedding_function=embeddings.get_embeddings(),
            knowledge_base_dir=str(self.knowledge_base_dir),
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            collection_name="documents",
        )
        vector_manager.index_documents()

        metrics = RetrievalMetrics()
        for item in local_dataset:
            results = vector_manager.get_relevant_documents(
                item.question, k=config.k, score_threshold=config.score_threshold
            )
            retrieved_docs = [doc for doc, _ in results]
            metrics.add(item.expected_chunk_ids, retrieved_docs, config.k)

        return ABResult(config=config, metrics=metrics.summary())

    def run_suite(self, configs: list[ABConfig]) -> list[ABResult]:
        results: list[ABResult] = []
        for config in configs:
            try:
                result = self.run_config(config)
                results.append(result)
            except Exception as exc:
                logger.exception("Config %s failed", config.name)
                results.append(ABResult(config=config, metrics={}, error=str(exc)))
        return results

    def save_results(self, results: list[ABResult], output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "ab_results.csv"
        md_path = output_dir / "ab_results.md"
        json_path = output_dir / "ab_results.json"

        if results:
            fieldnames = [
                "name",
                "chunk_size",
                "chunk_overlap",
                "embedding_model",
                "k",
                "score_threshold",
                "hit_at_k",
                "recall_at_k",
                "mrr",
                "ndcg_at_k",
                "queries_evaluated",
                "error",
            ]
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for result in results:
                    row = result.config.to_dict()
                    row.update(result.metrics)
                    row["error"] = result.error or ""
                    writer.writerow(row)

            with open(md_path, "w", encoding="utf-8") as f:
                f.write(
                    "| Name | chunk_size | chunk_overlap | embedding_model | k | hit@k | recall@k | MRR | nDCG | error |\n"
                )
                f.write(
                    "|------|------------|---------------|-----------------|---|-------|----------|-----|------|-------|\n"
                )
                for result in results:
                    m = result.metrics
                    f.write(
                        f"| {result.config.name} | {result.config.chunk_size} | {result.config.chunk_overlap} | {result.config.embedding_model} | {result.config.k} | {m.get('hit_at_k', '-'):.4f} | {m.get('recall_at_k', '-'):.4f} | {m.get('mrr', '-'):.4f} | {m.get('ndcg_at_k', '-'):.4f} | {result.error or ''} |\n"
                    )

            with open(json_path, "w", encoding="utf-8") as f:
                payload = [
                    {
                        **result.config.to_dict(),
                        "metrics": result.metrics,
                        "error": result.error,
                    }
                    for result in results
                ]
                json.dump(payload, f, ensure_ascii=False, indent=2)

        return md_path
