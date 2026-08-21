import json
import logging
import re
from pathlib import Path
from typing import Any

from app.eval.golden_dataset import GoldenDataset
from app.rag.chains import create_rag_chain
from app.rag.embeddings import Embeddings
from app.rag.stores.chroma import ChromaStore

logger = logging.getLogger(__name__)


FAITHFULNESS_PROMPT = """
Is every claim in the ANSWER directly supported by the CONTEXT? Respond with only 1.0 (yes), 0.5 (partially), or 0.0 (no).

CONTEXT:
{context}

ANSWER:
{answer}
"""

ANSWER_RELEVANCY_PROMPT = """
Does the ANSWER directly address the QUESTION? Respond with only 1.0 (yes), 0.5 (partially), or 0.0 (no).

QUESTION:
{question}

ANSWER:
{answer}
"""


class LLMJudge:
    """LLM-as-Judge для оценки генерации."""

    def __init__(self, llm_config: dict[str, Any]) -> None:
        self.llm_config = llm_config
        self.chain = create_rag_chain(llm_config)

    def _extract_score(self, text: str) -> float:
        text = text.strip()
        if text == "1.0":
            return 1.0
        if text == "0.5":
            return 0.5
        if text == "0.0":
            return 0.0
        matches = re.findall(r"0\.\d+|1\.0|0\.0", text)
        if matches:
            return float(matches[0])
        return 0.0

    def faithfulness(self, context: str, answer: str) -> float:
        prompt = FAITHFULNESS_PROMPT.format(context=context[:2000], answer=answer[:1000])
        try:
            raw = self.chain.invoke({"context": prompt, "question": ""})
            return self._extract_score(raw)
        except Exception as exc:
            logger.warning("Faithfulness judge failed: %s", exc)
            return 0.0

    def answer_relevancy(self, question: str, answer: str) -> float:
        prompt = ANSWER_RELEVANCY_PROMPT.format(question=question, answer=answer[:1000])
        try:
            raw = self.chain.invoke({"context": "", "question": prompt})
            return self._extract_score(raw)
        except Exception as exc:
            logger.warning("Answer relevancy judge failed: %s", exc)
            return 0.0


def get_llm_config() -> dict[str, Any]:
    from app.core.config import settings

    return {
        "type": "ollama",
        "base_url": settings.ollama_base_url,
        "model": settings.ollama_model,
        "temperature": 0.0,
    }


def run_generation_eval(
    k: int = 3,
    score_threshold: float = 0.0,
    output_path: str = "eval/reports/generation_report.json",
    limit: int | None = 5,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> dict[str, Any]:
    from app.core.config import settings

    project_root = Path(__file__).resolve().parents[2]
    dataset = GoldenDataset(project_root / "eval" / "golden_dataset.json")
    embeddings = Embeddings()
    _chunk_size = chunk_size if chunk_size is not None else settings.chunk_size
    _chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
    vector_manager = ChromaStore(
        persist_directory=str(project_root / "chroma_db"),
        embedding_function=embeddings.get_embeddings(),
        knowledge_base_dir=str(project_root / "knowledge_base"),
        chunk_size=_chunk_size,
        chunk_overlap=_chunk_overlap,
        collection_name="documents",
    )
    vector_manager.index_documents()
    judge = LLMJudge(get_llm_config())

    items = dataset.all()
    if limit is not None:
        items = items[:limit]

    results: list[dict[str, Any]] = []
    faithfulness_scores: list[float] = []
    relevancy_scores: list[float] = []

    for item in items:
        if item.source_doc == "multiple":
            continue
        retrieved = vector_manager.get_relevant_documents(item.question, k=k, score_threshold=score_threshold)
        if not retrieved:
            results.append(
                {
                    "id": item.id,
                    "question": item.question,
                    "answer": "NO_CONTEXT",
                    "faithfulness": 0.0,
                    "answer_relevancy": 0.0,
                }
            )
            faithfulness_scores.append(0.0)
            relevancy_scores.append(0.0)
            continue

        context = "\n\n".join(doc.page_content for doc, _ in retrieved)
        llm_config = get_llm_config()
        chain = create_rag_chain(llm_config)
        answer = chain.invoke({"context": context, "question": item.question})

        faithfulness = judge.faithfulness(context, answer)
        relevancy = judge.answer_relevancy(item.question, answer)

        results.append(
            {
                "id": item.id,
                "question": item.question,
                "answer": answer,
                "faithfulness": faithfulness,
                "answer_relevancy": relevancy,
            }
        )
        faithfulness_scores.append(faithfulness)
        relevancy_scores.append(relevancy)

    summary = {
        "queries_evaluated": len(results),
        "avg_faithfulness": sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0.0,
        "avg_answer_relevancy": sum(relevancy_scores) / len(relevancy_scores) if relevancy_scores else 0.0,
    }

    output_path_obj = project_root / output_path
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path_obj, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)

    logger.info("Generation eval summary: %s", summary)
    return summary
