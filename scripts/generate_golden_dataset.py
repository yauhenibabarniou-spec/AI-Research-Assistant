#!/usr/bin/env python3
"""Generate deterministic chunk IDs for golden dataset."""

import hashlib
import json
import logging
from pathlib import Path

from langchain_core.documents import Document

from app.rag.loaders import DocumentLoader
from app.rag.splitters import TextSplitter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def chunk_id(chunk: Document) -> str:
    src = chunk.metadata.get("source", "")
    start = chunk.metadata.get("start_index", 0)
    h = hashlib.sha256(chunk.page_content.encode()).hexdigest()[:16]
    return f"{src}:{start}:{h}"


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    dataset_path = project_root / "eval" / "golden_dataset.json"
    knowledge_base_dir = project_root / "knowledge_base"

    loader = DocumentLoader(str(knowledge_base_dir))
    splitter = TextSplitter(chunk_size=800, chunk_overlap=120)

    documents = loader.load_documents()
    if not documents:
        raise RuntimeError("No documents loaded from knowledge_base")

    chunks = splitter.split_documents(documents)

    chunk_map: dict[str, list[str]] = {}
    for chunk in chunks:
        src = chunk.metadata.get("source", "")
        chunk_map[src] = chunk_map.get(src, []) + [chunk_id(chunk)]

    logger.info("Indexed %d chunks across %d sources", len(chunks), len(chunk_map))

    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)

    updated = 0
    for item in dataset:
        source = item.get("source_doc")
        if source == "multiple":
            continue
        expected_src = None
        for full_src in chunk_map:
            if Path(full_src).name == source:
                expected_src = full_src
                break
        if expected_src:
            item["expected_chunk_ids"] = chunk_map[expected_src]
            updated += 1
        else:
            logger.warning("No chunks found for source: %s", source)

    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    logger.info("Updated expected_chunk_ids for %d items", updated)


if __name__ == "__main__":
    main()
