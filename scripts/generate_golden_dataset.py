import json
import logging
from pathlib import Path

from app.common.utils import chunk_id
from app.rag.embeddings import Embeddings
from app.rag.loaders import DocumentLoader
from app.rag.splitters import TextSplitter
from app.rag.stores.chroma import ChromaStore
from langchain_ollama import ChatOllama

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_question_for_chunk(llm, chunk_text: str, doc_name: str) -> str:
    """Generate a question that the given chunk can answer."""
    prompt = f"""На основе следующего фрагмента документа "{doc_name}" сгенерируй один конкретный вопрос, на который этот фрагмент содержит ответ. Вопрос должен быть на русском языке и требовать точного факта из текста.

Фрагмент:
{chunk_text[:800]}

Вопрос:"""

    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        logger.warning(f"Failed to generate question: {e}")
        return ""


def get_relevant_chunk_ids(store, question: str, k: int = 5) -> list[str]:
    """Get chunk IDs for documents relevant to the question."""
    results = store.get_relevant_documents(question, k=k, score_threshold=0.0)
    return [chunk_id(doc) for doc, _ in results]


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    golden_dataset_path = project_root / "eval" / "golden_dataset.json"

    # Load existing golden dataset
    with open(golden_dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Initialize components
    embeddings = Embeddings()
    store = ChromaStore(
        persist_directory=str(project_root / "chroma_db"),
        embedding_function=embeddings.get_embeddings(),
        knowledge_base_dir=str(project_root / "knowledge_base"),
        chunk_size=800,
        chunk_overlap=120,
        collection_name="documents",
    )
    store.index_documents()

    llm = ChatOllama(model="llama3.2", temperature=0.0)

    # New documents to process
    new_documents = [
        ("sql_basics.txt", "SQL"),
        ("git_workflow.txt", "Git"),
        ("docker_basics.txt", "Docker"),
        ("async_python.txt", "Async Python"),
        ("testing_python.txt", "Testing"),
    ]

    new_entries = []
    entry_id = len(dataset) + 1

    for doc_file, doc_label in new_documents:
        logger.info(f"Processing {doc_file}...")

        # Get all chunks for this document
        doc_chunks = []
        for chunk in store.vectorstore._collection.get()["documents"]:
            metadata = store.vectorstore._collection.get()["metadatas"][
                list(store.vectorstore._collection.get()["documents"]).index(chunk)
            ]
            if metadata and metadata.get("source", "").endswith(doc_file):
                doc_chunks.append(chunk)

        logger.info(f"Found {len(doc_chunks)} chunks for {doc_file}")

        # Select a few representative chunks (every nth chunk to avoid too many)
        selected_chunks = doc_chunks[::max(1, len(doc_chunks) // 8)][:8]

        for chunk_text in selected_chunks:
            if len(chunk_text.strip()) < 50:
                continue

            # Generate question
            question = generate_question_for_chunk(llm, chunk_text, doc_label)
            if not question:
                continue

            # Get relevant chunk IDs
            relevant_ids = get_relevant_chunk_ids(store, question, k=10)

            # Filter to only include chunks from this document
            doc_relevant_ids = [cid for cid in relevant_ids if doc_file in cid]

            if not doc_relevant_ids:
                logger.warning(f"No relevant chunks found for question: {question}")
                continue

            entry = {
                "id": f"{doc_file.replace('.txt', '')}_{entry_id:03d}",
                "question": question,
                "expected_answer": "Automatically generated - see chunk content",
                "expected_chunk_ids": doc_relevant_ids[:5],  # Top 5 most relevant
                "source_doc": doc_file,
                "difficulty": "medium",
                "category": doc_label.lower().replace(" ", "_"),
            }

            new_entries.append(entry)
            entry_id += 1
            logger.info(f"Generated: {question[:80]}...")

    # Add to dataset
    dataset.extend(new_entries)

    # Save updated dataset
    with open(golden_dataset_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    logger.info(f"Added {len(new_entries)} new entries to golden dataset")
    logger.info(f"Total entries: {len(dataset)}")


if __name__ == "__main__":
    main()