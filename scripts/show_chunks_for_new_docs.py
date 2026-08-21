from pathlib import Path

from langchain_core.documents import Document

from app.common.utils import chunk_id
from app.rag.embeddings import Embeddings
from app.rag.stores.chroma import ChromaStore

project_root = Path(__file__).resolve().parents[1]

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

# Print chunks with IDs for manual inspection
for doc_file in ["sql_basics.txt", "git_workflow.txt", "docker_basics.txt", "async_python.txt", "testing_python.txt"]:
    print(f"\n=== {doc_file} ===")
    for chunk in store.vectorstore._collection.get()["documents"]:
        metas = store.vectorstore._collection.get()["metadatas"]
        idx = list(store.vectorstore._collection.get()["documents"]).index(chunk)
        meta = metas[idx] if metas and idx < len(metas) else {}
        if meta.get("source", "").endswith(doc_file):
            cid = chunk_id(Document(page_content=chunk, metadata=meta))
            preview = chunk[:120].replace("\n", " ")
            print(f"{cid}\n  {preview}\n")
