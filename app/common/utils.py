import hashlib
import os
from pathlib import Path

from langchain_core.documents import Document


def chunk_id(doc: Document) -> str:
    """Generate deterministic ID for a document chunk."""
    src = doc.metadata.get("source", "")
    start = doc.metadata.get("start_index", 0)
    
    # Normalize path to absolute path
    if src and not os.path.isabs(src):
        src = os.path.abspath(src)
    
    h = hashlib.sha256(doc.page_content.encode()).hexdigest()[:16]
    return f"{src}:{start}:{h}"
