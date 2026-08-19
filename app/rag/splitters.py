from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextSplitter:
    """Разделение текста на чанки."""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            add_start_index=True,
        )

    def split_documents(self, documents: list) -> list:
        """Разделение документов на чанки."""
        chunks = []
        for doc in documents:
            chunks.extend(self.text_splitter.split_documents([doc]))
        return chunks
