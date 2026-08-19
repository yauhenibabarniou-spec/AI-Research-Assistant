from langchain_huggingface import HuggingFaceEmbeddings


class Embeddings:
    """Эмбеддинги для векторизации текста."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    def get_embeddings(self):
        """Получить объект эмбеддингов."""
        return self.embeddings
