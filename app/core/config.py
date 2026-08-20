from pydantic_settings import BaseSettings as PydanticBaseSettings
from pydantic_settings import SettingsConfigDict


class Settings(PydanticBaseSettings):
    """Настройки приложения."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Параметры векторного хранилища
    persist_directory: str = "./chroma_db"
    knowledge_base_dir: str = "./knowledge_base"
    chunk_size: int = 800
    chunk_overlap: int = 120
    collection_name: str = "documents"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Параметры модели
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    temperature: float = 0.7

    # Параметры API
    host: str = "0.0.0.0"
    port: int = 8000

    # Параметры CORS
    allowed_origins: list[str] = ["*"]

    # Параметры кэширования
    max_context_length: int = 4000

    # Параметры безопасности
    api_key: str | None = None

    # Параметры Rate Limiting
    rate_limit: int = 100  # запросов в минуту

    # Параметры гибридного поиска
    hybrid_bm25_k: int = 10
    hybrid_rerank_k: int = 5
    hybrid_alpha: float = 0.6


settings = Settings()
