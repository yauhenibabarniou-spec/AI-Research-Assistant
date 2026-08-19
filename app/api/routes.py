from fastapi import HTTPException

from app.api.schemas import ReindexRequest, QueryResponse, QueryRequest
from app.core.config import settings
from app.rag.chains import create_rag_chain


def get_llm_config():
    """Получение конфигурации LLM модели."""
    return {
        "type": "ollama",
        "base_url": settings.ollama_base_url,
        "model": settings.ollama_model,
        "temperature": 0.7,
    }


def register_routes(app):
    """Регистрация API маршрутов."""

    @app.get("/")
    def root():
        """Информация о сервисе."""
        return {
            "message": "AI Research Assistant API",
            "documents_indexed": app.state.vector_manager.get_document_count(),
            "endpoints": {
                "query": "POST /query - задать вопрос",
                "documents": "GET /documents - список документов",
                "reindex": "POST /reindex - переиндексировать",
            },
        }

    @app.get("/documents")
    def get_documents_info():
        """Информация о загруженных документах."""
        count = app.state.vector_manager.get_document_count()
        return {
            "document_count": count,
            "knowledge_base_dir": "./knowledge_base",
        }

    @app.post("/reindex")
    def reindex_documents(request: ReindexRequest):
        """Переиндексация документов."""
        try:
            if request.clear_first:
                app.state.vector_manager.clear_index()

            count = app.state.vector_manager.index_documents()
            return {
                "success": True,
                "documents_indexed": count,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/query", response_model=QueryResponse)
    def query(request: QueryRequest):
        """Поиск ответа на вопрос пользователя."""
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Вопрос не может быть пустым")

        try:
            # Поиск релевантных документов
            relevant_docs = app.state.vector_manager.get_relevant_documents(request.question, k=request.k)

            if not relevant_docs:
                return QueryResponse(
                    answer="К сожалению, я не нашёл релевантных документов для ответа на ваш вопрос. Попробуйте загрузить документы в папку knowledge_base/ и выполнить переиндексацию.",
                    sources=[],
                    model_used="no_context",
                )

            # Формирование контекста из документов
            context = "\n\n".join([doc.page_content for doc, _ in relevant_docs])
            sources = list(set([doc.metadata.get("source", "unknown") for doc, _ in relevant_docs]))

            # Получение LLM и создание цепочки
            llm_config = get_llm_config()
            chain = create_rag_chain(llm_config)

            # Генерация ответа
            answer = chain.invoke(
                {
                    "context": context,
                    "question": request.question,
                }
            )

            # Определение использованной модели
            model_used = settings.ollama_model

            return QueryResponse(
                answer=answer,
                sources=sources,
                model_used=model_used,
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Ошибка при обработке запроса: {str(e)}")
