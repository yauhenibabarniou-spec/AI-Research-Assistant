import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from vector_store import VectorStoreManager

# Инициализация приложения
app = FastAPI(
    title="AI Research Assistant",
    description="Система поиска и генерации ответов на основе документов",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Модели данных
class QueryRequest(BaseModel):
    question: str
    k: int = 3  # количество релевантных документов


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    model_used: str


class ReindexRequest(BaseModel):
    clear_first: bool = False


# Глобальный менеджер векторного хранилища
vector_manager = VectorStoreManager()


def get_llm():
    """Получение LLM модели в зависимости от настроек."""

    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2")

    # Возвращаем конфиг для ленивой инициализации
    return {
        "type": "ollama",
        "base_url": ollama_url,
        "model": ollama_model,
        "temperature": 0.7,
    }


def create_rag_chain(llm):
    """Создание RAG цепочки."""
    # Промпт для генерации ответа
    template = """Ты - помощник, который отвечает на вопросы только на основе предоставленных документов.
    
Контекст из документов:
{context}

Вопрос пользователя: {question}

Если ответ не может быть найден в документах, честно скажи об этом.
Ответ должен быть точным и полезным.

Ответ:"""

    prompt = ChatPromptTemplate.from_template(template)

    # Если llm это dict (Ollama конфиг), создаём модель лениво
    if isinstance(llm, dict):
        from langchain_community.llms import Ollama

        llm_instance = Ollama(
            base_url=llm["base_url"],
            model=llm["model"],
            temperature=llm["temperature"],
        )
    else:
        llm_instance = llm

    # Цепочка: контекст + вопрос → промпт → LLM → ответ
    chain = prompt | llm_instance | StrOutputParser()

    return chain


@app.get("/")
def root():
    """Информация о сервисе."""
    return {
        "message": "AI Research Assistant API",
        "documents_indexed": vector_manager.get_document_count(),
        "endpoints": {
            "query": "POST /query - задать вопрос",
            "documents": "GET /documents - список документов",
            "reindex": "POST /reindex - переиндексировать",
        },
    }


@app.get("/documents")
def get_documents_info():
    """Информация о загруженных документах."""
    count = vector_manager.get_document_count()
    return {
        "document_count": count,
        "knowledge_base_dir": "./knowledge_base",
    }


@app.post("/reindex")
def reindex_documents(request: ReindexRequest):
    """Переиндексация документов."""
    try:
        if request.clear_first:
            vector_manager.clear_index()

        count = vector_manager.index_documents()
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
        relevant_docs = vector_manager.get_relevant_documents(request.question, k=request.k)

        if not relevant_docs:
            return QueryResponse(
                answer="К сожалению, я не нашёл релевантных документов для ответа на ваш вопрос. Попробуйте загрузить документы в папку knowledge_base/ и выполнить переиндексацию.",
                sources=[],
                model_used="no_context",
            )

        # Формирование контекста из документов
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        sources = list(set([doc.metadata.get("source", "unknown") for doc in relevant_docs]))

        # Получение LLM и создание цепочки
        llm = get_llm()

        if llm is None:
            # Режим без LLM - просто возвращаем найденные документы
            answer = f"Найдено {len(relevant_docs)} релевантных фрагментов:\n\n"
            for i, doc in enumerate(relevant_docs, 1):
                answer += f"{i}. {doc.page_content[:500]}...\n\n"
            answer += "\nПримечание: LLM (Ollama) недоступна. Показаны только найденные фрагменты."

            model_used = os.getenv("OLLAMA_MODEL", "llama3.2")

            return QueryResponse(
                answer=answer,
                sources=sources,
                model_used=f"{model_used} (unavailable)",
            )

        chain = create_rag_chain(llm)

        # Генерация ответа
        answer = chain.invoke(
            {
                "context": context,
                "question": request.question,
            }
        )

        # Определение использованной модели
        use_openai = os.getenv("USE_OPENAI", "false").lower() == "true"
        model_used = "gpt-4o-mini" if use_openai else os.getenv("OLLAMA_MODEL", "llama3.2")

        return QueryResponse(
            answer=answer,
            sources=sources,
            model_used=model_used,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке запроса: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
