from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


def create_rag_chain(llm_config: dict):
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

    # Создание модели из конфига
    llm_instance = ChatOllama(
        base_url=llm_config["base_url"],
        model=llm_config["model"],
        temperature=llm_config["temperature"],
    )

    # Цепочка: контекст + вопрос → промпт → LLM → ответ
    chain = prompt | llm_instance | StrOutputParser()

    return chain
