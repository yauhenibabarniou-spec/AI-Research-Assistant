# AI Research Assistant

Пет-проект для изучения OpenAI API / LLM / AI agents.

## Цель
Пользователь задаёт вопрос → система ищет релевантные документы → генерирует точный ответ на основе найденного.

## Стек
- **Backend**: Python + FastAPI
- **LLM**: Ollama (бесплатная локальная модель) или OpenAI API (gpt-4o-mini)
- **База знаний**: PDF/текстовые файлы в папке `knowledge_base/`
- **Векторное хранилище**: ChromaDB (локальное, бесплатное)
- **Библиотеки**: langchain-core (базовые компоненты)

## Установка

```bash
pip install -r requirements.txt
```

### Опционально: Ollama для локальной LLM
Если нет подписки на OpenAI API, можно использовать Ollama:

1. Установите Ollama: https://ollama.ai
2. Скачайте модель: `ollama pull llama3.2` или `ollama pull mistral`

## Структура проекта

```
/workspace
├── app.py                 # FastAPI приложение
├── vector_store.py        # Работа с векторным хранилищем
├── knowledge_base/        # Папка с документами (PDF/TXT)
├── .env                   # Переменные окружения
└── requirements.txt       # Зависимости
```

## Настройка

Создайте файл `.env`:

```env
# Для OpenAI API (если есть ключ)
OPENAI_API_KEY=your-key-here
USE_OPENAI=false

# Для Ollama (локально, бесплатно)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
```

## Запуск

```bash
# 1. Индексация документов
python vector_store.py

# 2. Запуск сервера
uvicorn app:app --reload
```

## API Endpoints

- `POST /query` - задать вопрос и получить ответ
- `GET /documents` - список загруженных документов
- `POST /reindex` - переиндексировать документы

## Пример запроса

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Как установить FastAPI?"}'
```