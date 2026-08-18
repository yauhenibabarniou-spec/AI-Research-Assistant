# FastAPI Documentation

FastAPI автоматически генерирует интерактивную документацию:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Доступные эндпоинты

### GET /
Получить информацию о сервисе и количестве проиндексированных документов.

### GET /documents
Получить информацию о загруженных документах.

### POST /query
Задать вопрос и получить ответ на основе документов.

**Request Body:**
```json
{
  "question": "Как установить FastAPI?",
  "k": 3
}
```

**Response:**
```json
{
  "answer": "Для установки FastAPI выполните команду: pip install fastapi",
  "sources": ["./knowledge_base/fastapi_docs.pdf"],
  "model_used": "llama3.2"
}
```

### POST /reindex
Переиндексировать документы из папки knowledge_base/.

**Request Body:**
```json
{
  "clear_first": false
}
```

**Response:**
```json
{
  "success": true,
  "documents_indexed": 15
}
```
