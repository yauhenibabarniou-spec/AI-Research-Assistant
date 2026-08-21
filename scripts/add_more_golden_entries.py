import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

project_root = Path(__file__).resolve().parents[1]
golden_dataset_path = project_root / "eval" / "golden_dataset.json"

with open(golden_dataset_path, "r", encoding="utf-8") as f:
    dataset = json.load(f)

new_entries = [
    # Docker basics
    {
        "id": "docker_001",
        "question": "Что такое Docker и зачем он используется?",
        "expected_answer": "Docker — это платформа для разработки, доставки и запуска контейнеризированных приложений. Контейнеры упаковывают приложение со всеми зависимостями.",
        "expected_chunk_ids": [
            "/home/yevgeniy/PycharmProjects/AI-Research-Assistant/knowledge_base/docker_basics.txt:0:d31e864b5534923d"
        ],
        "source_doc": "docker_basics.txt",
        "difficulty": "easy",
        "category": "general",
    },
    {
        "id": "docker_002",
        "question": "Что такое контейнер и образ в Docker?",
        "expected_answer": "Образ — неизменяемый шаблон для создания контейнеров. Контейнер — лёгкий, автономный, исполняемый пакет с кодом, средой выполнения, библиотеками и настройками.",
        "expected_chunk_ids": [
            "/home/yevgeniy/PycharmProjects/AI-Research-Assistant/knowledge_base/docker_basics.txt:0:d31e864b5534923d"
        ],
        "source_doc": "docker_basics.txt",
        "difficulty": "easy",
        "category": "concepts",
    },
    {
        "id": "docker_003",
        "question": "Что такое Dockerfile?",
        "expected_answer": "Dockerfile — текстовый документ с командами для сборки образа. Содержит FROM, RUN, COPY, CMD, EXPOSE и другие инструкции.",
        "expected_chunk_ids": [
            "/home/yevgeniy/PycharmProjects/AI-Research-Assistant/knowledge_base/docker_basics.txt:4200:72535d670ab8e7a2",
            "/home/yevgeniy/PycharmProjects/AI-Research-Assistant/knowledge_base/docker_basics.txt:4256:d8beebb4a7d84e82",
        ],
        "source_doc": "docker_basics.txt",
        "difficulty": "easy",
        "category": "dockerfile",
    },
    {
        "id": "docker_004",
        "question": "Как установить Docker на Ubuntu?",
        "expected_answer": "Установка: sudo apt update, установка зависимостей, добавление GPG-ключа Docker, добавление репозитория, sudo apt install docker-ce docker-ce-cli containerd.io docker-compose-plugin, добавление пользователя в группу docker.",
        "expected_chunk_ids": [
            "/home/yevgeniy/PycharmProjects/AI-Research-Assistant/knowledge_base/docker_basics.txt:1557:e0debfae833f72d5",
            "/home/yevgeniy/PycharmProjects/AI-Research-Assistant/knowledge_base/docker_basics.txt:2187:fdef1cd1c5adcb4c",
        ],
        "source_doc": "docker_basics.txt",
        "difficulty": "medium",
        "category": "installation",
    },
    {
        "id": "docker_005",
        "question": "Что такое Docker Compose?",
        "expected_answer": "Docker Compose — инструмент для определения и запуска многоконтейнерных приложений через YAML-файл. Позволяет описывать сервисы, сети, тома.",
        "expected_chunk_ids": [
            "/home/yevgeniy/PycharmProjects/AI-Research-Assistant/knowledge_base/docker_basics.txt:6568:ff7e17b550b2ec7b"
        ],
        "source_doc": "docker_basics.txt",
        "difficulty": "easy",
        "category": "compose",
    },
    {
        "id": "docker_006",
        "question": "Какие типы сетей существуют в Docker?",
        "expected_answer": "Типы сетей: bridge, host, none, overlay, macvlan. Bridge — сеть моста по умолчанию. Host — контейнер использует сетевой стек хоста.",
        "expected_chunk_ids": [
            "/home/yevgeniy/PycharmProjects/AI-Research-Assistant/knowledge_base/docker_basics.txt:7988:fcf53a404b781378"
        ],
        "source_doc": "docker_basics.txt",
        "difficulty": "medium",
        "category": "networking",
    },
    {
        "id": "docker_007",
        "question": "Что такое тома в Docker?",
        "expected_answer": "Тома — механизм хранения данных. Типы: bind mounts, named volumes, tmpfs mounts. Позволяют сохранять данные после удаления контейнера.",
        "expected_chunk_ids": [
            "/home/yevgeniy/PycharmProjects/AI-Research-Assistant/knowledge_base/docker_basics.txt:9008:a4f65f5ca5c7aaa1"
        ],
        "source_doc": "docker_basics.txt",
        "difficulty": "medium",
        "category": "storage",
    },
    {
        "id": "docker_008",
        "question": "Что такое Docker Swarm?",
        "expected_answer": "Docker Swarm — встроенный инструмент для оркестрации контейнеров. Основные концепции: Node, Service, Task, Manager node, Worker node.",
        "expected_chunk_ids": [
            "/home/yevgeniy/PycharmProjects/AI-Research-Assistant/knowledge_base/docker_basics.txt:9695:ff2eb1e29145df40"
        ],
        "source_doc": "docker_basics.txt",
        "difficulty": "medium",
        "category": "orchestration",
    },
    {
        "id": "docker_009",
        "question": "Какие лучшие практики безопасности Docker рекомендуются?",
        "expected_answer": "Не запускать контейнеры от root, использовать официальные образы, регулярно обновлять образы, использовать минимальные базовые образы, сканировать на уязвимости, ограничивать ресурсы.",
        "expected_chunk_ids": [
            "/home/yevgeniy/PycharmProjects/AI-Research-Assistant/knowledge_base/docker_basics.txt:11537:76afc84512540b98"
        ],
        "source_doc": "docker_basics.txt",
        "difficulty": "medium",
        "category": "security",
    },
    {
        "id": "docker_010",
        "question": "Что такое многоэтапная сборка в Docker?",
        "expected_answer": "Многоэтапная сборка позволяет создавать маленькие и безопасные образы. Использует несколько FROM инструкций: один этап для сборки, другой для продакшена.",
        "expected_chunk_ids": [
            "/home/yevgeniy/PycharmProjects/AI-Research-Assistant/knowledge_base/docker_basics.txt:12165:d11fbcfc7ffa2d8c"
        ],
        "source_doc": "docker_basics.txt",
        "difficulty": "medium",
        "category": "optimization",
    },
    # Async Python
    {
        "id": "async_001",
        "question": "Что такое асинхронное программирование в Python?",
        "expected_answer": "Асинхронное программирование позволяет выполнять операции concurrently без многопоточности. Реализуется через asyncio, event loop, корутины с async/await.",
        "expected_chunk_ids": [
            "/home/yevgeniy/PycharmProjects/AI-Research-Assistant/knowledge_base/async_python.txt:0:f861b545993fc265"
        ],
        "source_doc": "async_python.txt",
        "difficulty": "easy",
        "category": "general",
    },
    {
        "id": "async_002",
        "question": "Что такое event loop?",
        "expected_answer": "Event loop — центральный компонент asyncio, который управляет выполнением асинхронных задач. Отслеживает события, выполняет корутины, обрабатывает callbacks.",
        "expected_chunk_ids": [
            "/home/yevgeniy/PycharmProjects/AI-Research-Assistant/knowledge_base/async_python.txt:775:1438e62724fb2f67"
        ],
        "source_doc": "async_python.txt",
        "difficulty": "medium",
        "category": "core",
    },
    {
        "id": "async_003",
        "question": "Что такое корутины в Python?",
        "expected_answer": "Корутины — функции, которые могут приостанавливать и возобновлять выполнение. Объявляются через async def. Используют await для ожидания асинхронных операций.",
        "expected_chunk_ids": [
            "/home/yevgeniy/PycharmProjects/AI-Research-Assistant/knowledge_base/async_python.txt:1492:3e870d075931414a"
        ],
        "source_doc": "async_python.txt",
        "difficulty": "easy",
        "category": "core",
    },
    {
        "id": "async_004",
        "question": "Как работает async/await?",
        "expected_answer": "async def объявляет корутину. await приостанавливает выполнение до завершения асинхронной операции без блокировки event loop.",
        "expected_chunk_ids": [
            "/home/yevgeniy/PycharmProjects/AI-Research-Assistant/knowledge_base/async_python.txt:1492:3e870d075931414a",
            "/home/yevgeniy/PycharmProjects/AI-Research-Assistant/knowledge_base/async_python.txt:2162:43b05297398c0961",
        ],
        "source_doc": "async_python.txt",
        "difficulty": "easy",
        "category": "syntax",
    },
    {
        "id": "async_005",
        "question": "Что такое asyncio.gather()?",
        "expected_answer": "asyncio.gather() запускает несколько корутин concurrently и возвращает список результатов. Подходит для параллельного выполнения независимых операций.",
        "expected_chunk_ids": [
            "/home/yevgeniy/PycharmProjects/AI-Research-Assistant/knowledge_base/async_python.txt:2162:43b05297398c0961"
        ],
        "source_doc": "async_python.txt",
        "difficulty": "medium",
        "category": "concurrency",
    },
    {
        "id": "async_006",
        "question": "Что такое aiohttp?",
        "expected_answer": "aiohttp — асинхронный HTTP-клиент и сервер. Позволяет выполнять HTTP-запросы concurrently без блокировки event loop.",
        "expected_chunk_ids": [
            "/home/yevgeniy/PycharmProjects/AI-Research-Assistant/knowledge_base/async_python.txt:3118:02cf3f3fe373dedb"
        ],
        "source_doc": "async_python.txt",
        "difficulty": "easy",
        "category": "libraries",
    },
    {
        "id": "async_007",
        "question": "Что такое asyncio.Queue?",
        "expected_answer": "asyncio.Queue — очередь для безопасного обмена данными между корутинами. Используется в паттернах Producer-Consumer.",
        "expected_chunk_ids": [
            "/home/yevgeniy/PycharmProjects/AI-Research-Assistant/knowledge_base/async_python.txt:6315:2eb3f99fdd6bcc03"
        ],
        "source_doc": "async_python.txt",
        "difficulty": "medium",
        "category": "synchronization",
    },
    {
        "id": "async_008",
        "question": "Что такое asyncio.TaskGroup()?",
        "expected_answer": "asyncio.TaskGroup() — контекстный менеджер Python 3.11+ для группового управления задачами. Все задачи завершаются при выходе из блока async with.",
        "expected_chunk_ids": [
            "/home/yevgeniy/PycharmProjects/AI-Research-Assistant/knowledge_base/async_python.txt:7829:e9e645a0f0837a9a"
        ],
        "source_doc": "async_python.txt",
        "difficulty": "medium",
        "category": "concurrency",
    },
    {
        "id": "async_009",
        "question": "Как обработать таймаут в асинхронном коде?",
        "expected_answer": "Используется asyncio.wait_for(coroutine, timeout) или asyncio.timeout(seconds) в Python 3.11+. При превышении времени возникает TimeoutError.",
        "expected_chunk_ids": [
            "/home/yevgeniy/PycharmProjects/AI-Research-Assistant/knowledge_base/async_python.txt:8502:574b497822f2a20c"
        ],
        "source_doc": "async_python.txt",
        "difficulty": "medium",
        "category": "error_handling",
    },
    {
        "id": "async_010",
        "question": "Что такое паттерн Rate Limiter в асинхронном коде?",
        "expected_answer": "Rate Limiter ограничивает количество одновременных запросов. Реализуется через семафоры или токены. Пример: класс RateLimiter с методом acquire().",
        "expected_chunk_ids": [
            "/home/yevgeniy/PycharmProjects/AI-Research-Assistant/knowledge_base/async_python.txt:10494:f94f93f394ec7e87",
            "/home/yevgeniy/PycharmProjects/AI-Research-Assistant/knowledge_base/async_python.txt:11188:2cb3e7d0a14c4369",
        ],
        "source_doc": "async_python.txt",
        "difficulty": "hard",
        "category": "patterns",
    },
    # Testing Python
    {
        "id": "testing_001",
        "question": "Что такое pytest?",
        "expected_answer": "pytest — популярный фреймворк для тестирования Python с простым синтаксисом и мощными возможностями: fixtures, параметризация, маркеры.",
        "expected_chunk_ids": [
            "/home/yevgeniy/PycharmProjects/AI-Research-Assistant/knowledge_base/testing_python.txt:0:cd8400e4f3d7e4ea"
        ],
        "source_doc": "testing_python.txt",
        "difficulty": "easy",
        "category": "frameworks",
    },
    {
        "id": "testing_002",
        "question": "Что такое fixtures в pytest?",
        "expected_answer": "Fixtures — функции, предоставляющие тестовые данные или настраивающие окружение. Объявляются через @pytest.fixture. Могут иметь scope: session, module, function.",
        "expected_chunk_ids": [
            "/home/yevgeniy/PycharmProjects/AI-Research-Assistant/knowledge_base/testing_python.txt:1887:6d3d2a9ba400c15e"
        ],
        "source_doc": "testing_python.txt",
        "difficulty": "easy",
        "category": "fixtures",
    },
    {
        "id": "testing_003",
        "question": "Как параметризовать тесты в pytest?",
        "expected_answer": "Используется @pytest.mark.parametrize с указанием параметров и ожидаемых значений. Позволяет запустить тест с несколькими наборами данных.",
        "expected_chunk_ids": [
            "/home/yevgeniy/PycharmProjects/AI-Research-Assistant/knowledge_base/testing_python.txt:2487:bf878c6821ea7b35"
        ],
        "source_doc": "testing_python.txt",
        "difficulty": "medium",
        "category": "parametrization",
    },
    {
        "id": "testing_004",
        "question": "Что такое мокирование в тестировании?",
        "expected_answer": "Мокирование — замена реальных зависимостей на тестовые двойники. В Python используется unittest.mock с Mock, patch, MagicMock.",
        "expected_chunk_ids": [
            "/home/yevgeniy/PycharmProjects/AI-Research-Assistant/knowledge_base/testing_python.txt:3864:c2a84893661fa9bf"
        ],
        "source_doc": "testing_python.txt",
        "difficulty": "medium",
        "category": "mocking",
    },
    {
        "id": "testing_005",
        "question": "Как тестировать исключения в pytest?",
        "expected_answer": "Используется pytest.raises(ExceptionType) как контекстный менеджер. Можно проверять тип исключения и сообщение через match.",
        "expected_chunk_ids": [
            "/home/yevgeniy/PycharmProjects/AI-Research-Assistant/knowledge_base/testing_python.txt:3864:c2a84893661fa9bf"
        ],
        "source_doc": "testing_python.txt",
        "difficulty": "medium",
        "category": "exceptions",
    },
    {
        "id": "testing_006",
        "question": "Что такое покрытие кода?",
        "expected_answer": "Покрытие кода — процент кода, выполненного во время тестов. Измеряется через pytest-cov. Цель — 80-90% покрытие.",
        "expected_chunk_ids": [
            "/home/yevgeniy/PycharmProjects/AI-Research-Assistant/knowledge_base/testing_python.txt:5291:19292c846b8b4f8d"
        ],
        "source_doc": "testing_python.txt",
        "difficulty": "easy",
        "category": "coverage",
    },
    {
        "id": "testing_007",
        "question": "Как тестировать FastAPI приложения?",
        "expected_answer": "Используется TestClient из fastapi.testclient. Создаётся фикстура client = TestClient(app). Запросы: client.get('/'), client.post('/', json=...). Проверка status_code и json().",
        "expected_chunk_ids": [
            "/home/yevgeniy/PycharmProjects/AI-Research-Assistant/knowledge_base/testing_python.txt:7212:406fa5fa9462f433"
        ],
        "source_doc": "testing_python.txt",
        "difficulty": "medium",
        "category": "web_testing",
    },
    {
        "id": "testing_008",
        "question": "Что такое pytest-asyncio?",
        "expected_answer": "pytest-asyncio — плагин для тестирования асинхронного кода. Использует @pytest.mark.asyncio для пометки асинхронных тестов.",
        "expected_chunk_ids": [
            "/home/yevgeniy/PycharmProjects/AI-Research-Assistant/knowledge_base/testing_python.txt:12442:155adb01aaa9dc81"
        ],
        "source_doc": "testing_python.txt",
        "difficulty": "medium",
        "category": "async_testing",
    },
    {
        "id": "testing_009",
        "question": "Что такое TDD?",
        "expected_answer": "TDD (Test-Driven Development) — разработка через тестирование. Сначала пишутся тесты, затем реализуется функционал, после чего рефакторинг.",
        "expected_chunk_ids": [
            "/home/yevgeniy/PycharmProjects/AI-Research-Assistant/knowledge_base/testing_python.txt:11357:636702dbfb0ea270"
        ],
        "source_doc": "testing_python.txt",
        "difficulty": "easy",
        "category": "methodology",
    },
    {
        "id": "testing_010",
        "question": "Какие лучшие практики тестирования рекомендуются?",
        "expected_answer": "Пишите тесты до кода (TDD), один тест — одна проверка, используйте описательные имена, изолируйте тесты, тестируйте граничные случаи, поддерживайте покрытие 80-90%, автоматизируйте в CI/CD.",
        "expected_chunk_ids": [
            "/home/yevgeniy/PycharmProjects/AI-Research-Assistant/knowledge_base/testing_python.txt:11357:636702dbfb0ea270"
        ],
        "source_doc": "testing_python.txt",
        "difficulty": "easy",
        "category": "best_practices",
    },
]

dataset.extend(new_entries)

with open(golden_dataset_path, "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

logger.info("Added %d entries. Total: %d", len(new_entries), len(dataset))
