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
    # SQL basics
    {
        "id": "sql_001",
        "question": "Что такое SQL и для чего он используется?",
        "expected_answer": "SQL (Structured Query Language) — это стандартный язык для работы с реляционными базами данных. Он используется для создания, управления и запроса данных в СУБД, таких как PostgreSQL, MySQL, SQLite.",
        "expected_chunk_ids": [
            "/home/yevgeniy/PycharmProjects/AI-Research-Assistant/knowledge_base/sql_basics.txt:0:6e1c937a7ab3919d"
        ],
        "source_doc": "sql_basics.txt",
        "difficulty": "easy",
        "category": "general",
    },
    {
        "id": "sql_002",
        "question": "Какие основные типы данных существуют в SQL?",
        "expected_answer": "Основные типы данных SQL: числовые (INTEGER, BIGINT, DECIMAL, FLOAT), строковые (CHAR, VARCHAR, TEXT), дата/время (DATE, TIME, TIMESTAMP), а также BOOLEAN, BLOB, JSON, ARRAY, UUID.",
        "expected_chunk_ids": [
            "/home/yevgeniy/PycharmProjects/AI-Research-Assistant/knowledge_base/sql_basics.txt:539:79a6e11b8beeb9b2",
            "/home/yevgeniy/PycharmProjects/AI-Research-Assistant/knowledge_base/sql_basics.txt:1270:7b3adabb7430edf2",
        ],
        "source_doc": "sql_basics.txt",
        "difficulty": "easy",
        "category": "data_types",
    },
    {
        "id": "sql_003",
        "question": "Что такое реляционная база данных?",
        "expected_answer": "Реляционная база данных хранит данные в виде таблиц (отношений). Основные концепции: таблица, строка, столбец, первичный ключ, внешний ключ.",
        "expected_chunk_ids": [
            "/home/yevgeniy/PycharmProjects/AI-Research-Assistant/knowledge_base/sql_basics.txt:0:6e1c937a7ab3919d",
            "/home/yevgeniy/PycharmProjects/AI-Research-Assistant/knowledge_base/sql_basics.txt:539:79a6e11b8beeb9b2",
        ],
        "source_doc": "sql_basics.txt",
        "difficulty": "easy",
        "category": "general",
    },
    {
        "id": "sql_004",
        "question": "Какие команды относятся к DDL и DML в SQL?",
        "expected_answer": "DDL: CREATE, ALTER, DROP, TRUNCATE, RENAME. DML: SELECT, INSERT, UPDATE, DELETE, MERGE.",
        "expected_chunk_ids": [
            "/home/yevgeniy/PycharmProjects/AI-Research-Assistant/knowledge_base/sql_basics.txt:2706:fafd54952f3b39c2"
        ],
        "source_doc": "sql_basics.txt",
        "difficulty": "easy",
        "category": "commands",
    },
    {
        "id": "sql_005",
        "question": "Что такое JOIN в SQL и какие его типы существуют?",
        "expected_answer": "JOIN объединяет строки из двух или более таблиц. Типы: INNER JOIN, LEFT JOIN, RIGHT JOIN, FULL JOIN, CROSS JOIN, SELF JOIN.",
        "expected_chunk_ids": [
            "/home/yevgeniy/PycharmProjects/AI-Research-Assistant/knowledge_base/sql_basics.txt:4741:30f17da730058c38"
        ],
        "source_doc": "sql_basics.txt",
        "difficulty": "medium",
        "category": "joins",
    },
    {
        "id": "sql_006",
        "question": "Что такое агрегирующие функции в SQL?",
        "expected_answer": "Агрегирующие функции: COUNT(), SUM(), AVG(), MIN(), MAX(). Используются с GROUP BY для вычислений над набором строк.",
        "expected_chunk_ids": [
            "/home/yevgeniy/PycharmProjects/AI-Research-Assistant/knowledge_base/sql_basics.txt:4099:2b00a36f221bd973"
        ],
        "source_doc": "sql_basics.txt",
        "difficulty": "easy",
        "category": "aggregation",
    },
    {
        "id": "sql_007",
        "question": "Что такое подзапрос в SQL?",
        "expected_answer": "Подзапрос — это запрос внутри другого запроса. Может использоваться в WHERE, FROM, HAVING. Пример: SELECT name FROM employees WHERE salary > (SELECT AVG(salary) FROM employees).",
        "expected_chunk_ids": [
            "/home/yevgeniy/PycharmProjects/AI-Research-Assistant/knowledge_base/sql_basics.txt:5422:f227091f9c058bb3"
        ],
        "source_doc": "sql_basics.txt",
        "difficulty": "medium",
        "category": "subqueries",
    },
    {
        "id": "sql_008",
        "question": "Что такое нормализация баз данных?",
        "expected_answer": "Нормализация — это организация данных для минимизации избыточности. Нормальные формы: 1NF, 2NF, 3NF, BCNF.",
        "expected_chunk_ids": [
            "/home/yevgeniy/PycharmProjects/AI-Research-Assistant/knowledge_base/sql_basics.txt:7467:a2a7388b567dd495"
        ],
        "source_doc": "sql_basics.txt",
        "difficulty": "medium",
        "category": "normalization",
    },
    {
        "id": "sql_009",
        "question": "Какие свойства транзакций описываются acronym ACID?",
        "expected_answer": "ACID: Atomicity (атомарность), Consistency (согласованность), Isolation (изолированность), Durability (долговечность).",
        "expected_chunk_ids": [
            "/home/yevgeniy/PycharmProjects/AI-Research-Assistant/knowledge_base/sql_basics.txt:6763:d90151380dfcf9c5"
        ],
        "source_doc": "sql_basics.txt",
        "difficulty": "medium",
        "category": "transactions",
    },
    {
        "id": "sql_010",
        "question": "Какие типы индексов используются в SQL?",
        "expected_answer": "Типы индексов: B-tree, Hash, GiST, GIN, BRIN. Индексы ускоряют выборку, но замедляют INSERT/UPDATE/DELETE.",
        "expected_chunk_ids": [
            "/home/yevgeniy/PycharmProjects/AI-Research-Assistant/knowledge_base/sql_basics.txt:6095:f154a905d8c296d5"
        ],
        "source_doc": "sql_basics.txt",
        "difficulty": "medium",
        "category": "indexes",
    },
    # Git workflow
    {
        "id": "git_001",
        "question": "Что такое Git и кто его создал?",
        "expected_answer": "Git — это распределённая система контроля версий, созданная Линусом Торвальдсом в 2005 году для разработки ядра Linux.",
        "expected_chunk_ids": [
            "/home/yevgeniy/PycharmProjects/AI-Research-Assistant/knowledge_base/git_workflow.txt:0:41e60ef384c9306d"
        ],
        "source_doc": "git_workflow.txt",
        "difficulty": "easy",
        "category": "general",
    },
    {
        "id": "git_002",
        "question": "Что такое коммит в Git?",
        "expected_answer": "Коммит — это снимок состояния файлов в определённый момент времени. Имеет уникальный хеш SHA-1, содержит автора, дату, сообщение, родительский коммит, изменённые файлы.",
        "expected_chunk_ids": [
            "/home/yevgeniy/PycharmProjects/AI-Research-Assistant/knowledge_base/git_workflow.txt:0:41e60ef384c9306d"
        ],
        "source_doc": "git_workflow.txt",
        "difficulty": "easy",
        "category": "general",
    },
    {
        "id": "git_003",
        "question": "Как создать и переключиться на новую ветку в Git?",
        "expected_answer": "git branch <branch-name> — создать ветку. git checkout <branch-name> или git switch <branch-name> — переключиться. git checkout -b <branch> — создать и переключиться.",
        "expected_chunk_ids": [
            "/home/yevgeniy/PycharmProjects/AI-Research-Assistant/knowledge_base/git_workflow.txt:2948:53c485240a307d62"
        ],
        "source_doc": "git_workflow.txt",
        "difficulty": "easy",
        "category": "branching",
    },
    {
        "id": "git_004",
        "question": "Что такое Pull Request?",
        "expected_answer": "Pull Request (PR) или Merge Request (MR) — это предложение объединить изменения из одной ветки в другую. Обычно сопровождается обсуждением и код-ревью.",
        "expected_chunk_ids": [
            "/home/yevgeniy/PycharmProjects/AI-Research-Assistant/knowledge_base/git_workflow.txt:709:537bdb581a35d5b0"
        ],
        "source_doc": "git_workflow.txt",
        "difficulty": "easy",
        "category": "workflow",
    },
    {
        "id": "git_005",
        "question": "Как отменить изменения в рабочей директории Git?",
        "expected_answer": "git restore <file> — отменить изменения в файле. git restore . — отменить все изменения. git restore --staged <file> — убрать из индекса.",
        "expected_chunk_ids": [
            "/home/yevgeniy/PycharmProjects/AI-Research-Assistant/knowledge_base/git_workflow.txt:1908:4e8d167fea268f63"
        ],
        "source_doc": "git_workflow.txt",
        "difficulty": "medium",
        "category": "undo",
    },
    {
        "id": "git_006",
        "question": "Что такое git stash?",
        "expected_answer": "Stash позволяет временно сохранить незафиксированные изменения. git stash — сохранить, git stash list — список, git stash apply — применить, git stash pop — применить и удалить.",
        "expected_chunk_ids": [
            "/home/yevgeniy/PycharmProjects/AI-Research-Assistant/knowledge_base/git_workflow.txt:6533:a90bc0006d7e7463"
        ],
        "source_doc": "git_workflow.txt",
        "difficulty": "medium",
        "category": "stash",
    },
    {
        "id": "git_007",
        "question": "Что такое rebase в Git?",
        "expected_answer": "Rebase — это альтернатива merge, которая перезаписывает историю коммитов. Позволяет изменять, объединять, удалять или переупорядочивать коммиты.",
        "expected_chunk_ids": [
            "/home/yevgeniy/PycharmProjects/AI-Research-Assistant/knowledge_base/git_workflow.txt:5262:6385a917205b5de0"
        ],
        "source_doc": "git_workflow.txt",
        "difficulty": "medium",
        "category": "rebase",
    },
    {
        "id": "git_008",
        "question": "Какие модели ветвления существуют в Git?",
        "expected_answer": "Git-flow, GitHub Flow, Trunk-based Development. Git-flow использует постоянные ветки main/develop. GitHub Flow — простой workflow с PR. Trunk-based — все разработчики работают в основной ветке.",
        "expected_chunk_ids": [
            "/home/yevgeniy/PycharmProjects/AI-Research-Assistant/knowledge_base/git_workflow.txt:8518:ee7f1aec5f002dd4"
        ],
        "source_doc": "git_workflow.txt",
        "difficulty": "medium",
        "category": "workflow",
    },
    {
        "id": "git_009",
        "question": "Как найти коммит, сломавший функциональность?",
        "expected_answer": "Используется git bisect — бинарный поиск по истории коммитов. git bisect start, git bisect bad HEAD, git bisect good <commit-hash>, затем проверять каждый коммит.",
        "expected_chunk_ids": [
            "/home/yevgeniy/PycharmProjects/AI-Research-Assistant/knowledge_base/git_workflow.txt:10929:447698fe758a6c8c"
        ],
        "source_doc": "git_workflow.txt",
        "difficulty": "hard",
        "category": "debugging",
    },
    {
        "id": "git_010",
        "question": "Что такое .gitignore и зачем он нужен?",
        "expected_answer": ".gitignore указывает Git, какие файлы игнорировать. Используется для исключения зависимостей, артефактов сборки, IDE файлов, логов, окружений.",
        "expected_chunk_ids": [
            "/home/yevgeniy/PycharmProjects/AI-Research-Assistant/knowledge_base/git_workflow.txt:7730:7f547bb04823cbcb"
        ],
        "source_doc": "git_workflow.txt",
        "difficulty": "easy",
        "category": "configuration",
    },
]

dataset.extend(new_entries)

with open(golden_dataset_path, "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

logger.info("Added %d entries. Total: %d", len(new_entries), len(dataset))
