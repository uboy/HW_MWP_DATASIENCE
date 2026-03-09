# RAG-конвейер для вопросов по PDF-документу

Этот проект отвечает на вопросы по произвольному PDF-документу с помощью Retrieval-Augmented Generation и использует только переданный PDF как источник знаний.

## Что делает скрипт при запуске

В самом начале CLI печатает техническую сводку:
- чем разбирается PDF;
- как выполняется чанкинг;
- какие эмбеддинги используются;
- какой план retrieval применяется;
- какой реранкер и какая LLM используются;
- какие защитные правила включены для false premise и prompt injection;
- какой файл вопросов загружен;
- какие выходные файлы будут сформированы.

После этого скрипт показывает:
1. список вопросов;
2. ход обработки;
3. красиво оформленные блоки `Вопрос` / `Ответ`;
4. пути к итоговым файлам.

## Архитектура

```text
PDF -> DocumentProcessor (pymupdf4llm) -> StructuredChunks
    -> IndexStore: FAISS (bge-m3) + BM25 + StructuralIndex

Вопрос
  -> QueryClassifier + детерминированные правила
  -> StrategyRetriever: dense + sparse + structural + RRF + reranker
  -> AnswerGenerator (структурированный JSON-вывод)
  -> заземлённый ответ или явный no-data / correction response
```

Подробности: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

Текущий retrieval-слой уже включает несколько generic-улучшений поверх базового RAG:
- `query rewriting` и `multi-query retrieval`;
- адаптивную retrieval-политику по типу вопроса;
- ранний `sentence-level evidence packing` перед генерацией;
- generic `list coverage check` для длинных вопросов-перечней.

## Подготовка окружения

Открой PowerShell в корне проекта и выполни:

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

После этого открой `.env` и задай минимум:
- `OLLAMA_BASE_URL`
- `OLLAMA_MODEL`

Опционально можно настроить retrieval-слой:
- `ADAPTIVE_RETRIEVAL_ENABLED=1` — включить adaptive retrieval policy;
- `EVIDENCE_MAX_SENTENCES=10` — общий budget предложений в evidence-pack;
- `LIST_COVERAGE_MIN_ITEMS=8` — минимальный порог покрытия для list-вопросов перед дополнительным расширением evidence.

## Как запускать

### 1. Интерактивный режим

Если не передавать `--question` и `--questions-file`, скрипт ждёт многострочный ввод.

Команда:

```powershell
.\venv\Scripts\python.exe -m rag.main --pdf C:\path\to\document.pdf
```

Что вводить:
- вводишь один вопрос в строку;
- когда все вопросы введены, нажимаешь Enter на пустой строке;
- после пустой строки начинается обработка.

Пример:

```text
Какие федеральные законы составляют правовую основу Стратегии?
Что в Стратегии понимается под искусственным интеллектом?

```

Последняя пустая строка запускает обработку.

### 2. Один вопрос из командной строки

Команда:

```powershell
.\venv\Scripts\python.exe -m rag.main --pdf C:\path\to\document.pdf --question "Какие федеральные законы составляют правовую основу Стратегии?"
```

### 3. Пакетный запуск на файле вопросов

Команда:

```powershell
.\venv\Scripts\python.exe -m rag.main --pdf C:\path\to\document.pdf --questions-file test_set.xlsx
```

### 4. Пакетный запуск с экспортом ответов

Команда:

```powershell
.\venv\Scripts\python.exe -m rag.main `
  --pdf C:\path\to\document.pdf `
  --questions-file test_set.xlsx `
  --submission-out results.xlsx `
  --answers-out .scratchpad\answers_submission.csv `
  --report-out .scratchpad\answers_report.txt `
  --debug-out .scratchpad\answers_debug.csv
```

Что делает эта команда:
- читает вопросы из `test_set.xlsx`;
- отвечает на все вопросы по переданному PDF;
- создаёт экспортированную таблицу `results.xlsx` с колонкой `answer`;
- создаёт отдельный CSV только с ответами;
- создаёт текстовый отчёт для просмотра;
- создаёт debug CSV с источниками и rerank score.

## Что именно куда вводить

### Если хочешь просто задать вопросы вручную

1. Открой PowerShell в папке проекта.
2. Выполни:

```powershell
.\venv\Scripts\python.exe -m rag.main --pdf C:\path\to\document.pdf
```

3. Введи вопросы построчно.
4. Поставь пустую строку.
5. Дождись вывода ответов.

### Если хочешь получить итоговую таблицу ответов

Ничего вручную вводить не нужно. Просто запусти:

```powershell
.\venv\Scripts\python.exe -m rag.main `
  --pdf C:\path\to\document.pdf `
  --questions-file test_set.xlsx `
  --submission-out results.xlsx `
  --answers-out .scratchpad\answers_submission.csv `
  --report-out .scratchpad\answers_report.txt `
  --debug-out .scratchpad\answers_debug.csv
```

После этого основной экспортированный файл будет таким:

```text
results.xlsx
```

## Аргументы CLI

| Аргумент | Что означает |
|---|---|
| `--pdf` | Путь к исходному PDF |
| `--question` | Один вопрос |
| `--questions-file` | Файл с вопросами `.txt`, `.csv` или `.xlsx` |
| `--submission-out` | Экспорт исходной таблицы вопросов с заполненной колонкой `answer` |
| `--answers-out` | CSV только с ответами |
| `--debug-out` | Отладочный CSV с источниками и оценками |
| `--report-out` | Текстовый отчёт с парами вопрос/ответ |
| `--no-debug-out` | Не создавать debug CSV |
| `--no-report-out` | Не создавать текстовый отчёт |
| `--answers-no-header` | Записать CSV с ответами без заголовка |

`--question` и `--questions-file` одновременно использовать нельзя.

## Какие файлы создаются

| Файл | Назначение |
|---|---|
| `results.xlsx` | Экспортированная таблица вопросов с заполненной колонкой `answer` |
| `.scratchpad\answers_submission.csv` | CSV только с ответами |
| `.scratchpad\answers_report.txt` | Читаемый форматированный отчёт |
| `.scratchpad\answers_debug.csv` | Отладочный CSV с источниками и оценками |

## Локальная оценка качества

Команда:

```powershell
.\venv\Scripts\python.exe -m rag.evaluate `
  --predictions .scratchpad\answers_submission.csv `
  --gold C:\path\to\gold_answers.csv `
  --out .scratchpad\homework_metrics_eval.csv `
  --summary-out .scratchpad\homework_metrics_summary.md
```

Что делает эта команда:
- читает файл с ответами;
- сравнивает его с указанным gold-набором;
- считает набор локальных proxy-метрик качества ответа;
- пишет CSV с метриками по каждому вопросу;
- пишет markdown summary со средними значениями и самыми слабыми вопросами.

Важно:
- это локальная proxy-оценка, а не гарантированная копия внешнего проверяющего контура;
- gold-файл нужно передать явно через `--gold`;
- в summary выводятся агрегированные показатели качества и самые слабые вопросы.

## Self-check verifier

В рантайме теперь есть второй проход `self-check verifier`, который помогает:
- убирать форматный шум вроде `а)` или `48.` в начале ответа;
- сжимать слишком длинные ответы;
- проверять полноту определений и списков;
- лучше обрабатывать false-premise и prompt-injection кейсы.

Основные env-переменные:
- `ENABLE_ANSWER_VERIFIER=1` — включить verifier;
- `VERIFIER_MAX_TOKENS=320` — лимит токенов для verifier;
- `VERIFIER_LENGTH_THRESHOLD=420` — с какой длины ответа verifier начинает считать ответ рискованным по verbosity.

По умолчанию verifier выключен. Его можно включать как опциональный экспериментальный режим, если нужен дополнительный проход по сокращению и нормализации ответов.
