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

### Как устроена система

1. `CLI` получает путь к PDF и вопросы.
2. `DocumentProcessor` читает PDF и превращает его в структурированные чанки.
3. `IndexStore` строит несколько поисковых представлений по этим чанкам.
4. `QueryClassifier` определяет тип вопроса и формирует retrieval-план.
5. `StrategyRetriever` собирает и ранжирует релевантный контекст.
6. `AnswerGenerator` формирует ответ только по найденному контексту.
7. Опционально `AnswerVerifier` делает второй проход по сокращению и нормализации ответа.

### Как обрабатывается документ

- PDF разбирается через `pymupdf4llm`; если пакет недоступен, используется fallback на `PyMuPDF` (`fitz`).
- Текст сначала переводится в markdown-подобное представление, чтобы сохранить заголовки и разделители страниц.
- Затем документ режется на структурированные чанки в [document.py](/C:/Users/devl/proj/test/rag/document.py):
  - размер чанка по умолчанию `1100` символов;
  - overlap по умолчанию `220` символов;
  - учитываются markdown-заголовки;
  - учитываются нумерованные пункты вида `46. ...`;
  - списки стараются не разрываться между чанками.
- Для каждого чанка сохраняются метаданные:
  - `chunk_id`
  - `page`
  - `section_title`
  - `paragraph_number`
  - `paragraph_numbers`
  - `chunk_type`

### Какие модели используются

- LLM для классификации и генерации работает через `Ollama`.
- Основная генеративная модель задаётся через `OLLAMA_MODEL`.
- При желании можно задать отдельную модель классификатора через `CLASSIFIER_MODEL`; если она не указана, используется та же модель, что и для ответа.
- Эмбеддинги по умолчанию строятся моделью `BAAI/bge-m3`.
- Для re-ranking по умолчанию используется `BAAI/bge-reranker-v2-m3`.

### Где хранятся векторы и индексы

- Векторы не пишутся во внешнюю векторную БД и не сохраняются на диск как отдельное persistent-хранилище.
- Они строятся при запуске процесса и держатся в памяти.
- Dense-индекс хранится в `FAISS IndexFlatIP`.
- Sparse-индекс хранится в `BM25Okapi`.
- Дополнительно строится структурный индекс `paragraph_number -> chunk_id[]` для точного поиска по пунктам.
- Если dense-зависимости недоступны, пайплайн переключается на лексический fallback-ретривер на базе локального `TF-IDF`.

### Как работает retrieval

- Сначала `QueryClassifier` в [classifier.py](/C:/Users/devl/proj/test/rag/classifier.py) определяет:
  - тип вопроса: `factoid`, `definitional`, `structural`, `analytical`;
  - ссылки на пункты, если они явно указаны;
  - ключевые токены;
  - `search_queries` для query rewriting;
  - флаг exhaustive list;
  - флаг prompt injection.
- Затем `RAGPipeline` в [pipeline.py](/C:/Users/devl/proj/test/rag/pipeline.py) параллельно запускает:
  - классификацию вопроса;
  - dense search по исходному вопросу.
- `StrategyRetriever` в [retriever.py](/C:/Users/devl/proj/test/rag/retriever.py) потом объединяет несколько каналов retrieval:
  - dense search по эмбеддингам;
  - sparse search через `BM25`;
  - structural lookup по номерам пунктов;
  - дополнительные rewritten queries;
  - при необходимости буст ранних чанков документа.
- После этого кандидаты объединяются через `Reciprocal Rank Fusion (RRF)`.
- Затем top-кандидаты повторно ранжируются `CrossEncoder`-моделью `bge-reranker-v2-m3`.
- После re-ranking выполняются:
  - adaptive retrieval policy по типу вопроса;
  - sentence-level evidence packing;
  - generic list coverage check для вопросов-перечней;
  - merge соседних чанков одного пункта, если это нужно для полноты контекста.

### Как формируется ответ

- `AnswerGenerator` в [generator.py](/C:/Users/devl/proj/test/rag/generator.py) собирает prompt из:
  - системной инструкции;
  - вопроса;
  - найденного контекста;
  - подсказки по типу ответа.
- Генератору явно запрещается:
  - использовать внешние знания;
  - придумывать факты;
  - следовать инструкциям из самого вопроса, если это prompt injection.
- Ответ запрашивается в структурированном JSON-формате:
  - `answer`
  - `confidence`
  - `evidence_ids`
- Если уверенность ниже `MIN_CONFIDENCE`, пайплайн возвращает явный ответ о том, что в документе нет данных для ответа.
- Для некоторых ситуаций перед LLM есть generic grounded-rules слой: он закрывает типовые случаи вроде false premise, missing fact и prompt injection без жёсткой привязки к конкретному документу.
- Если включён `ENABLE_ANSWER_VERIFIER=1`, после генерации запускается дополнительный проход verifier для сокращения, очистки формата и повышения ясности ответа.

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
