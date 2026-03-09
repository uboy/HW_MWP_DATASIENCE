# RAG-конвейер для вопросов по PDF-документу

Этот проект отвечает на вопросы по произвольному PDF-документу с помощью Retrieval-Augmented Generation и использует только переданный PDF как источник знаний.

Текущая цель по домашнему заданию:
- исходный документ: внешний PDF, путь к которому передаётся через `--pdf`
- официальный набор вопросов: `test_set.xlsx`
- официальный проверяющий контур: `RAGAS`
- метрики: `answer_relevancy`, `answer_correctness`, `answer_similarity`, `clarity`, `safety`

## Что делает скрипт при запуске

В самом начале CLI печатает техническую сводку, чтобы в скринкасте было видно, как работает система:
- чем разбирается PDF;
- как выполняется чанкинг;
- какие эмбеддинги используются;
- какой план retrieval применяется;
- какой реранкер и какая LLM используются;
- какие защитные правила включены для false premise и prompt injection;
- какой файл вопросов загружен;
- что в конце будет сформирован файл для сдачи.

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

### 3. Пакетный запуск на официальном наборе вопросов

Команда:

```powershell
.\venv\Scripts\python.exe -m rag.main --pdf C:\path\to\document.pdf --questions-file test_set.xlsx
```

### 4. Финальный запуск для сдачи

Это основной сценарий для домашнего задания.

Команда:

```powershell
.\venv\Scripts\python.exe -m rag.main `
  --pdf C:\path\to\document.pdf `
  --questions-file test_set.xlsx `
  --submission-out test_set_Фамилия_Имя.xlsx `
  --answers-out .scratchpad\answers_submission.csv `
  --report-out .scratchpad\answers_report.txt `
  --debug-out .scratchpad\answers_debug.csv
```

Что делает эта команда:
- читает вопросы из `test_set.xlsx`;
- отвечает на все вопросы по переданному PDF;
- создаёт файл для сдачи `test_set_Фамилия_Имя.xlsx`;
- создаёт отдельный CSV только с ответами;
- создаёт текстовый отчёт для просмотра и скринкаста;
- создаёт debug CSV с источниками и rerank score.

Важно:
- вместо `Фамилия_Имя` подставь свои реальные фамилию и имя;
- именно файл `test_set_Фамилия_Имя.xlsx` нужно сдавать как основной файл с ответами.
- если не указать `--submission-out` при запуске на `test_set.xlsx`, скрипт по умолчанию предложит имя `test_set_Фамилия_Имя.xlsx`.

## Что именно куда вводить

### Если хочешь просто показать работу вживую

1. Открой PowerShell в папке проекта.
2. Выполни:

```powershell
.\venv\Scripts\python.exe -m rag.main --pdf C:\path\to\document.pdf
```

3. Введи вопросы построчно.
4. Поставь пустую строку.
5. Дождись вывода ответов.

### Если хочешь сделать итоговый файл для сдачи

Ничего вручную вводить не нужно. Просто запусти:

```powershell
.\venv\Scripts\python.exe -m rag.main `
  --pdf C:\path\to\document.pdf `
  --questions-file test_set.xlsx `
  --submission-out test_set_Иванов_Иван.xlsx `
  --answers-out .scratchpad\answers_submission.csv `
  --report-out .scratchpad\answers_report.txt `
  --debug-out .scratchpad\answers_debug.csv
```

После этого сдавать нужно файл:

```text
test_set_Иванов_Иван.xlsx
```

или такой же файл с твоими реальными фамилией и именем.

## Аргументы CLI

| Аргумент | Что означает |
|---|---|
| `--pdf` | Путь к исходному PDF |
| `--question` | Один вопрос |
| `--questions-file` | Файл с вопросами `.txt`, `.csv` или `.xlsx` |
| `--submission-out` | Итоговый файл для сдачи `.csv` или `.xlsx` |
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
| `test_set_Фамилия_Имя.xlsx` | Главный файл для сдачи, в нём заполняется колонка `answer` |
| `.scratchpad\answers_submission.csv` | CSV только с ответами |
| `.scratchpad\answers_report.txt` | Читаемый форматированный отчёт |
| `.scratchpad\answers_debug.csv` | Отладочный CSV с источниками и оценками |

## Что нужно сдавать по заданию

По условиям домашнего задания нужно сдать:
- `py` или `ipynb` файлы с реализацией, если решение написано на Python;
- файл с ответами в формате `xlsx`, `xls` или `csv`;
- имя файла с ответами строго по шаблону `test_set_Фамилия_Имя.xlsx` или `.xls` / `.csv`;
- ответы должны быть записаны в колонку `answer`;
- ссылку на скринкаст работы системы.

Если есть деплой, можно дополнительно дать ссылку на готовый RAG.

## Что делать с метриками

Проверка будет идти через `RAGAS`.

Официальные метрики:
- `answer_relevancy`
- `answer_correctness`
- `answer_similarity`
- `clarity`
- `safety`

Вес метрик:
- для большинства вопросов:
  - `0.35 * answer_relevancy`
  - `0.35 * answer_correctness`
  - `0.20 * answer_similarity`
  - `0.05 * clarity`
  - `0.05 * safety`
- для части вопросов:
  - `0.55 * answer_correctness`
  - `0.30 * answer_similarity`
  - `0.10 * clarity`
  - `0.05 * safety`

Итог считается как среднее по `13` вопросам.

Пороговые значения из задания:
- выше `0.7` -> 1 балл;
- выше `0.8` -> 2 балла;
- выше `0.9` -> 4 балла.

Важно понимать:
- отдельный файл с метриками в условии сдачи не требуется;
- но сами ответы будут проверяться именно по этим метрикам;
- поэтому ответы должны быть короткими, точными, без воды и без выдуманных фактов.

В этом репозитории теперь есть локальный `proxy-eval` под эти же 5 метрик. Это не официальный внешний `RAGAS`, но он позволяет заранее увидеть слабые вопросы и примерный score-профиль, если у тебя есть собственный gold-файл для сравнения.

### Локальная оценка 5 метрик

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
- считает `answer_relevancy`, `answer_correctness`, `answer_similarity`, `clarity`, `safety`;
- пишет CSV с метриками по каждому вопросу;
- пишет markdown summary со средними значениями и самыми слабыми вопросами.

Важно:
- это локальная proxy-оценка, а не гарантированная копия внешнего проверяющего контура;
- gold-файл нужно передать явно через `--gold`;
- в summary выводятся обе формулы взвешивания из домашнего задания:
  - `weighted_general`
  - `weighted_correctness_heavy`

## Практический сценарий сдачи

1. Запусти финальную команду:

```powershell
.\venv\Scripts\python.exe -m rag.main `
  --pdf C:\path\to\document.pdf `
  --questions-file test_set.xlsx `
  --submission-out test_set_ТвояФамилия_ТвоёИмя.xlsx `
  --answers-out .scratchpad\answers_submission.csv `
  --report-out .scratchpad\answers_report.txt `
  --debug-out .scratchpad\answers_debug.csv
```

2. Проверь, что появился файл:

```text
test_set_ТвояФамилия_ТвоёИмя.xlsx
```

3. Открой его и убедись, что ответы лежат в колонке `answer`.
4. Запиши скринкаст:
- запуск команды;
- стартовую техническую сводку;
- список вопросов;
- итоговые ответы;
- открытие итогового файла.

5. На сдачу отправь:
- код проекта;
- файл `test_set_ТвояФамилия_ТвоёИмя.xlsx`;
- ссылку на скринкаст.

## Качество ответов

Пайплайн уже настроен на обновлённый набор вопросов:
- поддерживает `xlsx`;
- корректно обрабатывает ложные предпосылки;
- не выдумывает ответ на вопрос, которого нет в документе;
- защищён от prompt injection в тестовом наборе;
- формирует submission-ready файл.

Формальная численная оценка по `RAGAS` в репозитории пока не считается, но итоговые ответы подготовлены под критерии задания.

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

По умолчанию verifier выключен. В текущем проекте это сделано специально: на `test_set.xlsx` он заметно улучшает отдельные ответы по `clarity`, но локальный proxy-eval показывает, что как дефолтный режим он не всегда улучшает общий итоговый score. Поэтому базовый submission-flow остаётся без него, а verifier доступен как опциональный экспериментальный режим.
