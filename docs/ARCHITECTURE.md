# Architecture: Universal RAG Pipeline

## Overview

Universal Retrieval-Augmented Generation (RAG) pipeline for question answering over PDF documents. Designed to work on any PDF in any language without hardcoded language strings or document-specific patterns.

**Current homework target:** Национальная стратегия развития ИИ на период до 2030 года (с изменениями 2024 г.)
**Evaluation result:** 27/30 keyword match (29/30 factually correct)

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        INDEXING (one-time)                           │
│                                                                      │
│  PDF ──► DocumentProcessor ──► StructuredChunk[]                     │
│              pymupdf4llm           ├── text                          │
│              → Markdown            ├── chunk_id, page                │
│              → smart_split()       ├── section_title                 │
│                                    └── paragraph_number              │
│                                                                      │
│  StructuredChunk[] ──► IndexStore                                    │
│                           ├── FAISS IndexFlatIP  (dense)             │
│                           │    bge-m3 embeddings, cosine sim         │
│                           ├── BM25Okapi           (sparse)           │
│                           └── StructuralIndex                        │
│                                paragraph_number → chunk_id[]         │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                        QUERY (per question)                          │
│                                                                      │
│  Question                                                            │
│    │                                                                 │
│    ├──[Thread 1]──► QueryClassifier.classify()  LLM call #1         │
│    │                 Ollama /api/chat + JSON schema                  │
│    │                 └──► QueryPlan                                  │
│    │                       ├── query_type (factoid/definitional/     │
│    │                       │              structural/analytical)     │
│    │                       ├── top_k                                 │
│    │                       ├── paragraph_refs  [4, 46, ...]         │
│    │                       ├── boost_early_chunks                    │
│    │                       └── language  (ru/en/...)                │
│    │                                                                 │
│    └──[Thread 2]──► IndexStore.dense_search()                       │
│                      bge-m3 embed + FAISS top-K                     │
│                                                                      │
│    [merge] ──► StrategyRetriever.retrieve_with_dense()              │
│                 ├── dense_ids     (Thread 2 result)                  │
│                 ├── sparse_ids    BM25 top-K                         │
│                 ├── structural_ids  StructuralIndex(paragraph_refs) │
│                 ├── head_ids      first N chunks if boost flag       │
│                 ├── RRF fusion    Reciprocal Rank Fusion             │
│                 └── CrossEncoder rerank (bge-reranker-v2-m3)        │
│                      └──► RetrievedContext[]                         │
│                                                                      │
│  RetrievedContext[] ──► AnswerGenerator.generate()  LLM call #2     │
│                          ├── SYSTEM_TEMPLATE.format(language=...)    │
│                          ├── ANSWER_TYPE_HINTS[query_type]           │
│                          ├── Ollama /api/chat + JSON schema          │
│                          └──► StructuredAnswer                       │
│                                ├── answer: str                       │
│                                ├── confidence: float  0.0–1.0        │
│                                └── evidence_ids: list[int]           │
│                                                                      │
│  Pipeline: if confidence < MIN_CONFIDENCE → "No data found."        │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Module Reference

### `rag/config.py` — AppConfig

Loads all runtime settings from `.env` via `python-dotenv`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `OLLAMA_BASE_URL` | — | **Required.** Ollama server URL |
| `OLLAMA_MODEL` | auto-detect | LLM for generation and classification |
| `CLASSIFIER_MODEL` | = OLLAMA_MODEL | Optional separate (smaller) model for classification |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | SentenceTransformer embedding model |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | CrossEncoder reranker model |
| `CHUNK_SIZE` | 1100 | Target chunk size in characters |
| `CHUNK_OVERLAP` | 220 | Overlap between consecutive chunks |
| `DENSE_TOP_K` | 40 | FAISS candidates per query |
| `SPARSE_TOP_K` | 40 | BM25 candidates per query |
| `FUSED_TOP_K` | 30 | Chunks after RRF fusion |
| `CONTEXT_TOP_K` | 5 | Final chunks passed to LLM |
| `RRF_K` | 60 | RRF smoothing constant |
| `HEAD_CHUNK_BOOST_COUNT` | 3 | Number of early chunks to boost for title queries |
| `MIN_RERANK_SCORE` | 0.02 | Floor score for LexicalFallbackRetriever |
| `MIN_CONFIDENCE` | 0.3 | Confidence threshold; below → "No data found." |
| `DOCUMENT_LANGUAGE` | `auto` | `auto` \| `ru` \| `en` \| `de` \| … Override detected language |
| `OLLAMA_TEMPERATURE` | 0.0 | Generation temperature (0 = deterministic) |
| `OLLAMA_NUM_PREDICT` | 420 | Max generation tokens |
| `OLLAMA_SEED` | — | Fixed seed for reproducibility |
| `OLLAMA_TIMEOUT_SEC` | 180 | HTTP timeout per Ollama request |
| `OLLAMA_DISABLE_THINKING` | 1 | Suppress chain-of-thought (required for qwen3.5+) |

---

### `rag/document.py` — DocumentProcessor

Converts PDF to `StructuredChunk[]` with structural metadata.

**Step 1 — PDF → Markdown**
Uses `pymupdf4llm.to_markdown()` which preserves heading levels (`# H1`, `## H2`) and page separators (`-----`). Falls back to plain `fitz` text extraction if `pymupdf4llm` is not installed.

**Step 2 — smart_split()**
Line-by-line accumulator with three structure-aware rules:

| Pattern | Regex | Action |
|---------|-------|--------|
| Markdown header | `^#{1,6}\s+(.+)$` | Flush chunk, start new section, update `section_title` |
| Numbered paragraph | `^\s{0,4}(\d{1,3})\.\s+\S` | Flush if near `CHUNK_SIZE`, update `paragraph_number` |
| List item | `^\s*([а-яa-zA-Z]\)\|[•\-\*])\s` | **Do not split** — accumulate into current chunk |

**List detection note:** The `LIST_START_RE` pattern covers:
- Russian-style list markers: `а)`, `б)`, `в)` (Cyrillic letter + parenthesis)
- Latin-style list markers: `a)`, `b)`, `c)`
- Bullet markers: `•`, `-`, `*`

This prevents splitting a numbered list item across chunks (e.g., the sub-points under paragraph 26 of the strategy). Splitting there would break BM25 retrieval for questions about those sub-points.

**Output:** First chunk is always `chunk_type="header"` (document title page). All subsequent chunks are `chunk_type="body"`.

---

### `rag/classifier.py` — QueryClassifier

Single LLM call → `QueryPlan`. Replaces all hardcoded `FACTOID_PREFIXES` / `FACTOID_KEYWORDS` from the legacy code.

**Prompt:** English-only, language-agnostic. Asks the model to return a JSON object.

**Query types and their `top_k`:**

| Type | top_k | When |
|------|-------|------|
| `factoid` | 6 | Single fact, number, date, name |
| `definitional` | 8 | "What is X", "How is X defined" |
| `structural` | 6 | References explicit paragraph/section number |
| `analytical` | 10 | Broad analysis, comparison, summary |

**paragraph_refs** — extracted paragraph numbers (e.g., `[1, 46]`) feed directly into `StructuralIndex.search()` for exact lookup.

**boost_early_chunks** — set `true` by the model when the question is about document title, date, version, or amendments. Causes the first `HEAD_CHUNK_BOOST_COUNT` chunks to be added to the RRF rank lists.

**Language detection** — LLM returns `"ru"`, `"en"`, etc. If `DOCUMENT_LANGUAGE != "auto"` in config, the config value overrides the detected language.

**Fallback:** If LLM returns invalid JSON or call fails → `QueryPlan(type="analytical", top_k=8, paragraph_refs=[], boost=False, language from config)`.

---

### `rag/index.py` — IndexStore + StructuralIndex

**IndexStore** builds three indexes from `StructuredChunk[]`:

1. **FAISS IndexFlatIP** — exact inner product search over bge-m3 embeddings (normalized → cosine similarity). `dense_search(query, k)` encodes the query with the BGE instruction prefix: `"Represent this sentence for searching relevant passages: {query}"`.

2. **BM25Okapi** — classic TF-IDF sparse retrieval. `sparse_search(query, k)` tokenizes with `[A-Za-zА-Яа-яЁё0-9]+` regex.

3. **StructuralIndex** — dict `{paragraph_number → [chunk_id, ...]}`. Built once at load time. O(1) lookup for structural queries.

**Graceful degradation:** If `sentence-transformers` or `faiss-cpu` is not installed → IndexStore skips dense index → `LexicalFallbackRetriever` is used instead.

---

### `rag/retriever.py` — StrategyRetriever

Assembles rank lists and fuses them with **Reciprocal Rank Fusion (RRF)**:

```
score(d) = Σ  1 / (k + rank(d, list_i))
```

Priority order in rank lists (highest to lowest):
1. `structural_ids` — exact paragraph match (inserted at front if `paragraph_refs` non-empty)
2. `dense_ids` — semantic similarity
3. `sparse_ids` — BM25 keyword overlap
4. `head_ids` — early document chunks (appended if `boost_early_chunks`)

After RRF, the top `fused_top_k` candidates are reranked by `bge-reranker-v2-m3` CrossEncoder (query × chunk text pairs). Final `top_k` chunks are returned.

**LexicalFallbackRetriever** — TF-IDF cosine similarity with the same boosting logic, used when ML dependencies are absent.

---

### `rag/generator.py` — AnswerGenerator

No hardcoded Russian. Two template strings:

**System prompt:**
```
"You are a document assistant. Answer ONLY based on the provided context.
If the answer is not found, set confidence to 0.0 and answer with 'N/A'.
Do not add external facts. Respond in {language}."
```

**Type hints** (appended to user message):

| Type | Instruction |
|------|-------------|
| `factoid` | "Give a precise short answer (1-2 sentences). Facts only." |
| `definitional` | "Give a complete definition (2-4 sentences). Use exact wording from context." |
| `structural` | "Quote or closely paraphrase the relevant paragraph. 2-4 sentences." |
| `analytical` | "Give a structured answer (3-6 sentences). Cover all aspects from context." |

**Structured output:** Ollama `format` field receives the full JSON schema → model is constrained to return `{answer, confidence, evidence_ids}`.

**No repair loop.** Old code had a string-matching repair loop (`"нет данных"` → retry). New code: `confidence < MIN_CONFIDENCE` → pipeline returns `"No data found in document."` programmatically.

**JSON fallback parser:** 3-level: direct `json.loads` → regex-extract `{...}` block → default empty answer.

---

### `rag/pipeline.py` — RAGPipeline

Thin orchestrator. The only component that knows about parallel execution:

```python
# classify and dense search run concurrently
with ThreadPoolExecutor(max_workers=2):
    plan      = classifier.classify(question)   # ~200-400ms
    dense_ids = retriever.dense_search(question) # ~50-100ms
# latency ≈ max(classify, dense), not sum
```

Build function `build_pipeline(config, pdf_path)`:
1. `DocumentProcessor.load(pdf_path)` → chunks
2. `IndexStore(chunks, config)` → indexes
3. Auto-selects `StrategyRetriever` (hybrid) or `LexicalFallbackRetriever`
4. Wires `QueryClassifier`, `AnswerGenerator`, `OllamaClient`

---

## Data Flow Summary

```
PDF → Markdown → StructuredChunk[] → {FAISS, BM25, StructuralIndex}

Question
  → [parallel] QueryPlan + dense_ids
  → StrategyRetriever → RetrievedContext[top_k]
  → AnswerGenerator → StructuredAnswer{answer, confidence}
  → if confidence ≥ 0.3: return answer
  → else: "No data found in document."
```

---

## Known Issues and Limitations

| Issue | Description | Workaround |
|-------|-------------|-----------|
| Q04 regression | "пункте 1" in locative case may not be parsed as `paragraph_refs=[1]` by classifier | Add Russian case examples to CLASSIFY_PROMPT |
| Q13/Q25 evaluation | Keyword evaluator uses exact substring match; Russian inflection causes false FAIL | Fix is in eval script, not in pipeline |
| First-run latency | bge-m3 model download ~560MB on first run | Pre-download with `sentence-transformers` cache |
| Ollama structured output | `format=JSON_SCHEMA` requires Ollama ≥0.4.0 | Update Ollama or remove `format` param |
| Classifier adds latency | ~200-400ms extra LLM call per question | Use `CLASSIFIER_MODEL` with a smaller/faster model |
