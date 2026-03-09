# Screencast Recording Guide

Step-by-step instructions for recording a demo of the RAG pipeline.

---

## Prerequisites (check before recording)

```powershell
# 1. Ollama server responds
curl http://localhost:11434/api/tags

# 2. Model is loaded (fast first response)
# Run a warm-up question once before recording:
python -m rag.main --pdf "C:\path\to\document.pdf" --question "тест"

# 3. The PDF is in the project root
ls "C:\path\to\document.pdf"

# 4. test\test_questions_30.csv exists
ls test\test_questions_30.csv
```

---

## Recording Checklist

### Step 1 — Open terminal in project root

```
cd C:\Users\devl\proj\test
```

Show the project folder contents:
```powershell
ls
```
Viewer must see: `rag/`, `requirements.txt`, `.env`, the PDF, `test/`.

---

### Step 2 — Activate virtual environment

```powershell
venv\Scripts\activate
```

---

### Step 3 — Show the questions file

```powershell
type test\test_questions_30.csv
```
Or open it in a text editor. Viewer must see the 30 questions.

---

### Step 4 — Run the batch job (main demo)

```powershell
python -m rag.main `
  --pdf "C:\path\to\document.pdf" `
  --questions-file test\test_questions_30.csv `
  --answers-out answers_submission.csv
```

**What the viewer will see during run:**
- `Loading document: ...` — PDF loading
- `Chunks loaded: N` — chunking complete
- `Retriever mode: hybrid (bge-m3 + bm25 + rrf + reranker)` — index built
- Progress: `[1/30] На какой период рассчитана...`
- `Answered: 30`
- `Submission file: answers_submission.csv`
- `Debug file: answers_debug.csv`

---

### Step 5 — Show the output file

```powershell
type answers_submission.csv
```

Viewer must see all 30 answers in the CSV.

Optionally show the debug file with sources:
```powershell
# Show first 5 rows of debug file
python -c "
import csv
with open('answers_debug.csv', encoding='utf-8') as f:
    for i, row in enumerate(csv.DictReader(f)):
        print(f'Q{i+1}: {row[\"question\"][:50]}')
        print(f'   {row[\"answer\"][:100]}')
        print(f'   sources: {row[\"sources\"]}')
        print()
        if i >= 4: break
"
```

---

### Step 6 — Optional: single question demo

Show the pipeline works interactively:

```powershell
python -m rag.main `
  --pdf "C:\path\to\document.pdf" `
  --question "Какие основные механизмы реализации стратегии названы в пункте 46?"
```

Expected output:
```
Основными механизмами реализации Стратегии являются: «дорожная карта»
развития высокотехнологичного направления «Искусственный интеллект»...
sources: [82, 84, 83, ...]
```

---

## Recording Tips

| Tip | Details |
|-----|---------|
| **Font size** | Increase terminal font to 14–16pt before recording |
| **Window size** | Maximize terminal or use 1920×1080 |
| **Disable notifications** | Windows: Focus Assist → Priority Only |
| **Pre-warm model** | Run once before recording so first answer comes in ~5s not ~30s |
| **Screen recorder** | OBS Studio (free), or Windows built-in `Win+G` |
| **Keep answers visible** | Scroll slowly through `answers_submission.csv` at the end |

---

## Consistency Check

Before submitting, verify the recorded file matches the submitted file:

```powershell
# Re-run with same seed produces identical answers
python -m rag.main `
  --pdf "C:\path\to\document.pdf" `
  --questions-file test\test_questions_30.csv `
  --answers-out answers_submission_verify.csv

# Compare (should be identical)
fc answers_submission.csv answers_submission_verify.csv
```

If files differ → check `OLLAMA_SEED` and `OLLAMA_TEMPERATURE=0.0` are set in `.env`.
