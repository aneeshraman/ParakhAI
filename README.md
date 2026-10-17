# ParakhAI

Handwritten or printed Indian-language documents → native-script text → English → an open-weight LLM (Gemma).

```
document ──► Sarvam Vision ──► Sarvam language ID ──► Sarvam Translate ──► Gemma 4
(PDF/JPG/PNG)  (handwriting OCR)   (if --lang not given)    (→ English)       (reads original + English)
```

Gemma gets **both** the original-script OCR text and the English translation, and is told to check one against the other. This catches handwriting misreads that a translation alone would turn into confident, wrong English.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
copy .env.example .env      # then fill in the two keys
```

| Key | Where | Cost |
|---|---|---|
| `SARVAM_API_KEY` | https://dashboard.sarvam.ai | free credits on signup |
| `GEMINI_API_KEY` | https://aistudio.google.com/apikey | free tier, no card; serves Gemma 4 |

Check your live Gemma quota at https://aistudio.google.com/rate-limit. Google no longer publishes fixed free-tier numbers.

## Use

```powershell
# CLI
.venv\Scripts\python -m parakh letter.jpg --lang hi-IN
.venv\Scripts\python -m parakh notes.pdf --content-type mixed --task "Extract every date and amount as a table" --json out.json
.venv\Scripts\python -m parakh scan.png --no-llm          # OCR + translation only

# Web app at http://127.0.0.1:8000
.venv\Scripts\uvicorn parakh.server:app --reload

# Tests (offline, no keys needed)
.venv\Scripts\python -m pytest
```

## Switching the LLM

Any OpenAI-compatible `/chat/completions` server works. Set it in `.env`:

| Option | `LLM_PROVIDER` | `LLM_BASE_URL` | `LLM_MODEL` | Notes |
|---|---|---|---|---|
| Google AI Studio (default) | `gemini` | n/a | `gemma-4-31b-it` / `gemma-4-26b-a4b-it` | highest free quota |
| OpenRouter | `openai_compatible` | `https://openrouter.ai/api/v1` | `google/gemma-4-31b-it:free` | 20 req/min, 50 req/day free |
| Ollama (local) | `openai_compatible` | `http://localhost:11434/v1` | your `ollama list` tag | no limits, needs a GPU |

## Limits to know

- **Sarvam Vision:** PDFs up to 10 pages per job; ZIPs up to 10 page images (flat, ordered by filename).
- **Translation:** `sarvam-translate:v1` (default) covers all 22 scheduled languages. Text is chunked to its 2000-character limit at paragraph and sentence breaks (`।`, `॥` and `.` included).
- If no `--lang` is given, Sarvam Vision runs in Indic mode and the language is detected afterwards from the first 1000 characters. Pass `--lang en-IN` for English documents.

## Layout

```
parakh/
  ocr.py        Sarvam Vision digitise job: submit, poll, collect Markdown pages
  translate.py  language ID + chunked translation
  llm.py        Gemma via Google AI Studio, or any OpenAI-compatible server
  pipeline.py   the three stages wired together, plus the prompt
  cli.py        python -m parakh
  server.py     FastAPI app; static/index.html is the UI
tests/          offline tests with fake Sarvam/LLM clients
```
