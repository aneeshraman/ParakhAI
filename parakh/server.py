"""Web app: uvicorn parakh.server:app --reload"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sarvamai.core.api_error import ApiError

from .languages import CONTENT_TYPES, LANGUAGES
from .llm import LLMError
from .ocr import OCRError
from .pipeline import DEFAULT_TASK, Pipeline

STATIC = Path(__file__).parent / "static"

app = FastAPI(title="ParakhAI")


@lru_cache(maxsize=1)
def get_pipeline() -> Pipeline:
    return Pipeline.from_settings()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/options")
def options() -> dict:
    return {"languages": LANGUAGES, "content_types": CONTENT_TYPES, "default_task": DEFAULT_TASK}


# A plain `def` so FastAPI runs the blocking pipeline in its threadpool.
@app.post("/api/process")
def process(
    file: UploadFile = File(...),
    language: str = Form(""),
    content_type: str = Form("handwritten"),
    task: str = Form(""),
    use_llm: bool = Form(True),
) -> dict:
    if language and language not in LANGUAGES:
        raise HTTPException(400, f"Unknown language '{language}'.")
    if content_type not in CONTENT_TYPES:
        raise HTTPException(400, f"Unknown content type '{content_type}'.")
    try:
        result = get_pipeline().run(
            file.file.read(),
            file.filename or "upload",
            language=language or None,
            content_type=content_type,
            task=task.strip() or DEFAULT_TASK,
            use_llm=use_llm,
        )
    except (OCRError, LLMError) as e:
        raise HTTPException(502, str(e))
    except ApiError as e:
        raise HTTPException(502, f"Sarvam API error {e.status_code}: {e.body}")
    return result.to_dict()
