"""Cache paid API results on disk so evaluation reruns don't spend credits twice."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from sarvamai import SarvamAI

from parakh.config import Settings
from parakh.ocr import OCRResult, SarvamOCR

CACHE_DIR = Path(__file__).parent / "cache"


def doc_dir(pdf: Path) -> Path:
    path = CACHE_DIR / pdf.stem
    path.mkdir(parents=True, exist_ok=True)
    return path


def sarvam_ocr() -> SarvamOCR:
    settings = Settings()
    return SarvamOCR(SarvamAI(api_subscription_key=settings.require_sarvam_key()), settings)


def cached_ocr(pdf: Path, *, language: str | None, content_type: str, refresh: bool = False) -> OCRResult:
    """OCR `pdf` once. Job IDs are saved as soon as the jobs are submitted, so a crash
    while waiting or fetching can be resumed without paying for the pages again."""
    base = doc_dir(pdf) / f"ocr_{language or 'auto'}_{content_type}"
    result_path, jobs_path = base.with_suffix(".json"), base.with_suffix(".jobs.json")
    if result_path.exists() and not refresh:
        return OCRResult(**json.loads(result_path.read_text(encoding="utf-8")))

    ocr = sarvam_ocr()
    if jobs_path.exists() and not refresh:
        job_ids = json.loads(jobs_path.read_text(encoding="utf-8"))
    else:
        job_ids = ocr.submit(pdf.read_bytes(), pdf.name, language=language, content_type=content_type)
        jobs_path.write_text(json.dumps(job_ids), encoding="utf-8")

    result = ocr.collect(job_ids)
    result_path.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")
    return result
