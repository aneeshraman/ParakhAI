"""Stage 1: document -> native-language text, via Sarvam Vision (Document Intelligence)."""

from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from pathlib import PurePath

from pypdf import PdfReader, PdfWriter
from sarvamai import SarvamAI

from .config import Settings

MIME_TYPES = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".zip": "application/zip",
}
TERMINAL_STATUSES = {"completed", "partially_completed", "failed", "rejected"}
MAX_PDF_PAGES = 10  # Sarvam Vision's per-job limit


class OCRError(RuntimeError):
    pass


@dataclass
class OCRResult:
    text: str
    pages: list[str]
    job_ids: list[str]
    status: str
    warnings: list[str] = field(default_factory=list)


def mime_type_for(filename: str) -> str:
    suffix = PurePath(filename).suffix.lower()
    if suffix not in MIME_TYPES:
        raise OCRError(f"Unsupported file type '{suffix}'. Use one of: {', '.join(MIME_TYPES)}")
    return MIME_TYPES[suffix]


def split_pdf(data: bytes, max_pages: int = MAX_PDF_PAGES) -> list[bytes]:
    """Split a PDF into consecutive parts of at most `max_pages` pages."""
    reader = PdfReader(io.BytesIO(data))
    if len(reader.pages) <= max_pages:
        return [data]
    parts = []
    for start in range(0, len(reader.pages), max_pages):
        writer = PdfWriter()
        for page in reader.pages[start : start + max_pages]:
            writer.add_page(page)
        buffer = io.BytesIO()
        writer.write(buffer)
        parts.append(buffer.getvalue())
    return parts


class SarvamOCR:
    def __init__(self, client: SarvamAI, settings: Settings):
        self.client = client
        self.settings = settings

    def digitise(
        self,
        data: bytes,
        filename: str,
        *,
        language: str | None = None,
        content_type: str = "handwritten",
    ) -> OCRResult:
        """Run digitise jobs and return the document as Markdown, one entry per page.

        `language` may be None: Sarvam Vision then treats the document as Indic and works it out.
        PDFs longer than 10 pages are split and sent as several jobs, which run in parallel.
        """
        return self.collect(self.submit(data, filename, language=language, content_type=content_type))

    def submit(
        self,
        data: bytes,
        filename: str,
        *,
        language: str | None = None,
        content_type: str = "handwritten",
    ) -> list[str]:
        """Start the digitise job(s) and return their IDs without waiting."""
        mime = mime_type_for(filename)
        parts = split_pdf(data) if mime == "application/pdf" else [data]

        kwargs = {"content_type": content_type}
        if language:
            kwargs["language"] = language
        if self.settings.ocr_model:
            kwargs["model"] = self.settings.ocr_model

        stem, suffix = PurePath(filename).stem, PurePath(filename).suffix
        job_ids = []
        for i, part in enumerate(parts):
            part_name = filename if len(parts) == 1 else f"{stem}_part{i + 1}{suffix}"
            job = self.client.doc_ai.digitise(file=[(part_name, part, mime)], **kwargs)
            job_ids.append(job.job_id)
        return job_ids

    def collect(self, job_ids: list[str]) -> OCRResult:
        """Wait for the jobs and assemble their pages in order."""
        pages: list[str] = []
        warnings: list[str] = []
        statuses = []
        for part_number, job_id in enumerate(job_ids, start=1):
            status = self._wait(job_id)
            statuses.append(status)
            label = f"Part {part_number}/{len(job_ids)}" if len(job_ids) > 1 else "OCR"
            if status in ("failed", "rejected"):
                warnings.append(f"{label} (job {job_id}) ended with status '{status}'; its pages are missing.")
                continue
            results = self.client.doc_ai.get_results(job_id)
            pages += [page_text(page) for document in results.documents for page in sorted(document.pages or [], key=_page_number)]
            if status == "partially_completed":
                warnings.append(f"{label}: {results.usage.pages_failed} page(s) could not be read.")

        if all(s in ("failed", "rejected") for s in statuses):
            raise OCRError(f"Sarvam Vision could not read the document (jobs: {', '.join(job_ids)}).")

        text = "\n\n".join(p.strip() for p in pages if p.strip())
        if not text:
            warnings.append("OCR returned no text.")
        status = "completed" if all(s == "completed" for s in statuses) else "partially_completed"
        return OCRResult(text=text, pages=pages, job_ids=job_ids, status=status, warnings=warnings)

    def _wait(self, job_id: str) -> str:
        deadline = time.monotonic() + self.settings.ocr_timeout_seconds
        while True:
            status = self.client.doc_ai.get_status(job_id).status.lower()
            if status in TERMINAL_STATUSES:
                return status
            if time.monotonic() > deadline:
                raise OCRError(f"Sarvam Vision job {job_id} did not finish in {self.settings.ocr_timeout_seconds:.0f}s.")
            time.sleep(self.settings.ocr_poll_seconds)


# The JSON results carry fields the SDK's typed model doesn't declare (page_num, blocks);
# they arrive as pydantic extras, so read them by attribute with fallbacks.
def _page_number(page) -> int:
    return getattr(page, "page_num", None) or page.page_number or 0


def page_text(page) -> str:
    """Join a page's text blocks in reading order."""
    blocks = getattr(page, "blocks", None)
    if not blocks:
        return page.content or ""
    ordered = sorted(blocks, key=lambda b: b.get("reading_order") or 0)
    return "\n\n".join(b["text"].strip() for b in ordered if (b.get("text") or "").strip())
