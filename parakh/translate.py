"""Stage 2: native-language text -> English, via Sarvam's translation API."""

from __future__ import annotations

import re

from sarvamai import SarvamAI

from .config import Settings
from .languages import LANGUAGES, MAYURA_LANGUAGES

CHAR_LIMITS = {"mayura:v1": 1000, "sarvam-translate:v1": 2000}
IDENTIFY_CHAR_LIMIT = 1000

# Sentence ends for Latin scripts, Devanagari/Bengali danda (। ॥), Urdu full stop (۔) and line breaks.
_SENTENCE_END = re.compile(r"(?<=[.!?।॥۔\n])\s+")


def chunk_text(text: str, limit: int) -> list[str]:
    """Split text into pieces of at most `limit` characters, breaking at paragraphs, then
    sentences, then whitespace, so that no request to the translation API is over its limit."""
    chunks: list[str] = []
    current = ""

    def flush():
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = ""

    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        # Pack short paragraphs into one request to keep the request count (and rate-limit use) low.
        for i, piece in enumerate(_fit(paragraph, limit)):
            separator = " " if i else "\n\n"
            candidate = f"{current}{separator}{piece}" if current else piece
            if len(candidate) <= limit:
                current = candidate
            else:
                flush()
                current = piece
    flush()
    return chunks


def _fit(paragraph: str, limit: int) -> list[str]:
    if len(paragraph) <= limit:
        return [paragraph]
    pieces: list[str] = []
    current = ""
    for sentence in _SENTENCE_END.split(paragraph):
        for part in _hard_split(sentence, limit):
            candidate = f"{current} {part}" if current else part
            if len(candidate) <= limit:
                current = candidate
            else:
                pieces.append(current)
                current = part
    if current:
        pieces.append(current)
    return pieces


def _hard_split(sentence: str, limit: int) -> list[str]:
    parts = []
    while len(sentence) > limit:
        cut = sentence.rfind(" ", 0, limit)
        if cut <= 0:
            cut = limit
        parts.append(sentence[:cut].strip())
        sentence = sentence[cut:].strip()
    if sentence:
        parts.append(sentence)
    return parts


class SarvamTranslator:
    def __init__(self, client: SarvamAI, settings: Settings):
        self.client = client
        self.model = settings.translate_model
        if self.model not in CHAR_LIMITS:
            raise ValueError(f"Unknown translation model '{self.model}'. Use one of: {', '.join(CHAR_LIMITS)}")

    def detect_language(self, text: str) -> str | None:
        """Return a supported BCP-47 code such as 'hi-IN', or None if Sarvam can't tell."""
        sample = text.strip()[:IDENTIFY_CHAR_LIMIT]
        if not sample:
            return None
        code = self.client.text.identify_language(input=sample).language_code
        return code if code in LANGUAGES else None

    def supports(self, language: str) -> bool:
        if self.model == "mayura:v1":
            return language in MAYURA_LANGUAGES
        return language in LANGUAGES

    def translate(self, text: str, source: str, target: str = "en-IN") -> str:
        if source == target or not text.strip():
            return text
        return "\n\n".join(translation for _, translation in self.translate_chunks(text, source, target))

    def translate_chunks(self, text: str, source: str, target: str = "en-IN") -> list[tuple[str, str]]:
        """Translate chunk by chunk and return (source chunk, translation) pairs, so each
        pair can be checked on its own."""
        return [(chunk, self._translate_one(chunk, source, target)) for chunk in chunk_text(text, CHAR_LIMITS[self.model])]

    def back_translate(self, translation: str, source: str, target: str = "en-IN") -> str:
        """Translate `translation` (in `target`) back into `source`, for round-trip checks.
        An English chunk can come out longer than the limit, so it is re-chunked."""
        return " ".join(self._translate_one(c, target, source) for c in chunk_text(translation, CHAR_LIMITS[self.model]))

    def _translate_one(self, chunk: str, source: str, target: str) -> str:
        return self.client.text.translate(
            input=chunk,
            source_language_code=source,
            target_language_code=target,
            model=self.model,
        ).translated_text
