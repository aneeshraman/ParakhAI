"""Checks on a translation, chunk by chunk, before it reaches the LLM.

Cheap deterministic checks run on every chunk: numbers and formula symbols must survive,
no source script may leak through untranslated, and the length must be plausible. Two
model-based checks catch what those can't: a back-translation (target -> source) compared
with the original, and an LLM judge that reads each source/translation pair.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

from .languages import language_name

if TYPE_CHECKING:
    from .llm import LLM
    from .translate import SarvamTranslator

# Defaults, calibrated on the Hindi physics evaluation in eval/ (see eval/README.md).
BACK_SIMILARITY_WARN = 0.45
LENGTH_RATIO_RANGE = (0.45, 2.2)
MIN_CHARS_FOR_RATIO = 60
JUDGE_BATCH_CHARS = 12000

_NUMBER = re.compile(r"\d+(?:\.\d+)?")
# Latin letters and Greek/maths symbols inside Indic text are almost always formula
# variables or units (ε, λ, B, eV, Hz, ...), which a translation must keep unchanged.
_SYMBOL_TOKEN = re.compile(r"[A-Za-zΑ-Ωα-ωµ∞√∆∇°]+")
_INDIC = re.compile(r"[ऀ-෿]")


@dataclass
class Issue:
    chunk: int
    check: str  # numbers | symbols | script | length | back_translation | judge
    severity: str  # error | warning
    message: str


@dataclass
class ChunkReport:
    index: int
    source: str
    translation: str
    back_translation: str | None = None
    back_similarity: float | None = None
    issues: list[Issue] = field(default_factory=list)


@dataclass
class TranslationReport:
    chunks: list[ChunkReport]

    @property
    def issues(self) -> list[Issue]:
        return [issue for chunk in self.chunks for issue in chunk.issues]

    def summary(self) -> str:
        errors = sum(i.severity == "error" for i in self.issues)
        warnings = sum(i.severity == "warning" for i in self.issues)
        flagged = sum(bool(c.issues) for c in self.chunks)
        return f"{flagged}/{len(self.chunks)} translated chunks flagged ({errors} errors, {warnings} warnings)."

    def to_dict(self) -> dict:
        return {"summary": self.summary(), "chunks": [asdict(c) for c in self.chunks]}


# --- deterministic checks ----------------------------------------------------------------


def ascii_digits(text: str) -> str:
    """Map every Unicode digit (Devanagari ०-९, Bengali, superscripts, ...) to 0-9."""
    return "".join(str(unicodedata.digit(ch)) if unicodedata.digit(ch, None) is not None else ch for ch in text)


def numbers(text: str) -> Counter:
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", ascii_digits(text))  # 1,000 -> 1000
    return Counter(_NUMBER.findall(text))


def symbol_tokens(text: str) -> set[str]:
    return set(_SYMBOL_TOKEN.findall(text))


def check_numbers(index: int, source: str, translation: str) -> list[Issue]:
    src, tgt = numbers(source), numbers(translation)
    issues = []
    if missing := src - tgt:
        issues.append(Issue(index, "numbers", "error", f"Numbers in the original missing from the translation: {_fmt(missing)}"))
    if extra := tgt - src:
        issues.append(Issue(index, "numbers", "warning", f"Numbers in the translation not in the original: {_fmt(extra)}"))
    return issues


def check_symbols(index: int, source: str, translation: str) -> list[Issue]:
    missing = symbol_tokens(source) - symbol_tokens(translation)
    if not missing:
        return []
    return [Issue(index, "symbols", "warning", f"Formula symbols or units missing from the translation: {', '.join(sorted(missing))}")]


def check_script(index: int, translation: str, target: str) -> list[Issue]:
    if target != "en-IN":
        return []
    leaked = [m.start() for m in _INDIC.finditer(translation)]
    if not leaked:
        return []
    start = max(0, leaked[0] - 20)
    snippet = translation[start : leaked[0] + 40].replace("\n", " ")
    return [Issue(index, "script", "warning", f"{len(leaked)} untranslated Indic characters, e.g. “…{snippet}…”")]


def check_length(index: int, source: str, translation: str) -> list[Issue]:
    if not translation.strip():
        return [Issue(index, "length", "error", "Translation is empty.")]
    if len(source) < MIN_CHARS_FOR_RATIO:
        return []
    ratio = len(translation) / len(source)
    low, high = LENGTH_RATIO_RANGE
    if ratio < low:
        return [Issue(index, "length", "warning", f"Translation is only {ratio:.0%} of the original's length; content may be missing.")]
    if ratio > high:
        return [Issue(index, "length", "warning", f"Translation is {ratio:.0%} of the original's length; content may be added or repeated.")]
    return []


def basic_checks(index: int, source: str, translation: str, target: str = "en-IN") -> list[Issue]:
    return (
        check_length(index, source, translation)
        + check_numbers(index, source, translation)
        + check_symbols(index, source, translation)
        + check_script(index, translation, target)
    )


def chrf(reference: str, hypothesis: str, max_n: int = 6, beta: float = 2.0) -> float:
    """Character n-gram F-score (chrF, Popović 2015), 0..1. Works for any script."""
    ref, hyp = re.sub(r"\s+", "", reference), re.sub(r"\s+", "", hypothesis)
    scores = []
    for n in range(1, max_n + 1):
        ref_grams = Counter(ref[i : i + n] for i in range(len(ref) - n + 1))
        hyp_grams = Counter(hyp[i : i + n] for i in range(len(hyp) - n + 1))
        if not ref_grams or not hyp_grams:
            continue
        overlap = sum((ref_grams & hyp_grams).values())
        precision, recall = overlap / sum(hyp_grams.values()), overlap / sum(ref_grams.values())
        if precision + recall:
            scores.append((1 + beta**2) * precision * recall / (beta**2 * precision + recall))
        else:
            scores.append(0.0)
    return sum(scores) / len(scores) if scores else 0.0


def _fmt(counter: Counter) -> str:
    return ", ".join(f"{n} (×{c})" if c > 1 else n for n, c in sorted(counter.items()))


# --- model-based checks ------------------------------------------------------------------

JUDGE_SYSTEM = """\
You review machine translations of handwritten exam answers into English. For each numbered \
pair, compare the translation with the original and report only errors of meaning:
- a statement whose sense changed (e.g. a negation added or lost, more/less, increases/decreases)
- a technical term translated wrongly (e.g. maxima vs minima, interference vs diffraction)
- a number, unit, symbol or formula that was changed, dropped or added
- a sentence or clause that was dropped or invented
Ignore style, word order, grammar and spelling, and ignore errors that are already present in \
the original. The original came from handwriting OCR and may itself be garbled; only report \
places where the translation does not say what the original says.

Reply with JSON only, no prose and no code fence:
{"issues": [{"chunk": <number>, "severity": "error" | "warning", "problem": "<one sentence>", \
"original": "<short excerpt>", "translation": "<short excerpt>"}]}
Use "error" when a reader would learn something false from the translation, "warning" when \
it is incomplete or unclear. Reply {"issues": []} if every pair is faithful."""


def judge_prompt(pairs: list[tuple[int, str, str]], language: str) -> str:
    parts = []
    for index, source, translation in pairs:
        parts += [f'<pair chunk="{index}">', f"<original>\n{source}\n</original>", f"<translation>\n{translation}\n</translation>", "</pair>"]
    return f"Original language: {language_name(language)}\n\n" + "\n".join(parts)


def parse_judge(reply: str, valid_chunks: set[int]) -> list[Issue]:
    start, end = reply.find("{"), reply.rfind("}")
    try:
        items = json.loads(reply[start : end + 1])["issues"]
    except (ValueError, KeyError, TypeError):
        return [Issue(min(valid_chunks), "judge", "warning", "The judge's reply could not be parsed; this batch was not reviewed.")]
    issues = []
    for item in items:
        chunk = item.get("chunk")
        if chunk not in valid_chunks:
            continue
        severity = "error" if item.get("severity") == "error" else "warning"
        detail = item.get("problem", "").strip()
        if item.get("original") or item.get("translation"):
            detail += f" Original: “{item.get('original', '')}” → translation: “{item.get('translation', '')}”."
        issues.append(Issue(chunk, "judge", severity, detail))
    return issues


class TranslationChecker:
    def __init__(self, translator: SarvamTranslator, judge: LLM | None = None, *, back_translate: bool = True):
        self.translator = translator
        self.judge = judge
        self.back_translate = back_translate

    def check(self, pairs: list[tuple[str, str]], source: str, target: str = "en-IN") -> TranslationReport:
        chunks = [ChunkReport(i, s, t, issues=basic_checks(i, s, t, target)) for i, (s, t) in enumerate(pairs, start=1)]
        if self.back_translate:
            for chunk in chunks:
                self.add_back_translation(chunk, source, target)
        if self.judge:
            for issue in self.run_judge(chunks, source):
                chunks[issue.chunk - 1].issues.append(issue)
        return TranslationReport(chunks)

    def add_back_translation(self, chunk: ChunkReport, source: str, target: str) -> None:
        if not chunk.translation.strip():
            return
        chunk.back_translation = self.translator.back_translate(chunk.translation, source, target)
        chunk.back_similarity = round(chrf(chunk.source, chunk.back_translation), 3)
        if chunk.back_similarity < BACK_SIMILARITY_WARN:
            chunk.issues.append(
                Issue(
                    chunk.index,
                    "back_translation",
                    "warning",
                    f"Translating back to {language_name(source)} gives text only {chunk.back_similarity:.0%} similar to the original.",
                )
            )

    def run_judge(self, chunks: list[ChunkReport], source: str) -> list[Issue]:
        issues: list[Issue] = []
        batch: list[tuple[int, str, str]] = []
        size = 0
        for chunk in chunks:
            pair_size = len(chunk.source) + len(chunk.translation)
            if batch and size + pair_size > JUDGE_BATCH_CHARS:
                issues += self._judge_batch(batch, source)
                batch, size = [], 0
            batch.append((chunk.index, chunk.source, chunk.translation))
            size += pair_size
        if batch:
            issues += self._judge_batch(batch, source)
        return issues

    def _judge_batch(self, batch: list[tuple[int, str, str]], source: str) -> list[Issue]:
        reply = self.judge.generate(JUDGE_SYSTEM, judge_prompt(batch, source))
        return parse_judge(reply, {index for index, _, _ in batch})
