"""OCR accuracy against reference transcriptions."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

# Printed parts of an MP Board answer sheet that OCR picks up but the references leave out.
_BOILERPLATE = [
    re.compile(r"प्रश?्?न\s*क्र\.?"),
    re.compile(r"\bB\s*S\s*E\b"),
]
_PAGE_TOTAL_LINE = re.compile(r"कुल अंक|के अंक|योग पूर्व|पृष्ठ\s*\d")
_SYMBOLS_ONLY_LINE = re.compile(r"[\s\d+=()①-⑳.,|]*")
_DEVANAGARI_WORD = re.compile(r"[ऀ-ॣ०-ॿ]+")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    lines = []
    for line in text.splitlines():
        for pattern in _BOILERPLATE:
            line = pattern.sub(" ", line)
        if (_PAGE_TOTAL_LINE.search(line) and len(line) < 50) or _SYMBOLS_ONLY_LINE.fullmatch(line):
            continue
        lines.append(line)
    text = " ".join(lines)
    text = text.replace("$", " ").replace("|", "।")  # LaTeX delimiters; ASCII bar written for a danda
    return re.sub(r"\s+", " ", text).strip()


def devanagari_words(text: str) -> list[str]:
    return _DEVANAGARI_WORD.findall(normalize(text))


def edit_distance(a: str | list, b: str | list) -> int:
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        current = [i]
        for j, y in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def cer(reference: str, hypothesis: str) -> float:
    ref = normalize(reference)
    return edit_distance(ref, normalize(hypothesis)) / max(len(ref), 1)


def hindi_cer(reference: str, hypothesis: str) -> float:
    """CER over Devanagari words only: ignores maths notation and Latin symbols."""
    ref = " ".join(devanagari_words(reference))
    return edit_distance(ref, " ".join(devanagari_words(hypothesis))) / max(len(ref), 1)


def hindi_wer(reference: str, hypothesis: str) -> float:
    ref = devanagari_words(reference)
    return edit_distance(ref, devanagari_words(hypothesis)) / max(len(ref), 1)


def word_f1(reference: str, hypothesis: str) -> float:
    """Bag-of-words F1 over Devanagari words; insensitive to reading order."""
    ref, hyp = Counter(devanagari_words(reference)), Counter(devanagari_words(hypothesis))
    overlap = sum((ref & hyp).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / sum(hyp.values()), overlap / sum(ref.values())
    return 2 * precision * recall / (precision + recall)


def word_diff(reference: str, hypothesis: str, limit: int = 12) -> list[tuple[str, str]]:
    """The most common (reference word -> OCR word) substitutions, aligned by edit distance."""
    ref, hyp = devanagari_words(reference), devanagari_words(hypothesis)
    rows = len(ref) + 1
    cols = len(hyp) + 1
    table = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        table[i][0] = i
    for j in range(cols):
        table[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            table[i][j] = min(table[i - 1][j] + 1, table[i][j - 1] + 1, table[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    pairs = []
    i, j = len(ref), len(hyp)
    while i > 0 and j > 0:
        if table[i][j] == table[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            if ref[i - 1] != hyp[j - 1]:
                pairs.append((ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif table[i][j] == table[i - 1][j] + 1:
            pairs.append((ref[i - 1], "∅"))
            i -= 1
        else:
            pairs.append(("∅", hyp[j - 1]))
            j -= 1
    return [pair for pair, _ in Counter(reversed(pairs)).most_common(limit)]
