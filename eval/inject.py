"""Plant known errors in English translations, to measure which checks catch which errors."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

# Physics terms whose swap flips the meaning but leaves a fluent sentence.
TERM_PAIRS = [
    ("interference", "diffraction"),
    ("maxima", "minima"),
    ("maximum", "minimum"),
    ("bright", "dark"),
    ("increases", "decreases"),
    ("increase", "decrease"),
    ("series", "parallel"),
    ("positive", "negative"),
    ("internal", "external"),
    ("inner", "outer"),
    ("polarized", "unpolarized"),
    ("polarised", "unpolarised"),
    ("electric", "magnetic"),
    ("current", "voltage"),
    ("convex", "concave"),
    ("more", "less"),
    ("alternating", "direct"),
    ("parallel", "perpendicular"),
]
_AUXILIARY = re.compile(r"\b(is|are|was|were|does|do|will|can|has|have)\b(?! not\b)")
_NOT = re.compile(r"\b(is|are|was|were|does|do|will|can|has|have) not\b")


@dataclass
class Mutation:
    kind: str
    text: str
    description: str


def _is_list_marker(text: str, m: re.Match) -> bool:
    line_start = text.rfind("\n", 0, m.start()) + 1
    return not text[line_start : m.start()].strip() and text[m.end() : m.end() + 1] in (".", ")")


def change_number(text: str, rng: random.Random) -> Mutation | None:
    candidates = [m for m in re.finditer(r"\d+(?:\.\d+)?", text) if not _is_list_marker(text, m)]
    if not candidates:
        return None
    m = rng.choice(candidates)
    old = m.group()
    new = f"{float(old) * 10:g}" if "." in old else str(int(old) + 1 if int(old) < 10 else int(old) * 2)
    return Mutation("number", text[: m.start()] + new + text[m.end() :], f"{old} → {new}")


def flip_negation(text: str, rng: random.Random) -> Mutation | None:
    removable = list(_NOT.finditer(text))
    if removable and rng.random() < 0.5:
        m = rng.choice(removable)
        return Mutation("negation", text[: m.start()] + m.group(1) + text[m.end() :], f"removed “not” after “{m.group(1)}”")
    addable = list(_AUXILIARY.finditer(text))
    if not addable:
        return None
    m = rng.choice(addable)
    return Mutation("negation", text[: m.end()] + " not" + text[m.end() :], f"added “not” after “{m.group()}”")


def swap_term(text: str, rng: random.Random) -> Mutation | None:
    options = []
    for a, b in TERM_PAIRS:
        for old, new in ((a, b), (b, a)):
            options += [(m, new) for m in re.finditer(rf"\b{old}\b", text, re.I)]
    if not options:
        return None
    m, new = rng.choice(options)
    if m.group()[0].isupper():
        new = new.capitalize()
    return Mutation("term_swap", text[: m.start()] + new + text[m.end() :], f"{m.group()} → {new}")


def drop_sentence(text: str, rng: random.Random) -> Mutation | None:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    candidates = [i for i, s in enumerate(sentences) if len(s) >= 40 and 0 < i < len(sentences) - 1]
    if len(sentences) < 3 or not candidates:
        return None
    i = rng.choice(candidates)
    dropped = sentences.pop(i)
    return Mutation("drop_sentence", " ".join(sentences), f"dropped “{dropped[:60]}…”")


MUTATORS = [change_number, flip_negation, swap_term, drop_sentence]
