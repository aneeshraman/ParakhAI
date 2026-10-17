"""Evaluate OCR, translation and the translation checks on one answer book.

    python -m eval.run eval/data/mpbse_12_2025_physics_hindi.pdf --lang hi-IN

Every paid call (OCR, translation, back-translation, judge) is cached under eval/cache/, so
re-running only pays for what changed. Writes eval/reports/<document>.md and .json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
from dataclasses import asdict
from pathlib import Path

from sarvamai import SarvamAI

from parakh.checks import BACK_SIMILARITY_WARN, ChunkReport, TranslationChecker, basic_checks
from parakh.config import Settings
from parakh.llm import build_llm
from parakh.translate import SarvamTranslator

from . import metrics
from .cache import cached_ocr, doc_dir
from .inject import MUTATORS

ROOT = Path(__file__).parent
CHECKS = ["numbers", "symbols", "script", "length", "back_translation", "judge"]


class Memo:
    """A JSON file of key -> result, for paid calls that are deterministic enough to reuse."""

    def __init__(self, path: Path):
        self.path = path
        self.data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def get(self, key_parts, compute):
        key = hashlib.sha256(json.dumps(key_parts, ensure_ascii=False).encode()).hexdigest()[:24]
        if key not in self.data:
            self.data[key] = compute()
            self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), encoding="utf-8")
        return self.data[key]


class CachedTranslator:
    def __init__(self, inner: SarvamTranslator, memo: Memo):
        self.inner, self.memo, self.model = inner, memo, inner.model

    def translate_chunks(self, text, source, target="en-IN"):
        return [tuple(p) for p in self.memo.get(["translate", self.model, text, source, target], lambda: self.inner.translate_chunks(text, source, target))]

    def back_translate(self, translation, source, target="en-IN"):
        return self.memo.get(["back", self.model, translation, source, target], lambda: self.inner.back_translate(translation, source, target))


class CachedLLM:
    def __init__(self, inner, memo: Memo):
        self.inner, self.memo, self.name = inner, memo, inner.name

    def generate(self, system, prompt):
        return self.memo.get(["llm", self.name, system, prompt], lambda: self.inner.generate(system, prompt))


def evaluate_ocr(pdf: Path, pages: list[str]) -> list[dict]:
    rows = []
    for ref_path in sorted((ROOT / "references" / pdf.stem).glob("p*.txt")):
        number = int(ref_path.stem[1:])
        reference, hypothesis = ref_path.read_text(encoding="utf-8"), pages[number - 1]
        rows.append(
            {
                "page": number,
                "cer": metrics.cer(reference, hypothesis),
                "hindi_cer": metrics.hindi_cer(reference, hypothesis),
                "hindi_wer": metrics.hindi_wer(reference, hypothesis),
                "word_f1": metrics.word_f1(reference, hypothesis),
                "substitutions": metrics.word_diff(reference, hypothesis),
            }
        )
    return rows


def flags(chunk: ChunkReport) -> dict[str, int]:
    return {check: sum(i.check == check for i in chunk.issues) for check in CHECKS}


def evaluate_injection(checker: TranslationChecker, clean: list[ChunkReport], source: str, per_kind: int, seed: int) -> list[dict]:
    """Mutate up to `per_kind` chunks per error type and record which checks newly fire."""
    rng = random.Random(seed)
    results = []
    for mutate in MUTATORS:
        mutated = []
        for chunk in rng.sample(clean, len(clean)):
            mutation = mutate(chunk.translation, rng)
            if mutation:
                mutated.append((chunk, mutation))
            if len(mutated) == per_kind:
                break
        reports = [ChunkReport(c.index, c.source, m.text, issues=basic_checks(c.index, c.source, m.text)) for c, m in mutated]
        for report in reports:
            checker.add_back_translation(report, source, "en-IN")
        if checker.judge and reports:
            for issue in checker.run_judge(reports, source):
                next(r for r in reports if r.index == issue.chunk).issues.append(issue)
        for (original, mutation), report in zip(mutated, reports):
            before, after = flags(original), flags(report)
            caught = {check: after[check] > before[check] for check in CHECKS}
            results.append(
                {
                    "kind": mutation.kind,
                    "chunk": original.index,
                    "change": mutation.description,
                    "caught": caught,
                    "caught_any": any(caught.values()),
                    "similarity_before": original.back_similarity,
                    "similarity_after": report.back_similarity,
                }
            )
    return results


def write_report(path: Path, pdf: Path, ocr, ocr_rows, clean, injection, settings, judge_name) -> None:
    lines = [f"# Evaluation: {pdf.name}", ""]
    lines += [
        f"- Pages: {len(ocr.pages)} · OCR characters: {len(ocr.text)} · translation chunks: {len(clean)}",
        f"- Translation model: `{settings.translate_model}` · judge: `{judge_name or 'off'}`",
        "",
        "## 1. OCR accuracy (Sarvam Vision vs reference transcriptions)",
        "",
        "Lower is better for CER/WER; higher is better for word F1. *Hindi* metrics ignore maths and Latin symbols.",
        "",
        "| Page | CER | Hindi CER | Hindi WER | Hindi word F1 |",
        "|---|---|---|---|---|",
    ]
    for r in ocr_rows:
        lines.append(f"| {r['page']} | {r['cer']:.1%} | {r['hindi_cer']:.1%} | {r['hindi_wer']:.1%} | {r['word_f1']:.1%} |")
    if ocr_rows:
        mean = lambda key: statistics.mean(r[key] for r in ocr_rows)  # noqa: E731
        lines.append(f"| **mean** | **{mean('cer'):.1%}** | **{mean('hindi_cer'):.1%}** | **{mean('hindi_wer'):.1%}** | **{mean('word_f1'):.1%}** |")
        lines += ["", "Most frequent word errors (reference → OCR):", ""]
        for r in ocr_rows:
            if r["substitutions"]:
                lines.append(f"- p{r['page']}: " + ", ".join(f"{a} → {b}" for a, b in r["substitutions"][:8]))

    lines += ["", "## 2. Checks on the real translation", ""]
    similarities = [c.back_similarity for c in clean if c.back_similarity is not None]
    if similarities:
        lines.append(
            f"Round-trip similarity (chrF, Hindi → English → Hindi): median {statistics.median(similarities):.0%}, "
            f"min {min(similarities):.0%}, max {max(similarities):.0%}; warning threshold {BACK_SIMILARITY_WARN:.0%}."
        )
    lines += ["", "| Chunk | Round-trip | Flags |", "|---|---|---|"]
    for c in clean:
        found = "<br>".join(f"**{i.severity}** {i.check}: {i.message}" for i in c.issues) or "—"
        sim = f"{c.back_similarity:.0%}" if c.back_similarity is not None else "—"
        lines.append(f"| {c.index} | {sim} | {found.replace('|', '/')} |")
    lines += ["", "These flags are on the unmodified translation: some are real translation errors, some false alarms. Read them against the chunks in the JSON report.", ""]

    lines += ["## 3. Planted errors: which checks catch them", ""]
    kinds = sorted({r["kind"] for r in injection})
    lines += ["| Error type | n | " + " | ".join(CHECKS) + " | any check |", "|---|---|" + "---|" * (len(CHECKS) + 1)]
    for kind in kinds:
        rows = [r for r in injection if r["kind"] == kind]
        cells = [f"{sum(r['caught'][c] for r in rows)}/{len(rows)}" for c in CHECKS]
        lines.append(f"| {kind} | {len(rows)} | " + " | ".join(cells) + f" | **{sum(r['caught_any'] for r in rows)}/{len(rows)}** |")
    lines += ["", "Each planted error:", ""]
    for r in injection:
        by = ", ".join(c for c in CHECKS if r["caught"][c]) or "**missed**"
        sim = ""
        if r["similarity_before"] is not None and r["similarity_after"] is not None:
            sim = f" (round-trip {r['similarity_before']:.0%} → {r['similarity_after']:.0%})"
        lines.append(f"- {r['kind']}, chunk {r['chunk']}: {r['change']} — caught by {by}{sim}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval.run", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--lang", default="hi-IN")
    parser.add_argument("--content-type", default="handwritten")
    parser.add_argument("--per-kind", type=int, default=4, help="planted errors per error type (each costs a back-translation)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-judge", action="store_true")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    settings = Settings()
    cache = doc_dir(args.pdf)
    translator = CachedTranslator(SarvamTranslator(SarvamAI(api_subscription_key=settings.require_sarvam_key()), settings), Memo(cache / "translate.json"))
    judge = None if args.no_judge else CachedLLM(build_llm(settings), Memo(cache / "judge.json"))
    checker = TranslationChecker(translator, judge)

    print("1/4 OCR (cached after the first run)…")
    ocr = cached_ocr(args.pdf, language=args.lang, content_type=args.content_type)
    ocr_rows = evaluate_ocr(args.pdf, ocr.pages)

    print("2/4 Translating…")
    pairs = translator.translate_chunks(ocr.text, args.lang)
    print(f"3/4 Checking {len(pairs)} chunks (back-translation{', judge' if judge else ''})…")
    report = checker.check(pairs, args.lang)

    print(f"4/4 Planting up to {args.per_kind} errors of each type…")
    injection = evaluate_injection(checker, report.chunks, args.lang, args.per_kind, args.seed)

    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    md, js = out_dir / f"{args.pdf.stem}.md", out_dir / f"{args.pdf.stem}.json"
    write_report(md, args.pdf, ocr, ocr_rows, report.chunks, injection, settings, judge.name if judge else None)
    js.write_text(
        json.dumps({"ocr": ocr_rows, "translation": report.to_dict(), "injection": injection, "ocr_pages": ocr.pages}, ensure_ascii=False, indent=2, default=asdict),
        encoding="utf-8",
    )
    print(f"\nWrote {md} and {js}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
