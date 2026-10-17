"""Command line: python -m parakh path/to/letter.jpg --lang hi-IN"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sarvamai.core.api_error import ApiError

from .languages import CONTENT_TYPES, LANGUAGES
from .llm import LLMError
from .ocr import OCRError
from .pipeline import DEFAULT_TASK, Pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="parakh", description=__doc__)
    parser.add_argument("file", type=Path, help="PDF (split into 10-page jobs), PNG, JPG, or ZIP of page images")
    parser.add_argument("--lang", choices=sorted(LANGUAGES), help="document language; detected if omitted")
    parser.add_argument("--content-type", choices=CONTENT_TYPES, default="handwritten")
    parser.add_argument("--task", default=DEFAULT_TASK, help="what the LLM should do with the text")
    parser.add_argument("--no-llm", action="store_true", help="stop after OCR and translation")
    parser.add_argument("--no-back-translation", action="store_true", help="skip the round-trip translation check")
    parser.add_argument("--no-judge", action="store_true", help="skip the LLM review of the translation")
    parser.add_argument("--json", type=Path, metavar="OUT", help="also write the full result as JSON")
    args = parser.parse_args(argv)

    # Windows consoles default to a legacy code page that can't print Indic scripts.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    try:
        pipeline = Pipeline.from_settings(
            with_llm=not args.no_llm,
            back_translate=False if args.no_back_translation else None,
            judge=False if args.no_judge else None,
        )
        result = pipeline.run(
            args.file.read_bytes(),
            args.file.name,
            language=args.lang,
            content_type=args.content_type,
            task=args.task,
            use_llm=not args.no_llm,
        )
    except ApiError as e:
        print(f"error: Sarvam API error {e.status_code}: {e.body}", file=sys.stderr)
        return 1
    except (OCRError, LLMError, RuntimeError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    _section(f"Original text ({result.source_language_name})", result.native_text)
    if result.english_text is not None and result.english_text != result.native_text:
        _section("English translation", result.english_text)
    if result.translation_report:
        report = result.translation_report
        lines = [report["summary"]]
        for chunk in report["chunks"]:
            similarity = chunk["back_similarity"]
            header = f"Chunk {chunk['index']}" + (f" (round-trip similarity {similarity:.0%})" if similarity is not None else "")
            lines += [f"{header}: [{i['severity']}] {i['check']}: {i['message']}" for i in chunk["issues"]]
        _section("Translation checks", "\n".join(lines))
    if result.llm_output is not None:
        _section(f"LLM output ({result.llm_name})", result.llm_output)
    for warning in result.warnings:
        print(f"warning: {warning}", file=sys.stderr)

    if args.json:
        args.json.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved {args.json}")
    return 0


def _section(title: str, body: str) -> None:
    print(f"\n=== {title} ===\n{body or '(empty)'}")
