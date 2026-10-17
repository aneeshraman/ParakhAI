"""Document -> native text -> English (checked) -> LLM, wired together."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from sarvamai import SarvamAI

from .checks import TranslationChecker, TranslationReport
from .config import Settings
from .languages import language_name
from .llm import LLM, build_llm
from .ocr import SarvamOCR
from .translate import SarvamTranslator

DEFAULT_TASK = (
    "Give a faithful, cleaned-up English rendering of the document. Then write a short summary "
    "and list the key facts (names, dates, amounts, places, identifiers). Finally, list any "
    "passages that look illegible, garbled or uncertain."
)

SYSTEM_PROMPT = """\
You work with documents written in Indian languages, many of them handwritten.

You receive the document twice: the original-language text produced by OCR, and a machine \
translation of it into English. Both can be wrong. Handwriting OCR misreads characters, \
merges or splits words and garbles names and numbers; translation can then turn a misread \
word into a confident but wrong English phrase. When automatic checks flagged parts of the \
translation, you also get their findings; treat those passages with extra care.

Read the original-language text yourself and use it to check the translation. Where they \
disagree, prefer the reading that fits the original script and the surrounding context, and \
say so. Keep names, numbers and dates exactly as written in the original unless they are \
clearly an OCR error. Do not fill gaps with invented content: say what is missing or unclear."""


@dataclass
class PipelineResult:
    filename: str
    source_language: str | None
    source_language_name: str
    native_text: str
    english_text: str | None
    llm_output: str | None
    llm_name: str | None
    ocr_job_ids: list[str]
    translation_report: dict | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class Pipeline:
    def __init__(
        self,
        ocr: SarvamOCR,
        translator: SarvamTranslator,
        llm: LLM | None,
        checker: TranslationChecker | None = None,
    ):
        self.ocr = ocr
        self.translator = translator
        self.llm = llm
        self.checker = checker

    @classmethod
    def from_settings(
        cls,
        settings: Settings | None = None,
        *,
        with_llm: bool = True,
        back_translate: bool | None = None,
        judge: bool | None = None,
    ) -> Pipeline:
        settings = settings or Settings()
        client = SarvamAI(api_subscription_key=settings.require_sarvam_key())
        translator = SarvamTranslator(client, settings)
        llm = build_llm(settings) if with_llm else None
        use_judge = settings.check_judge if judge is None else judge
        checker = TranslationChecker(
            translator,
            judge=(llm or build_llm(settings)) if use_judge else None,
            back_translate=settings.check_back_translation if back_translate is None else back_translate,
        )
        return cls(SarvamOCR(client, settings), translator, llm, checker)

    def run(
        self,
        data: bytes,
        filename: str,
        *,
        language: str | None = None,
        content_type: str = "handwritten",
        task: str = DEFAULT_TASK,
        use_llm: bool = True,
    ) -> PipelineResult:
        ocr = self.ocr.digitise(data, filename, language=language, content_type=content_type)
        warnings = list(ocr.warnings)

        source = language or self.translator.detect_language(ocr.text)
        english: str | None = None
        report: TranslationReport | None = None
        if not ocr.text:
            pass
        elif source is None:
            warnings.append("Could not identify the document language; skipped translation. Pass the language explicitly.")
        elif source == "en-IN":
            english = ocr.text
        elif not self.translator.supports(source):
            warnings.append(f"{language_name(source)} is not supported by {self.translator.model}; skipped translation.")
        else:
            pairs = self.translator.translate_chunks(ocr.text, source)
            english = "\n\n".join(translation for _, translation in pairs)
            if self.checker:
                report = self.checker.check(pairs, source)
                if report.issues:
                    warnings.append(f"Translation checks: {report.summary()}")

        llm_output = None
        if use_llm and self.llm and ocr.text:
            llm_output = self.llm.generate(SYSTEM_PROMPT, build_prompt(ocr.text, english, source, task, report))

        return PipelineResult(
            filename=filename,
            source_language=source,
            source_language_name=language_name(source),
            native_text=ocr.text,
            english_text=english,
            llm_output=llm_output,
            llm_name=self.llm.name if llm_output is not None else None,
            ocr_job_ids=ocr.job_ids,
            translation_report=report.to_dict() if report else None,
            warnings=warnings,
        )


def build_prompt(
    native_text: str,
    english_text: str | None,
    language: str | None,
    task: str,
    report: TranslationReport | None = None,
) -> str:
    parts = [f'<document language="{language_name(language)}">', "<original_text>", native_text, "</original_text>"]
    if english_text is not None and english_text != native_text:
        parts += ["<english_translation>", english_text, "</english_translation>"]
    parts.append("</document>")
    if report and report.issues:
        parts += ["", "<translation_checks>"]
        for issue in report.issues:
            excerpt = report.chunks[issue.chunk - 1].translation[:80].replace("\n", " ")
            parts.append(f"- [{issue.severity}] {issue.check}, in the passage starting “{excerpt}…”: {issue.message}")
        parts.append("</translation_checks>")
    parts += ["", f"Task: {task.strip()}"]
    return "\n".join(parts)
