from parakh.checks import ChunkReport, Issue, TranslationReport
from parakh.ocr import OCRResult
from parakh.pipeline import Pipeline, build_prompt


class FakeOCR:
    def __init__(self, text):
        self.text = text
        self.calls = []

    def digitise(self, data, filename, *, language=None, content_type="handwritten"):
        self.calls.append({"language": language, "content_type": content_type})
        return OCRResult(text=self.text, pages=[self.text], job_ids=["job-1"], status="completed")


class FakeTranslator:
    model = "sarvam-translate:v1"

    def __init__(self, detected="hi-IN"):
        self.detected = detected
        self.translated = []

    def detect_language(self, text):
        return self.detected

    def supports(self, language):
        return True

    def translate_chunks(self, text, source, target="en-IN"):
        self.translated.append(source)
        return [(text, f"EN({text})")]


class FakeLLM:
    name = "fake:gemma"

    def __init__(self):
        self.prompts = []

    def generate(self, system, prompt):
        self.prompts.append(prompt)
        return "summary"


def test_full_run_detects_language_translates_and_calls_llm():
    llm = FakeLLM()
    translator = FakeTranslator(detected="ta-IN")
    result = Pipeline(FakeOCR("வணக்கம்"), translator, llm).run(b"img", "note.jpg")

    assert result.source_language == "ta-IN"
    assert result.source_language_name == "Tamil"
    assert result.english_text == "EN(வணக்கம்)"
    assert result.llm_output == "summary"
    assert result.llm_name == "fake:gemma"
    assert translator.translated == ["ta-IN"]
    assert "<original_text>\nவணக்கம்\n</original_text>" in llm.prompts[0]
    assert "<english_translation>\nEN(வணக்கம்)\n</english_translation>" in llm.prompts[0]


def test_explicit_language_is_passed_to_ocr_and_skips_detection():
    ocr = FakeOCR("नमस्ते")
    translator = FakeTranslator(detected=None)
    result = Pipeline(ocr, translator, None).run(b"img", "a.png", language="hi-IN", content_type="mixed")

    assert ocr.calls == [{"language": "hi-IN", "content_type": "mixed"}]
    assert result.source_language == "hi-IN"
    assert result.llm_output is None


def test_english_documents_are_not_translated():
    translator = FakeTranslator(detected="en-IN")
    result = Pipeline(FakeOCR("Hello"), translator, None).run(b"img", "a.png")

    assert result.english_text == "Hello"
    assert translator.translated == []


def test_undetected_language_still_reaches_llm_with_warning():
    llm = FakeLLM()
    result = Pipeline(FakeOCR("ꯃꯤꯇꯩ"), FakeTranslator(detected=None), llm).run(b"img", "a.png")

    assert result.english_text is None
    assert result.llm_output == "summary"
    assert any("identify" in w for w in result.warnings)
    assert "<english_translation>" not in llm.prompts[0]


def test_empty_ocr_skips_llm():
    llm = FakeLLM()
    result = Pipeline(FakeOCR(""), FakeTranslator(), llm).run(b"img", "a.png", use_llm=True)

    assert result.llm_output is None
    assert llm.prompts == []


def test_check_findings_reach_the_llm_prompt():
    class FlaggingChecker:
        def check(self, pairs, source):
            return TranslationReport([ChunkReport(1, pairs[0][0], pairs[0][1], issues=[Issue(1, "numbers", "error", "13.6 missing")])])

    llm = FakeLLM()
    result = Pipeline(FakeOCR("ऊर्जा 13.6 eV"), FakeTranslator(), llm, FlaggingChecker()).run(b"img", "a.png")

    assert "<translation_checks>" in llm.prompts[0]
    assert "[error] numbers" in llm.prompts[0] and "13.6 missing" in llm.prompts[0]
    assert result.translation_report["summary"].startswith("1/1 translated chunks flagged")
    assert any("Translation checks" in w for w in result.warnings)


def test_prompt_contains_task():
    prompt = build_prompt("text", None, "hi-IN", "  List the names.  ")
    assert prompt.endswith("Task: List the names.")
    assert 'language="Hindi"' in prompt
