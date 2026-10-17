from parakh.checks import (
    TranslationChecker,
    basic_checks,
    check_length,
    check_numbers,
    check_script,
    check_symbols,
    chrf,
    numbers,
    parse_judge,
)


def test_numbers_normalise_indic_digits_and_superscripts():
    assert numbers("ऊर्जा १३.६ eV, आवृत्ति 4×10¹⁴ Hz") == numbers("energy 13.6 eV, frequency 4×1014 Hz")


def test_missing_number_is_an_error_and_extra_is_a_warning():
    issues = check_numbers(1, "धारा 30 A है, r = 0.4", "the current is 3 A, r = 0.4")
    assert [(i.severity, i.message.split(":")[1].strip()) for i in issues] == [("error", "30"), ("warning", "3")]


def test_numbers_that_survive_are_fine():
    assert check_numbers(1, "E = 12 वोल्ट, r = 0.4 ओम", "E = 12 volts, r = 0.4 ohms") == []


def test_dropped_formula_symbol_is_flagged():
    issues = check_symbols(1, "विवर्तन में λ/a कोण पर", "in diffraction at angle 1/a")
    assert issues and "λ" in issues[0].message


def test_untranslated_hindi_is_flagged():
    assert check_script(1, "The intensity of उच्चिष्ठ is maximum", "en-IN")
    assert check_script(1, "The intensity of the maxima is maximum", "en-IN") == []


def test_length_ratio():
    source = "व्यतिकरण में केंद्रीय दीप्त फ्रिंज की तीव्रता व अन्य दीप्त फ्रिंजों की तीव्रता एकसमान होती है।"
    assert check_length(1, source, "Same.")[0].check == "length"
    assert check_length(1, source, "In interference the central bright fringe has the same intensity as the others.") == []
    assert check_length(1, source, "   ")[0].severity == "error"


def test_chrf():
    assert chrf("नमस्ते दुनिया", "नमस्ते दुनिया") == 1.0
    assert chrf("तीव्रता में परिवर्तन होता है", "तीव्रता में परिवर्तन नहीं होता है") > 0.7
    assert chrf("तीव्रता में परिवर्तन होता है", "the quick brown fox") == 0.0


def test_parse_judge_keeps_only_known_chunks_and_tolerates_prose():
    reply = 'Here you go:\n{"issues": [{"chunk": 2, "severity": "error", "problem": "maxima became minima", "original": "उच्चिष्ठ", "translation": "minima"}, {"chunk": 9, "severity": "error", "problem": "x"}]}'
    issues = parse_judge(reply, {1, 2})
    assert len(issues) == 1 and issues[0].chunk == 2 and issues[0].severity == "error"
    assert "minima" in issues[0].message


def test_unparseable_judge_reply_becomes_a_warning():
    issues = parse_judge("I could not do this", {3, 4})
    assert issues[0].severity == "warning" and issues[0].chunk == 3


class FakeTranslator:
    def back_translate(self, translation, source, target="en-IN"):
        return {"good": "ऊर्जा 13.6 eV होगी", "bad": "कुछ और ही बात"}[translation.split()[0]]


class FakeJudge:
    name = "fake"

    def __init__(self):
        self.calls = 0

    def generate(self, system, prompt):
        self.calls += 1
        return '{"issues": [{"chunk": 2, "severity": "warning", "problem": "unclear"}]}'


def test_checker_combines_all_checks():
    judge = FakeJudge()
    pairs = [("ऊर्जा 13.6 eV होगी", "good energy will be 13.6 eV"), ("ऊर्जा 13.6 eV होगी", "bad energy will be 3.6 eV")]
    report = TranslationChecker(FakeTranslator(), judge).check(pairs, "hi-IN")

    first, second = report.chunks
    assert first.back_similarity == 1.0 and first.issues == []
    assert {i.check for i in second.issues} == {"numbers", "back_translation", "judge"}
    assert judge.calls == 1
    assert report.summary() == "1/2 translated chunks flagged (1 errors, 3 warnings)."


def test_basic_checks_pass_a_faithful_translation():
    source = "अतः प्रथम आयनन ऊर्जा 13.6 eV होगी।"
    assert basic_checks(1, source, "Hence the first ionisation energy will be 13.6 eV.") == []
