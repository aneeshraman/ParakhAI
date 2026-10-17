from parakh.translate import chunk_text


def test_short_text_is_one_chunk():
    assert chunk_text("नमस्ते दुनिया", 100) == ["नमस्ते दुनिया"]


def test_short_paragraphs_are_packed_together():
    text = "पहला अनुच्छेद।\n\nदूसरा अनुच्छेद।\n\n\nतीसरा।"
    assert chunk_text(text, 1000) == ["पहला अनुच्छेद।\n\nदूसरा अनुच्छेद।\n\nतीसरा।"]


def test_no_chunk_exceeds_limit_and_no_text_is_lost():
    sentence = "यह एक लंबा वाक्य है जो बार बार दोहराया गया है। "
    text = (sentence * 40 + "\n\n") * 3
    chunks = chunk_text(text, 200)
    assert all(len(c) <= 200 for c in chunks)
    assert "".join(text.split()) == "".join("".join(chunks).split())


def test_splits_on_danda_before_whitespace():
    text = "क" * 60 + "। " + "ख" * 60 + "।"
    assert chunk_text(text, 100) == ["क" * 60 + "।", "ख" * 60 + "।"]


def test_unbroken_text_is_hard_split():
    chunks = chunk_text("x" * 250, 100)
    assert [len(c) for c in chunks] == [100, 100, 50]


def test_blank_text_gives_no_chunks():
    assert chunk_text("  \n\n  ", 100) == []
