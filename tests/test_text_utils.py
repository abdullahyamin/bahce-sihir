from src.text_utils import tokenize_turkish, turkish_lower


def test_turkish_lower_handles_dotted_and_dotless_i():
    assert turkish_lower("İSTANBUL") == "istanbul"
    assert turkish_lower("ISPARTA") == "ısparta"


def test_tokenize_turkish_extracts_words_only():
    tokens = tokenize_turkish("Öğrenci, sınav sonucuna 3 iş günü içinde itiraz edebilir.")
    assert "öğrenci" in tokens
    assert "sınav" in tokens
    assert "3" in tokens
    assert "," not in tokens
    assert "." not in tokens


def test_tokenize_turkish_removes_stopwords_by_default():
    tokens = tokenize_turkish("bu ve şu için bir örnek")
    assert "bu" not in tokens
    assert "ve" not in tokens
    assert "için" not in tokens
    assert "örnek" in tokens


def test_tokenize_turkish_can_keep_stopwords():
    tokens = tokenize_turkish("bu bir örnek", remove_stopwords=False)
    assert "bu" in tokens
    assert "bir" in tokens
