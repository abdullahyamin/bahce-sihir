import re

_WORD_RE = re.compile(r"[a-zçğıöşü0-9]+", re.IGNORECASE)
_TURKISH_LOWER_MAP = str.maketrans({"I": "ı", "İ": "i"})

TURKISH_STOPWORDS = {
    "ve", "veya", "ile", "bu", "bir", "da", "de", "ki", "ise", "için",
    "gibi", "ama", "fakat", "ancak", "çünkü", "her", "hem", "ya", "mi",
    "mı", "mu", "mü", "ne", "o", "şu", "en", "daha", "çok", "az", "ise",
}


def turkish_lower(text: str) -> str:
    return text.translate(_TURKISH_LOWER_MAP).lower()


def tokenize_turkish(text: str, remove_stopwords: bool = True) -> list[str]:
    tokens = _WORD_RE.findall(turkish_lower(text))
    if remove_stopwords:
        tokens = [t for t in tokens if t not in TURKISH_STOPWORDS]
    return tokens
