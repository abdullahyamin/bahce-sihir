"""Lightweight keyword-based query-intent boosting.

Not a general-purpose classifier — targets one diagnosed failure mode: the corpus
now has ~200 structurally near-identical program-description chunks (one AKTS file
per degree level), so plain embedding similarity can't reliably tell "Doktora
Computer Engineering" apart from "Lisans Computer Engineering". When a query names
a specific degree level, we boost that level's file so it survives into the small
rerank candidate pool instead of losing a close race to the other three levels.
"""

import re

from src.retrieval.store import ChunkStore

# Order matters: check the more specific "yüksek lisans" / "önlisans" patterns before
# the bare "lisans" pattern, since "yüksek lisans" contains "lisans" as a substring.
_DEGREE_LEVEL_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"(?i)\bdoktora\b|\bphd\b|\bph\.d\.?\b|\bdoctorate\b|\bdoctoral\b"), "Doktora"),
    (re.compile(r"(?i)y[üu]ksek\s*lisans|\bmaster'?s?\b|\bgraduate\s+program\b"), "Yuksek_Lisans"),
    (re.compile(r"(?i)\b[öo]nlisans\b|\bassociate\s+degree\b"), "Onlisans"),
    (re.compile(r"(?i)\blisans\b|\bundergraduate\b|\bbachelor'?s?\b"), "Lisans"),
]

_LEVEL_TO_FILE = {
    "Onlisans": "BAU_AKTS_Onlisans_Programlari.txt",
    "Lisans": "BAU_AKTS_Lisans_Programlari.txt",
    "Yuksek_Lisans": "BAU_AKTS_Yuksek_Lisans_Programlari.txt",
    "Doktora": "BAU_AKTS_Doktora_Programlari.txt",
}

DEGREE_LEVEL_BOOST = 0.04


def detect_degree_level(query: str) -> str | None:
    for pattern, label in _DEGREE_LEVEL_PATTERNS:
        if pattern.search(query):
            return label
    return None


def apply_category_boost(
    fused: list[tuple[int, float]], query: str, store: ChunkStore
) -> list[tuple[int, float]]:
    level = detect_degree_level(query)
    if level is None:
        return fused

    target_file = _LEVEL_TO_FILE[level]
    boosted = [
        (idx, score + DEGREE_LEVEL_BOOST if store[idx].source_file == target_file else score)
        for idx, score in fused
    ]
    return sorted(boosted, key=lambda item: item[1], reverse=True)
