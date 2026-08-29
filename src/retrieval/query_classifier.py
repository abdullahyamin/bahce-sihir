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

# Academic-calendar documents (e.g. BAU_Akademik_Takvim_*.txt) span a full multi-year
# document, and every one of their fallback-split chunks gets the document's own
# title line prepended (see chunking.py's _extract_title) — which itself names the
# document's headline year, e.g. "2026-2027 Akademik Yılı ...". That means EVERY
# chunk superficially contains the headline year regardless of which specific dated
# entries it actually holds, so a plain "boost chunks containing the query's year"
# can't discriminate between them. What *does* discriminate: a handful of chunks
# straddle two academic years in their body text (e.g. a July-2027 registration
# deadline for the *next* year, 2027-2028, sitting near the end of a 2026-2027
# calendar). When the query names one specific year, we penalize chunks that also
# contain a *different* explicit year — demoting the ambiguous straddling chunks
# relative to the clean single-year ones, so the generator is less likely to be
# handed a chunk containing the wrong year's date for the sentence it needs.
_ACADEMIC_YEAR_RE = re.compile(r"\b(20\d{2}-20\d{2})\b")
# A flat penalty can't compete with chunks whose raw fused score is an order of
# magnitude above everything else (observed: a mixed-year chunk scored ~0.16 vs.
# ~0.015 for the correct single-year chunk) — the discount has to scale with the
# chunk's own score, not subtract a fixed amount.
ACADEMIC_YEAR_CONFLICT_MULTIPLIER = 0.1
ACADEMIC_YEAR_MATCH_BOOST = 0.04


def detect_degree_level(query: str) -> str | None:
    for pattern, label in _DEGREE_LEVEL_PATTERNS:
        if pattern.search(query):
            return label
    return None


def detect_academic_year(query: str) -> str | None:
    m = _ACADEMIC_YEAR_RE.search(query)
    return m.group(1) if m else None


def apply_category_boost(
    fused: list[tuple[int, float]], query: str, store: ChunkStore
) -> list[tuple[int, float]]:
    level = detect_degree_level(query)
    year = detect_academic_year(query)
    if level is None and year is None:
        return fused

    target_file = _LEVEL_TO_FILE.get(level) if level else None

    boosted = []
    for idx, score in fused:
        adjusted = score
        if target_file is not None and store[idx].source_file == target_file:
            adjusted += DEGREE_LEVEL_BOOST
        if year is not None:
            years_in_chunk = set(_ACADEMIC_YEAR_RE.findall(store[idx].text))
            if years_in_chunk == {year}:
                adjusted += ACADEMIC_YEAR_MATCH_BOOST
            elif years_in_chunk:
                adjusted *= ACADEMIC_YEAR_CONFLICT_MULTIPLIER
        boosted.append((idx, adjusted))

    return sorted(boosted, key=lambda item: item[1], reverse=True)
