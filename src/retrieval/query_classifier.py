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

# "Mühendislik fakültesinde staj başvurusu için hangi belge gereklidir?" names the
# faculty but no department, so the answer lives in the faculty-wide Staj Yönergesi.
# But per-department staj pages (BAU_Staj_Endustri_Muhendisligi.txt etc.) are written
# as dense staj how-to guides, so they out-embed the Yönergesi's drier legal-definition
# language even for a query that never names their department. Detect the generic,
# no-department phrasing and boost the faculty-wide Yönergesi specifically.
_STAJ_RE = re.compile(r"(?i)\bstaj")
_ENGINEERING_FACULTY_RE = re.compile(
    r"(?i)m[üu]hendislik(\s+ve\s+do[ğg]a\s+bilimleri)?\s+fak[üu]ltesi|mimarl[ıi]k\s+ve\s+tasar[ıi]m\s+fak[üu]ltesi"
)
_ENGINEERING_DEPARTMENT_RE = re.compile(
    r"(?i)bilgisayar|elektrik[\s-]*elektronik|end[üu]stri|in[şs]aat|i[şs]letme|yaz[ıi]l[ıi]m|mekatronik|makine|biyomedikal"
)
FACULTY_STAJ_FILE = (
    "BAHÇEŞEHİR ÜNİVERSİTESİ MÜHENDİSLİK VE DOĞA BİLİMLERİ FAKÜLTESİ MİMARLIK VE "
    "TASARIM FAKÜLTESİ STAJ YÖNERGESİ.pdf"
)
FACULTY_STAJ_BOOST = 0.15

# "Burs ile destek arasındaki fark nedir?" — a "Tanımlar" article's lettered items
# (see chunking.py's _split_definitions_body) are short, dry one-line legal
# definitions ("Burs: Öğrenim ücretine yapılan desteği,"), so they lose the embedding
# race against longer, topically-richer chunks even when they're the exact answer —
# badly enough that the correct chunk sometimes never lands in ANY retriever's top-10
# for any query variant, so a post-fusion score bump can't help (it isn't in the fused
# list to bump). When a query is plainly asking to define/compare specific term(s), we
# scan every lettered-definition chunk directly for one whose defined term the query
# names, and inject it into the candidate pool outright rather than hoping it surfaces.
_DEFINITION_QUERY_RE = re.compile(
    r"(?i)nedir\s*\??\s*$|ne\s+demektir|arasındaki\s+fark|fark[ıi]\s+nedir|tanım[ıi]\b"
)
_LETTERED_ARTICLE_SUFFIX_RE = re.compile(r"-[a-zçğıöşü]$")
_DEFINED_TERM_RE = re.compile(r"(?m)^[a-zçğıöşü]\)\s*([^:]+):")
DEFINITION_TERM_INJECT_BOOST = 1.0

# "which date is the add and drop week" — the calendar chunk with the actual dates
# ("EKLE-SİL haftası. ... ders ekleme - bırakma haftası") is real and dense-retrievable,
# but calendar files are long and split into many chunks about registration/payment
# deadlines that all share similar vocabulary ("dönem", "tarih", "son gün"), so the one
# specific chunk with the add-drop dates routinely lands just outside the rerank pool
# (observed: rank 7-8 of ~86, one or two spots short of the top 6) rather than being
# absent or badly ranked — a small boost is enough, unlike the definition-injection fix.
_ADD_DROP_QUERY_RE = re.compile(r"(?i)add[\s-]*(?:and[\s-]*)?drop|ekle[\s-]*(?:ve[\s-]*)?(?:sil|b[ıi]rak)")
_ADD_DROP_CHUNK_RE = re.compile(r"(?i)ekle[\s-]*sil")
ADD_DROP_BOOST = 0.05


def is_add_drop_query(query: str) -> bool:
    return bool(_ADD_DROP_QUERY_RE.search(query))


# "Yaz okulunda en fazla kaç ders alabilirim?" — Yaz Okulu Yönergesi's own Madde 9
# ("toplam ders yükü 10 kredidir") answers a related but incomplete version of this:
# it's the *credit* cap, not the *course-count* cap. The actual "en fazla dört ders"
# rule lives in a different document (the general Ön Lisans/Lisans Eğitim-Öğretim
# Yönetmeliği's Madde 18), which loses the embedding race badly once Yaz Okulu
# Yönergesi's own now-properly-chunked articles (see chunking.py's period-header fix)
# dominate the rerank pool with closely related content. Boost the specific chunk
# that states the course-count limit rather than just the credit limit.
_SUMMER_SCHOOL_COURSE_LIMIT_QUERY_RE = re.compile(
    r"(?i)yaz\s*okul.*(ka[çc]|en\s*fazla|en\s*[çc]ok|maks|limit)|summer\s*school.*course|"
    r"how\s*many\s*courses.*summer|max.*courses?.*summer"
)
_SUMMER_SCHOOL_COURSE_LIMIT_CHUNK_RE = re.compile(r"(?i)yaz\s*okul.*(d[öo]rt\s*ders|4\s*ders)")
SUMMER_SCHOOL_COURSE_LIMIT_BOOST = 0.1


def is_summer_school_course_limit_query(query: str) -> bool:
    return bool(_SUMMER_SCHOOL_COURSE_LIMIT_QUERY_RE.search(query))


def is_definition_seeking_query(query: str) -> bool:
    return bool(_DEFINITION_QUERY_RE.search(query))


def find_definition_chunks_matching_query(query: str, store: ChunkStore) -> list[int]:
    matches = []
    for idx in range(len(store)):
        record = store[idx]
        if not (record.article_no and _LETTERED_ARTICLE_SUFFIX_RE.search(record.article_no)):
            continue
        term_match = _DEFINED_TERM_RE.search(record.text)
        if not term_match:
            continue
        term = term_match.group(1).strip()
        if re.search(rf"(?i)\b{re.escape(term)}\b", query):
            matches.append(idx)
    return matches


def detect_degree_level(query: str) -> str | None:
    for pattern, label in _DEGREE_LEVEL_PATTERNS:
        if pattern.search(query):
            return label
    return None


def detect_academic_year(query: str) -> str | None:
    m = _ACADEMIC_YEAR_RE.search(query)
    return m.group(1) if m else None


def is_generic_engineering_faculty_staj_query(query: str) -> bool:
    return bool(
        _STAJ_RE.search(query)
        and _ENGINEERING_FACULTY_RE.search(query)
        and not _ENGINEERING_DEPARTMENT_RE.search(query)
    )


def apply_category_boost(
    fused: list[tuple[int, float]], query: str, store: ChunkStore
) -> list[tuple[int, float]]:
    level = detect_degree_level(query)
    year = detect_academic_year(query)
    faculty_staj = is_generic_engineering_faculty_staj_query(query)
    definition_seeking = is_definition_seeking_query(query)
    add_drop = is_add_drop_query(query)
    summer_course_limit = is_summer_school_course_limit_query(query)
    if (
        level is None
        and year is None
        and not faculty_staj
        and not definition_seeking
        and not add_drop
        and not summer_course_limit
    ):
        return fused

    target_file = _LEVEL_TO_FILE.get(level) if level else None

    boosted = []
    for idx, score in fused:
        adjusted = score
        if target_file is not None and store[idx].source_file == target_file:
            adjusted += DEGREE_LEVEL_BOOST
        if (
            faculty_staj
            and store[idx].source_file == FACULTY_STAJ_FILE
            # Exclude the Yönergesi's own Tanımlar items: "hangi belge gereklidir"
            # doesn't name a specific defined term, so a flat file-wide boost just
            # hands the race to whichever lettered definition happens to embed
            # closest — often the wrong one (e.g. "Staj sicil belgesi" instead of
            # "Zorunlu staj formu" / "Staj başvuru formu"). The real answer lives in
            # a procedural article like Madde 12, not a one-line legal definition.
            and not (store[idx].article_no and _LETTERED_ARTICLE_SUFFIX_RE.search(store[idx].article_no))
        ):
            adjusted += FACULTY_STAJ_BOOST
        if add_drop and _ADD_DROP_CHUNK_RE.search(store[idx].text):
            adjusted += ADD_DROP_BOOST
        if summer_course_limit and _SUMMER_SCHOOL_COURSE_LIMIT_CHUNK_RE.search(store[idx].text):
            adjusted += SUMMER_SCHOOL_COURSE_LIMIT_BOOST
        if year is not None:
            years_in_chunk = set(_ACADEMIC_YEAR_RE.findall(store[idx].text))
            if years_in_chunk == {year}:
                adjusted += ACADEMIC_YEAR_MATCH_BOOST
            elif years_in_chunk:
                adjusted *= ACADEMIC_YEAR_CONFLICT_MULTIPLIER
        boosted.append((idx, adjusted))

    if definition_seeking:
        matched = find_definition_chunks_matching_query(query, store)
        if matched:
            # A matched chunk may already be present in `fused`, but at whatever rank
            # embedding/BM25 gave it — sometimes low, sometimes absent entirely (see
            # comment above: retrieval recall for these is unreliable run-to-run).
            # Promoting it unconditionally rather than only when absent keeps the fix
            # from depending on that recall accident.
            scores = dict(boosted)
            injected_score = max((score for _, score in boosted), default=0.0) + DEFINITION_TERM_INJECT_BOOST
            for idx in matched:
                scores[idx] = max(scores.get(idx, injected_score), injected_score)
            boosted = list(scores.items())

    return sorted(boosted, key=lambda item: item[1], reverse=True)
