from src.retrieval.query_classifier import (
    ACADEMIC_YEAR_CONFLICT_MULTIPLIER,
    ACADEMIC_YEAR_MATCH_BOOST,
    DEGREE_LEVEL_BOOST,
    apply_category_boost,
    detect_academic_year,
    detect_degree_level,
)
from src.retrieval.store import ChunkRecord


def _record(source_file: str, text: str = "x") -> ChunkRecord:
    return ChunkRecord(text=text, source_file=source_file, category="web_sss", section=None, article_no=None)


class _FakeStore:
    def __init__(self, records: dict[int, ChunkRecord]):
        self._records = records

    def __getitem__(self, idx: int) -> ChunkRecord:
        return self._records[idx]


def test_detect_degree_level_doktora_turkish_and_english():
    assert detect_degree_level("Doktora programının hedefi nedir?") == "Doktora"
    assert detect_degree_level("What is the goal of the PhD program?") == "Doktora"
    assert detect_degree_level("Ph.D. requirements") == "Doktora"


def test_detect_degree_level_yuksek_lisans_not_misdetected_as_lisans():
    # "yüksek lisans" contains "lisans" as a substring — this is the exact bug this
    # module exists to avoid, so it gets its own explicit regression test.
    assert detect_degree_level("Yüksek Lisans programı ne kadar sürer?") == "Yuksek_Lisans"
    assert detect_degree_level("master's program duration") == "Yuksek_Lisans"


def test_detect_degree_level_bare_lisans():
    assert detect_degree_level("Lisans programına nasıl kayıt olurum?") == "Lisans"
    assert detect_degree_level("undergraduate program requirements") == "Lisans"


def test_detect_degree_level_onlisans():
    assert detect_degree_level("Önlisans programları nelerdir?") == "Onlisans"
    assert detect_degree_level("associate degree options") == "Onlisans"


def test_detect_degree_level_none_when_no_keyword():
    assert detect_degree_level("Moleküler Biyoloji ve Genetik programı nedir?") is None
    assert detect_degree_level("How do I borrow a library book?") is None


def test_apply_category_boost_promotes_matching_file():
    store = _FakeStore({
        0: _record("BAU_AKTS_Lisans_Programlari.txt"),
        1: _record("BAU_AKTS_Doktora_Programlari.txt"),
        2: _record("BAU_AKTS_Yuksek_Lisans_Programlari.txt"),
    })
    fused = [(0, 0.05), (1, 0.03), (2, 0.02)]

    boosted = apply_category_boost(fused, "What is the Doktora program goal?", store)

    assert boosted[0][0] == 1
    assert boosted[0][1] == 0.03 + DEGREE_LEVEL_BOOST


def test_apply_category_boost_no_op_when_no_level_detected():
    store = _FakeStore({0: _record("BAU_AKTS_Lisans_Programlari.txt")})
    fused = [(0, 0.05)]

    boosted = apply_category_boost(fused, "How do I borrow a library book?", store)

    assert boosted == fused


def test_detect_academic_year():
    assert detect_academic_year("2026-2027 güz yarıyılı ne zaman başlar?") == "2026-2027"
    assert detect_academic_year("Yaz okulu ne zaman başlar?") is None


def test_apply_category_boost_promotes_unambiguous_year_match():
    # Regression test: BAU_Akademik_Takvim files get their own title (naming the
    # headline year) prepended to every chunk, so a plain "contains the target year"
    # check can't discriminate — every chunk contains it. Only a chunk mentioning
    # *exclusively* the target year should get the positive boost.
    store = _FakeStore({
        0: _record("cal.txt", text="2026-2027 akademik yılı ... 17 Temmuz 2026 ..."),
        1: _record("cal.txt", text="2026-2027 akademik yılı ... 2027-2028 akademik yılı ..."),
    })
    fused = [(0, 0.05), (1, 0.05)]

    boosted = apply_category_boost(fused, "2026-2027 güz yarıyılı ne zaman başlar?", store)
    boosted_scores = dict(boosted)

    assert boosted_scores[0] == 0.05 + ACADEMIC_YEAR_MATCH_BOOST
    assert boosted_scores[1] == 0.05 * ACADEMIC_YEAR_CONFLICT_MULTIPLIER


def test_apply_category_boost_penalizes_conflicting_year_chunk():
    store = _FakeStore({
        0: _record("cal.txt", text="2027-2028 akademik yılı ... hiç 2026-2027 yok"),
    })
    fused = [(0, 0.2)]

    boosted = apply_category_boost(fused, "2026-2027 güz yarıyılı ne zaman başlar?", store)

    assert boosted[0][1] == 0.2 * ACADEMIC_YEAR_CONFLICT_MULTIPLIER


def test_apply_category_boost_neutral_when_chunk_has_no_year():
    store = _FakeStore({0: _record("reg.pdf", text="Madde 5 - genel hüküm.")})
    fused = [(0, 0.1)]

    boosted = apply_category_boost(fused, "2026-2027 güz yarıyılı ne zaman başlar?", store)

    assert boosted[0][1] == 0.1
