from src.retrieval.query_classifier import (
    ACADEMIC_YEAR_CONFLICT_MULTIPLIER,
    ACADEMIC_YEAR_MATCH_BOOST,
    ADD_DROP_BOOST,
    DEGREE_LEVEL_BOOST,
    FACULTY_STAJ_BOOST,
    FACULTY_STAJ_FILE,
    apply_category_boost,
    detect_academic_year,
    detect_degree_level,
    SUMMER_SCHOOL_COURSE_LIMIT_BOOST,
    find_definition_chunks_matching_query,
    is_add_drop_query,
    is_definition_seeking_query,
    is_generic_engineering_faculty_staj_query,
    is_summer_school_course_limit_query,
)
from src.retrieval.store import ChunkRecord


def _record(source_file: str, text: str = "x", article_no: str | None = None) -> ChunkRecord:
    return ChunkRecord(text=text, source_file=source_file, category="web_sss", section=None, article_no=article_no)


class _FakeStore:
    def __init__(self, records: dict[int, ChunkRecord]):
        self._records = records

    def __getitem__(self, idx: int) -> ChunkRecord:
        return self._records[idx]

    def __len__(self) -> int:
        return max(self._records) + 1 if self._records else 0


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


def test_is_generic_engineering_faculty_staj_query_true_without_department():
    assert is_generic_engineering_faculty_staj_query(
        "Mühendislik fakültesinde staj başvurusu için hangi belge gereklidir?"
    )


def test_is_generic_engineering_faculty_staj_query_false_with_department_named():
    # A department-specific staj query should keep favoring that department's own
    # staj page, not the faculty-wide Yönergesi.
    assert not is_generic_engineering_faculty_staj_query(
        "Endüstri Mühendisliği bölümünde staj başvurusu için hangi belge gereklidir?"
    )


def test_is_generic_engineering_faculty_staj_query_false_without_staj_keyword():
    assert not is_generic_engineering_faculty_staj_query(
        "Mühendislik fakültesinde kaç bölüm vardır?"
    )


def test_apply_category_boost_promotes_faculty_staj_yonergesi():
    store = _FakeStore({
        0: _record("BAU_Staj_Endustri_Muhendisligi.txt"),
        1: _record(FACULTY_STAJ_FILE),
    })
    fused = [(0, 0.05), (1, 0.03)]

    boosted = apply_category_boost(
        fused, "Mühendislik fakültesinde staj başvurusu için hangi belge gereklidir?", store
    )

    assert boosted[0][0] == 1
    assert boosted[0][1] == 0.03 + FACULTY_STAJ_BOOST


def test_apply_category_boost_skips_lettered_definition_items_in_faculty_staj_file():
    # Regression test: the Yönergesi's own Tanımlar article gets split into one chunk
    # per lettered term (see chunking.py's _split_definitions_body). A generic "which
    # documents are required" query doesn't name a specific term, so boosting every
    # chunk in the file equally just hands the race to whichever definition item
    # happens to embed closest to the query — the real answer is in a procedural
    # article like Madde 12, so lettered-definition chunks must not get this boost.
    store = _FakeStore({
        0: _record(FACULTY_STAJ_FILE, article_no="4-ğ"),  # Tanımlar item, e.g. "Staj sicil belgesi"
        1: _record(FACULTY_STAJ_FILE, article_no="12"),  # actual procedural article
    })
    fused = [(0, 0.05), (1, 0.03)]

    boosted = apply_category_boost(
        fused, "Mühendislik fakültesinde staj başvurusu için hangi belge gereklidir?", store
    )
    boosted_scores = dict(boosted)

    assert boosted_scores[0] == 0.05
    assert boosted_scores[1] == 0.03 + FACULTY_STAJ_BOOST


def test_is_definition_seeking_query():
    assert is_definition_seeking_query("Burs ile destek arasındaki fark nedir?")
    assert is_definition_seeking_query("CO-OP nedir?")
    assert not is_definition_seeking_query("Kütüphaneden kaç kitap ödünç alabilirim?")


def test_find_definition_chunks_matching_query_finds_lettered_term():
    store = _FakeStore({
        0: _record("reg.pdf", text="Tanımlar\na) Burs: Öğrenim ücretine yapılan desteği,", article_no="4-a"),
        1: _record(
            "reg.pdf",
            text="Tanımlar\nb) CO-OP: Şirketler ile yapılan anlaşmayı koordine eden birimi,",
            article_no="4-b",
        ),
        2: _record("reg.pdf", text="Madde 9 - genel hüküm, burs ile ilgisi yok.", article_no="9"),
    })

    matches = find_definition_chunks_matching_query("Burs ile destek arasındaki fark nedir?", store)

    assert matches == [0]


def test_apply_category_boost_injects_definition_chunk_absent_from_fused_pool():
    # Regression test: the correct chunk sometimes doesn't rank in ANY dense/sparse
    # top-k for a definitional query, so it never reaches apply_category_boost via
    # `fused` at all — the fix must inject it directly, not just re-score it.
    store = _FakeStore({
        0: _record("other.pdf", text="Alakasız içerik."),
        1: _record("reg.pdf", text="Tanımlar\na) Burs: Öğrenim ücretine yapılan desteği,", article_no="4-a"),
    })
    fused = [(0, 0.05)]  # chunk 1 is absent — simulates it missing from retrieval entirely

    boosted = apply_category_boost(fused, "Burs nedir?", store)

    assert boosted[0][0] == 1
    assert boosted[0][1] > 0.05


def test_apply_category_boost_promotes_definition_chunk_even_when_already_ranked_low():
    # Regression test: whether a matched definition chunk shows up in `fused` at all
    # varies run-to-run (embedding/BM25 recall for these short dry definitions is
    # unreliable — see is_definition_seeking_query's module comment). A chunk that IS
    # present but buried near the bottom of `fused` must be promoted just as much as
    # one that's fully absent — an earlier version of this fix only injected absent
    # chunks and left already-present-but-buried ones stuck at their tiny original
    # score, so the same query could pass or fail depending on retrieval noise alone.
    store = _FakeStore({
        0: _record("other.pdf", text="Alakasız içerik.", article_no="1"),
        1: _record("reg.pdf", text="Tanımlar\na) Burs: Öğrenim ücretine yapılan desteği,", article_no="4-a"),
    })
    fused = [(0, 0.5), (1, 0.001)]  # chunk 1 present, but buried far below chunk 0

    boosted = apply_category_boost(fused, "Burs nedir?", store)

    assert boosted[0][0] == 1
    assert boosted[0][1] > 0.5


def test_is_add_drop_query_matches_english_and_turkish_phrasings():
    assert is_add_drop_query("which date is the add and drop week")
    assert is_add_drop_query("ekle sil haftası ne zaman?")
    assert is_add_drop_query("ders ekle bırak haftası hangi tarihte?")
    assert not is_add_drop_query("Kütüphaneden kaç kitap ödünç alabilirim?")


def test_apply_category_boost_promotes_add_drop_calendar_chunk():
    # Regression test: the calendar chunk naming the actual add-drop dates is real
    # and dense-retrievable, but calendar files are long and full of similarly-worded
    # deadline/date chunks, so this one routinely lands just outside the rerank pool
    # (observed rank 7-8 of ~86) rather than being absent — a small boost is enough.
    store = _FakeStore({
        0: _record("other.pdf", text="Ödeme son tarihi ile ilgili genel bilgi."),
        1: _record(
            "BAU_Akademik_Takvim_Onlisans_Lisans.txt",
            text="28 Eylül-02 Ekim 2026 Pazartesi-Cuma — EKLE-SİL haftası.",
        ),
    })
    fused = [(0, 0.05), (1, 0.03)]

    boosted = apply_category_boost(fused, "which date is the add and drop week", store)

    assert boosted[0][0] == 1
    assert boosted[0][1] == 0.03 + ADD_DROP_BOOST


def test_is_summer_school_course_limit_query():
    assert is_summer_school_course_limit_query("Yaz okulunda en fazla kaç ders alabilirim?")
    assert is_summer_school_course_limit_query("How many courses can I take in summer school?")
    assert not is_summer_school_course_limit_query("Yaz okulu ne zaman başlar?")


def test_apply_category_boost_promotes_summer_school_course_count_chunk():
    # Regression test: Yaz Okulu Yönergesi's own Madde 9 (the credit cap, "10 kredi")
    # answers a related-but-incomplete version of this question and, once properly
    # chunked, dominates the rerank pool on its own — crowding out the chunk in a
    # DIFFERENT document (the general Eğitim-Öğretim Yönetmeliği) that states the
    # actual course-count cap ("en fazla dört ders").
    store = _FakeStore({
        0: _record(
            "BAHÇEŞEHİR ÜNİVERSİTESİ YAZ OKULU YÖNERGESİ.pdf",
            text="Yaz Okulu'nda toplam ders yükü 10 (On) kredidir.",
        ),
        1: _record(
            "BAHÇEŞEHİR ÜNİVERSİTESİ ÖN LİSANS VE LİSANS EĞİTİM-ÖĞRETİM VE SINAV YÖNETMELİĞİ.pdf",
            text="Bir öğrenci, yaz okulunda 10 ulusal krediyi geçmemek üzere en fazla dört ders alabilir.",
        ),
    })
    fused = [(0, 0.1), (1, 0.03)]

    boosted = apply_category_boost(fused, "Yaz okulunda en fazla kaç ders alabilirim?", store)
    boosted_scores = dict(boosted)

    assert boosted_scores[0] == 0.1
    assert boosted_scores[1] == 0.03 + SUMMER_SCHOOL_COURSE_LIMIT_BOOST
