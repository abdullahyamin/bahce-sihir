from src.ingestion.chunking import MAX_CHUNK_CHARS, chunk_document


def test_article_structure_produces_one_chunk_per_madde():
    text = (
        "BİRİNCİ BÖLÜM\n"
        "Amaç ve Kapsam\n"
        "Madde 1- Bu yönergenin amacı öğrenci işlerini düzenlemektir.\n"
        "Madde 2- Bu yönerge tüm öğrencileri kapsar.\n"
    )
    chunks = chunk_document(text, "test.pdf", "yonergeler")

    assert len(chunks) == 2
    assert chunks[0].article_no == "1"
    assert chunks[1].article_no == "2"
    assert all(c.source_file == "test.pdf" for c in chunks)
    assert all(c.category == "yonergeler" for c in chunks)


def test_article_structure_recognizes_period_terminated_headers():
    # Regression test: some older yönerge documents (e.g. Yaz Okulu Yönergesi) use
    # "Madde 1." instead of "Madde 1 -". Without recognizing this, the whole document
    # fell through to naive character-based chunking and lost article structure
    # entirely, which buried a specific fact (max summer-school course count) deep in
    # an unstructured chunk instead of a cleanly retrievable per-article one.
    text = (
        "BİRİNCİ BÖLÜM\n"
        "Amaç ve Kapsam\n"
        "Madde 1. \n"
        "Bu yönergenin amacı öğrenci işlerini düzenlemektir.\n"
        "Madde 2. \n"
        "Bu yönerge tüm öğrencileri kapsar.\n"
    )
    chunks = chunk_document(text, "test.pdf", "yonergeler")

    assert len(chunks) == 2
    assert chunks[0].article_no == "1"
    assert chunks[1].article_no == "2"


def test_article_chunk_carries_chapter_as_section():
    text = (
        "BİRİNCİ BÖLÜM\n"
        "Genel Hükümler\n"
        "Madde 1- İçerik burada.\n"
    )
    chunks = chunk_document(text, "test.pdf", "yonergeler")

    assert chunks[0].section is not None
    assert "BİRİNCİ BÖLÜM" in chunks[0].section
    assert "Genel Hükümler" in chunks[0].section


def test_oversized_article_splits_on_subclauses():
    body = "Madde 5- " + "".join(
        f"({n}) " + ("Uzun madde metni. " * 40) for n in range(1, 5)
    )
    chunks = chunk_document(body, "test.pdf", "yonergeler")

    assert len(chunks) > 1
    assert all(c.article_no.startswith("5 (bölüm") for c in chunks)
    assert all(len(c.text) <= MAX_CHUNK_CHARS + 250 for c in chunks)


def test_qa_structure_produces_one_chunk_per_pair():
    text = (
        "BAU Kütüphane SSS\n\n"
        "SORU: Kaç kitap ödünç alabilirim?\n"
        "CEVAP: Lisans öğrencileri 10 kitap ödünç alabilir.\n\n"
        "SORU: Geç iade cezası nedir?\n"
        "CEVAP: Günlük 1 TL ceza uygulanır.\n"
    )
    chunks = chunk_document(text, "sss.txt", "web_sss")

    assert len(chunks) == 2
    assert all(c.article_no is None for c in chunks)
    assert "Kaç kitap" in chunks[0].text
    assert "Geç iade" in chunks[1].text
    # title line gets prefixed onto every chunk for topical context
    assert "BAU Kütüphane SSS" in chunks[0].text
    assert "BAU Kütüphane SSS" in chunks[1].text


def test_qa_takes_priority_over_article_splitting():
    # A QA-formatted doc should never fall through to the MADDE splitter even if the
    # answer text happens to mention "Madde" in prose.
    text = (
        "SORU: Hangi maddeye göre karar verilir?\n"
        "CEVAP: Madde 5 uyarınca karar verilir.\n"
    )
    chunks = chunk_document(text, "sss.txt", "web_sss")

    assert len(chunks) == 1
    assert chunks[0].article_no is None


def test_no_structure_falls_back_to_character_splitter_with_title():
    text = "BAU Yurt İmkanları\n\n" + ("Uzun genel bilgi metni. " * 200)
    chunks = chunk_document(text, "genel.txt", "web_genel_bilgi")

    assert len(chunks) >= 1
    assert chunks[0].article_no is None
    assert chunks[0].section is None
    assert chunks[0].text.startswith("BAU Yurt İmkanları")


def test_empty_document_produces_no_crash():
    chunks = chunk_document("", "empty.txt", "web_genel_bilgi")
    assert chunks == [] or all(c.text == "" for c in chunks)


def test_tanimlar_article_splits_one_chunk_per_lettered_definition():
    # Regression test: a "Tanımlar" (definitions) article bundling many unrelated
    # terms as lettered items diluted the embedding for any single term (e.g. "Burs"
    # unretrievable because it shared a chunk with CO-OP, Enstitü, Üst Yönetim, etc.).
    text = (
        "BİRİNCİ BÖLÜM\n"
        "Tanımlar\n"
        "Madde 4- (1) Bu Yönergede geçen,\n"
        "a) Burs: Öğrenim ücretine yapılan desteği,\n"
        "b) CO-OP: Şirketler ile yapılan eğitim anlaşmalarını koordine eden birimi,\n"
        "c) Destek: Öğrenciye yapılan nakdi nitelikteki katkıları,\n"
        "ifade eder.\n"
    )
    chunks = chunk_document(text, "test.pdf", "yonergeler")

    assert len(chunks) == 3
    assert [c.article_no for c in chunks] == ["4-a", "4-b", "4-c"]
    burs_chunk = chunks[0]
    assert "Burs:" in burs_chunk.text
    assert "CO-OP" not in burs_chunk.text
    assert "Destek:" not in burs_chunk.text
    # each item still carries the article's section/heading context
    assert "Tanımlar" in burs_chunk.text


def test_tanimlar_article_with_few_items_falls_back_to_normal_splitting():
    # Fewer than 3 lettered items isn't the bundling problem this fix targets —
    # avoid needlessly fragmenting short definitions articles.
    text = (
        "BİRİNCİ BÖLÜM\n"
        "Tanımlar\n"
        "Madde 4- (1) Bu Yönergede geçen,\n"
        "a) Üniversite: Bahçeşehir Üniversitesini,\n"
        "b) Rektör: Üniversite Rektörünü,\n"
        "ifade eder.\n"
    )
    chunks = chunk_document(text, "test.pdf", "yonergeler")

    assert len(chunks) == 1
    assert chunks[0].article_no == "4"
