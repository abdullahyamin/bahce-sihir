import pytest

from src.ingestion.loaders import CATEGORIES, iter_corpus, load_document


def _make_category_dir(tmp_path, category: str):
    # iter_corpus loops over every fixed CATEGORIES entry and calls .iterdir()
    # unconditionally, so every category directory must exist even if empty.
    for name in CATEGORIES:
        (tmp_path / name).mkdir(exist_ok=True)
    return tmp_path / category


def test_load_document_txt_override_takes_priority_over_pdf(tmp_path):
    d = _make_category_dir(tmp_path, "yonergeler")
    pdf_path = d / "scanned.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake, unparseable on purpose")
    (d / "scanned.txt").write_text("Hand-transcribed content.", encoding="utf-8")

    text = load_document(pdf_path)

    assert text == "Hand-transcribed content."


def test_load_document_unsupported_extension_raises(tmp_path):
    d = _make_category_dir(tmp_path, "yonergeler")
    path = d / "notes.md"
    path.write_text("irrelevant", encoding="utf-8")

    with pytest.raises(ValueError):
        load_document(path)


def test_iter_corpus_skips_txt_that_overrides_a_pdf(tmp_path):
    d = _make_category_dir(tmp_path, "yonergeler")
    (d / "doc.pdf").write_bytes(b"%PDF-1.4 fake")
    (d / "doc.txt").write_text("Override text.", encoding="utf-8")

    docs = list(iter_corpus(raw_dir=tmp_path))

    # Only one RawDocument should be yielded for this stem — the .pdf entry, whose
    # content came from the .txt override — not a second, separate entry for the .txt.
    matching = [doc for doc in docs if doc.source_file.startswith("doc.")]
    assert len(matching) == 1
    assert matching[0].source_file == "doc.pdf"
    assert matching[0].text == "Override text."


def test_iter_corpus_includes_standalone_txt_with_no_sibling(tmp_path):
    d = _make_category_dir(tmp_path, "web_sss")
    (d / "faq.txt").write_text("SORU: test\nCEVAP: cevap", encoding="utf-8")

    docs = list(iter_corpus(raw_dir=tmp_path))

    assert len(docs) == 1
    assert docs[0].source_file == "faq.txt"
    assert docs[0].category == "web_sss"


def test_iter_corpus_ignores_unrelated_file_types(tmp_path):
    d = _make_category_dir(tmp_path, "yonergeler")
    (d / "readme.md").write_text("not a corpus doc", encoding="utf-8")
    (d / ".gitkeep").write_text("", encoding="utf-8")

    docs = list(iter_corpus(raw_dir=tmp_path))

    assert docs == []
