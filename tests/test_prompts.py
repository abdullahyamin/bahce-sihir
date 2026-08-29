from src.generation.prompts import build_context_block, build_prompt
from src.retrieval.store import ChunkRecord


def _record(**overrides) -> ChunkRecord:
    defaults = dict(text="Örnek metin.", source_file="test.pdf", category="yonergeler",
                     section=None, article_no=None)
    defaults.update(overrides)
    return ChunkRecord(**defaults)


def test_build_context_block_empty_chunks_gives_placeholder():
    block = build_context_block([])
    assert "No relevant" in block


def test_build_context_block_includes_source_and_article():
    chunk = _record(article_no="9", section="BİRİNCİ BÖLÜM")
    block = build_context_block([chunk])

    assert "test.pdf" in block
    assert "Madde 9" in block
    assert "BİRİNCİ BÖLÜM" in block
    assert "Örnek metin." in block


def test_build_context_block_numbers_multiple_sources():
    chunks = [_record(source_file="a.pdf"), _record(source_file="b.pdf")]
    block = build_context_block(chunks)

    assert "[Kaynak 1]" in block
    assert "[Kaynak 2]" in block


def test_build_prompt_includes_query_and_sources_instruction():
    chunk = _record()
    prompt = build_prompt("Sınav sonuçlarına nasıl itiraz ederim?", [chunk])

    assert "Sınav sonuçlarına nasıl itiraz ederim?" in prompt
    assert "SOURCES:" in prompt
    assert "Answer:" in prompt
