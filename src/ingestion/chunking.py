import re
from bisect import bisect_right
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

MAX_CHUNK_CHARS = 1800
CHUNK_OVERLAP = 200

# Some older yönerge documents (e.g. Yaz Okulu Yönergesi) terminate the article
# heading with a period ("Madde 1.") instead of the far more common dash/en-dash/
# em-dash ("Madde 1 -"). Without matching that, the whole document falls through
# to naive character-based chunking and loses article-level structure entirely.
_ARTICLE_RE = re.compile(r"(?im)^[ \t]*Madde[ \t]+(\d+)[ \t]*[-–—.]")
_QA_RE = re.compile(r"(?im)^SORU:[ \t]*")
_SUBCLAUSE_RE = re.compile(r"(?m)^\((\d+)\)[ \t]")
_LETTERED_ITEM_RE = re.compile(r"(?m)^([a-zçğıöşü])\)[ \t]")
_CHAPTER_RE = re.compile(
    r"(?im)^[ \t]*("
    r"BİRİNCİ|İKİNCİ|ÜÇÜNCÜ|DÖRDÜNCÜ|BEŞİNCİ|ALTINCI|YEDİNCİ|SEKİZİNCİ|DOKUZUNCU|ONUNCU"
    r")[ \t]+BÖLÜM[ \t]*$"
)

_fallback_splitter = RecursiveCharacterTextSplitter(
    chunk_size=MAX_CHUNK_CHARS,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


@dataclass
class Chunk:
    text: str
    source_file: str
    category: str
    section: str | None
    article_no: str | None


def _find_chapter_headings(lines: list[str]) -> tuple[list[int], list[str]]:
    chapter_line_indices = []
    chapter_titles = []
    for i, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line or not _CHAPTER_RE.match(line):
            continue
        title = ""
        for j in range(i + 1, min(i + 3, len(lines))):
            candidate = lines[j].strip()
            if candidate:
                title = candidate
                break
        chapter_line_indices.append(i)
        chapter_titles.append(f"{line} - {title}" if title else line)
    return chapter_line_indices, chapter_titles


def _short_heading_before(lines: list[str], idx: int) -> str | None:
    for j in range(idx - 1, max(idx - 3, -1), -1):
        line = lines[j].strip()
        if not line:
            continue
        if len(line) < 100 and not line.endswith((".", ")", ":")) and not _ARTICLE_RE.match(line):
            return line
        break
    return None


def _split_oversized_body(body: str) -> list[str]:
    # Long articles often bundle several unrelated numbered sub-clauses "(N) ..." into one
    # block (e.g. course-load reduction, GPA-based load increase, summer school limits all
    # under one MADDE). Splitting blindly by character count dilutes each sub-clause's
    # embedding with unrelated neighbors. Split on sub-clause boundaries first when present.
    matches = list(_SUBCLAUSE_RE.finditer(body))
    if len(matches) < 2:
        return _fallback_splitter.split_text(body)

    boundaries = [m.start() for m in matches]
    segments = []
    prev = 0
    for b in boundaries:
        if b > prev:
            segments.append(body[prev:b])
        prev = b
    segments.append(body[prev:])
    segments = [s for s in segments if s.strip()]

    chunks: list[str] = []
    current = ""
    for seg in segments:
        if len(seg) > MAX_CHUNK_CHARS:
            if current.strip():
                chunks.append(current.strip())
                current = ""
            chunks.extend(_fallback_splitter.split_text(seg))
        elif current and len(current) + len(seg) > MAX_CHUNK_CHARS:
            chunks.append(current.strip())
            current = seg
        else:
            current += seg
    if current.strip():
        chunks.append(current.strip())
    return chunks


def _split_definitions_body(body: str) -> tuple[str, list[str]] | None:
    # "Tanımlar" (definitions) articles bundle many unrelated terms as lettered items
    # (a) b) c) ...) under one MADDE — e.g. a scholarship regulation's Tanımlar article
    # defines "Burs", "CO-OP", "Enstitü", "Üst Yönetim" and a dozen other unrelated terms
    # in the same block. Even well under MAX_CHUNK_CHARS, that dilutes the embedding for
    # any single term a student actually asks about. Split one chunk per lettered item
    # regardless of overall length — dilution, not overflow, is the problem here.
    matches = list(_LETTERED_ITEM_RE.finditer(body))
    if len(matches) < 3:
        return None

    local_preamble = body[: matches[0].start()].strip()
    boundaries = [m.start() for m in matches]
    items = []
    for i, b in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(body)
        items.append(body[b:end].strip())
    return local_preamble, items


def _split_by_article(text: str, source_file: str, category: str) -> list[Chunk]:
    lines = text.split("\n")
    match_line_indices = []
    char_pos = 0
    line_start_positions = []
    for line in lines:
        line_start_positions.append(char_pos)
        char_pos += len(line) + 1

    matches = list(_ARTICLE_RE.finditer(text))
    if not matches:
        return []

    for m in matches:
        line_idx = bisect_right(line_start_positions, m.start()) - 1
        match_line_indices.append((m, line_idx))

    chapter_line_indices, chapter_titles = _find_chapter_headings(lines)

    chunks: list[Chunk] = []
    for idx, (m, line_idx) in enumerate(match_line_indices):
        article_no = m.group(1)
        chapter_idx = bisect_right(chapter_line_indices, line_idx) - 1
        section = chapter_titles[chapter_idx] if chapter_idx >= 0 else None
        heading = _short_heading_before(lines, line_idx)

        body_start = m.start()
        body_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()

        prefix_parts = [p for p in (section, heading) if p]
        prefix = "\n".join(prefix_parts)
        full_text = f"{prefix}\n{body}" if prefix else body

        definitions = (
            _split_definitions_body(body) if heading and heading.strip().lower() == "tanımlar" else None
        )
        if definitions is not None:
            local_preamble, items = definitions
            for item in items:
                letter = item[0] if item else "?"
                item_with_context = f"{prefix}\n{local_preamble}\n{item}" if prefix else f"{local_preamble}\n{item}"
                chunks.append(
                    Chunk(text=item_with_context, source_file=source_file, category=category,
                          section=section, article_no=f"{article_no}-{letter}")
                )
        elif len(full_text) <= MAX_CHUNK_CHARS:
            chunks.append(
                Chunk(text=full_text, source_file=source_file, category=category,
                      section=section, article_no=article_no)
            )
        else:
            sub_texts = _split_oversized_body(body)
            for part_idx, sub in enumerate(sub_texts):
                sub_with_prefix = f"{prefix}\n{sub}" if prefix else sub
                chunks.append(
                    Chunk(text=sub_with_prefix, source_file=source_file, category=category,
                          section=section, article_no=f"{article_no} (bölüm {part_idx + 1})")
                )

    preamble = text[: matches[0].start()].strip()
    if len(preamble) > 100:
        chunks.insert(
            0,
            Chunk(text=preamble, source_file=source_file, category=category,
                  section=None, article_no=None),
        )

    return chunks


def _split_by_qa(text: str, source_file: str, category: str) -> list[Chunk]:
    # FAQ-scraped pages (web_sss) have no MADDE/article structure — each question is its
    # own self-contained unit, so one chunk per Q&A pair retrieves far more precisely than
    # forcing them through the generic character-based splitter.
    matches = list(_QA_RE.finditer(text))
    if not matches:
        return []

    title = _extract_title(text[: matches[0].start()]) or _extract_title(text)

    chunks: list[Chunk] = []
    for idx, m in enumerate(matches):
        start = m.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        pair = text[start:end].strip()
        full_text = f"{title}\n\n{pair}" if title else pair

        if len(full_text) <= MAX_CHUNK_CHARS:
            chunks.append(
                Chunk(text=full_text, source_file=source_file, category=category,
                      section=title or None, article_no=None)
            )
        else:
            for part_idx, sub in enumerate(_fallback_splitter.split_text(pair)):
                sub_with_prefix = f"{title}\n\n{sub}" if title else sub
                chunks.append(
                    Chunk(text=sub_with_prefix, source_file=source_file, category=category,
                          section=title or None, article_no=f"parça {part_idx + 1}")
                )
    return chunks


def _extract_title(text: str) -> str:
    # Documents without MADDE structure (e.g. guides, not formal regulations) get no
    # section/heading prefix from _split_by_article, so their fallback chunks start
    # mid-sentence with no topical context — this hurts embedding quality for chunks
    # that are dense with dates/URLs/codes rather than plain descriptive language.
    # Prepending the document's own title line gives every chunk that context back.
    for line in text.split("\n"):
        stripped = line.strip()
        if len(stripped) > 15:
            return stripped[:200]
    return ""


def chunk_document(text: str, source_file: str, category: str) -> list[Chunk]:
    qa_chunks = _split_by_qa(text, source_file, category)
    if qa_chunks:
        return qa_chunks

    article_chunks = _split_by_article(text, source_file, category)
    if article_chunks:
        return article_chunks

    title = _extract_title(text)
    return [
        Chunk(
            text=f"{title}\n\n{part}" if title else part,
            source_file=source_file,
            category=category,
            section=None,
            article_no=None,
        )
        for part in _fallback_splitter.split_text(text)
    ]
