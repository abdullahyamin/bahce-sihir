from src.retrieval.store import ChunkRecord

SYSTEM_INSTRUCTIONS = """You are Bahce-sihir, an assistant that answers Bahçeşehir University \
students' questions using ONLY the official university regulation excerpts provided as context.

Rules:
1. Answer using ONLY information present in the provided context. Do not use outside knowledge.
2. If the context does not contain enough information to answer the question, say so clearly \
instead of guessing.
3. Write the answer itself as clean, uninterrupted prose with NO citations mixed into the \
sentences. Then, after the answer, on a new line, add ONE line starting exactly with "SOURCES:" \
followed by a compact list of the source document names and article/section numbers used \
(e.g. "SOURCES: Yaz Okulu Yönergesi, Madde 9; Sınav Yönergesi, Madde 14"). Always include this \
SOURCES line, even for a short answer.
4. Respond in English by default. Only respond in Turkish if the student's question itself \
was written in Turkish. This applies regardless of the language of the source documents.
5. If the question asks for a personal opinion, a prediction, or is about a specific individual, \
politely explain this is outside what you can help with and suggest contacting university \
staff directly."""


def build_context_block(chunks: list[ChunkRecord]) -> str:
    if not chunks:
        return "(No relevant regulation excerpts were found for this question.)"

    parts = []
    for i, chunk in enumerate(chunks, start=1):
        header = f"[Kaynak {i}] {chunk.source_file}"
        if chunk.section:
            header += f" — {chunk.section}"
        if chunk.article_no:
            header += f" — Madde {chunk.article_no}"
        parts.append(f"{header}\n{chunk.text}")
    return "\n\n---\n\n".join(parts)


def build_prompt(query: str, chunks: list[ChunkRecord]) -> str:
    context = build_context_block(chunks)
    return (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        f"Context excerpts from official Bahçeşehir University regulations:\n\n"
        f"{context}\n\n"
        f"Student question: {query}\n\n"
        f"IMPORTANT: write your answer in English, unless the student question above was "
        f"written in Turkish — in that case, write your answer in Turkish. Do not switch to "
        f"the context's language. Remember to end with a \"SOURCES:\" line listing the "
        f"documents/articles used, on its own line, after the answer.\n\n"
        f"Answer:"
    )
