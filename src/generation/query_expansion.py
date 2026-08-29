from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from src.config import settings

_llm: ChatGoogleGenerativeAI | None = None


class QueryVariants(BaseModel):
    turkish: list[str] = Field(description="Alternative Turkish phrasings of the question")
    english: list[str] = Field(description="Alternative English phrasings of the question")


def _get_llm() -> ChatGoogleGenerativeAI:
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            temperature=0.0,
            google_api_key=settings.gemini_api_key,
        )
    return _llm


def expand_query(query: str, n: int | None = None) -> list[str]:
    n = n or settings.query_expansion_n
    structured_llm = _get_llm().with_structured_output(QueryVariants)

    prompt = (
        f"You are helping search a Turkish university regulations database. "
        f"Given the student's question below, generate {n} alternative Turkish phrasings "
        f"and {n} alternative English phrasings that preserve the exact same meaning and intent. "
        f"Do this regardless of what language the original question is written in — always "
        f"produce both Turkish and English variants.\n\nQuestion: {query}"
    )
    result: QueryVariants = structured_llm.invoke(prompt)

    variants = [query, *result.turkish, *result.english]
    seen: set[str] = set()
    deduped: list[str] = []
    for variant in variants:
        key = variant.strip().lower()
        if key and key not in seen:
            seen.add(key)
            deduped.append(variant.strip())
    return deduped
