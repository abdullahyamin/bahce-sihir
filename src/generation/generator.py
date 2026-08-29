from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import settings
from src.generation.prompts import build_prompt
from src.retrieval.store import ChunkRecord

_llm: ChatGoogleGenerativeAI | None = None


def _get_llm() -> ChatGoogleGenerativeAI:
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            temperature=0.0,
            google_api_key=settings.gemini_api_key,
        )
    return _llm


def generate_answer(query: str, chunks: list[ChunkRecord]) -> str:
    prompt = build_prompt(query, chunks)
    response = _get_llm().invoke(prompt)
    return response.content
