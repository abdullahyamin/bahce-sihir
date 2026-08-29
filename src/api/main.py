from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.config import settings
from src.retrieval.pipeline import RAGPipeline

pipeline: RAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    pipeline = RAGPipeline()
    yield


app = FastAPI(title="Bahce-sihir", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "gemini_api_key_loaded": bool(settings.gemini_api_key),
    }


class QueryRequest(BaseModel):
    query: str


class SourceRef(BaseModel):
    source_file: str
    category: str
    section: str | None
    article_no: str | None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceRef]


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    result = pipeline.answer(request.query)
    return QueryResponse(**result)
