"""API-layer tests. The real RAGPipeline loads heavy ML models on construction, so
it's replaced with a lightweight fake before the app's lifespan constructs it —
these tests exercise the FastAPI request/response contract, not the retrieval
pipeline itself (that's covered by the retrieval-focused test modules).
"""

import pytest
from fastapi.testclient import TestClient

import src.api.main as api_main


class _FakePipeline:
    def answer(self, query: str) -> dict:
        return {
            "answer": f"Fake answer for: {query}\n\nSOURCES: test.pdf, Madde 1",
            "sources": [
                {"source_file": "test.pdf", "category": "yonergeler", "section": None, "article_no": "1"}
            ],
        }


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api_main, "RAGPipeline", _FakePipeline)
    with TestClient(api_main.app) as c:
        yield c


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "gemini_api_key_loaded" in body


def test_query_endpoint_returns_answer_and_sources(client):
    resp = client.post("/query", json={"query": "Yaz okulunda en fazla kaç ders alabilirim?"})
    assert resp.status_code == 200
    body = resp.json()

    assert "Yaz okulunda en fazla kaç ders alabilirim?" in body["answer"]
    assert len(body["sources"]) == 1
    assert body["sources"][0]["source_file"] == "test.pdf"
    assert body["sources"][0]["article_no"] == "1"


def test_query_endpoint_rejects_missing_query_field(client):
    resp = client.post("/query", json={})
    assert resp.status_code == 422
