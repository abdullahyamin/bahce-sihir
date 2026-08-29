"""Integration tests for the three retrieval components that load real ML models
and the real production index (data/processed/*): DenseRetriever, SparseRetriever,
Reranker. These are deliberately NOT mocked — the point is to verify real search
behavior against the actual corpus, not just call-shape. Each model loads once per
test session (session-scoped fixtures), so the whole file costs one model-load
pass (tens of seconds), not one per test.

Run everything except these with: pytest -m "not integration"
"""

import pytest

from src.retrieval.dense import DenseRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.sparse import SparseRetriever
from src.retrieval.store import ChunkStore

pytestmark = pytest.mark.integration


@pytest.fixture(scope="session")
def store():
    return ChunkStore()


@pytest.fixture(scope="session")
def dense():
    return DenseRetriever()


@pytest.fixture(scope="session")
def sparse():
    return SparseRetriever()


@pytest.fixture(scope="session")
def reranker():
    return Reranker()


def _source_files(results, store):
    return [store[idx].source_file for idx, _score in results]


# --- DenseRetriever -----------------------------------------------------------

def test_dense_search_returns_k_results_sorted_descending(dense, store):
    results = dense.search("Sınav sonuçlarına nasıl itiraz edebilirim?", k=8)

    assert len(results) == 8
    assert all(isinstance(idx, int) and 0 <= idx < len(store) for idx, _ in results)
    scores = [score for _idx, score in results]
    assert scores == sorted(scores, reverse=True)


def test_dense_search_batch_matches_query_count(dense):
    queries = ["Yaz okulu kuralları nelerdir?", "How can I apply for Erasmus?"]
    batch = dense.search_batch(queries, k=5)

    assert len(batch) == 2
    assert all(len(r) == 5 for r in batch)


def test_dense_search_finds_topically_relevant_document(dense, store):
    # A stable, well-known regulation from the original corpus — not something that
    # depends on this session's later corpus-growth additions.
    results = dense.search("Yaz okulunda en fazla kaç ders alabilirim?", k=10)
    sources = _source_files(results, store)

    assert any("YAZ OKULU" in s.upper() for s in sources)


# --- SparseRetriever (BM25) -----------------------------------------------------

def test_sparse_search_returns_only_positive_scores(sparse):
    results = sparse.search("sınav itiraz sonuç", k=10)

    assert all(score > 0 for _idx, score in results)


def test_sparse_search_scores_sorted_descending(sparse):
    results = sparse.search("burs destek öğrenci yönerge", k=10)
    scores = [score for _idx, score in results]

    assert scores == sorted(scores, reverse=True)


def test_sparse_search_exact_keyword_match_surfaces_right_document(sparse, store):
    results = sparse.search("yaz okulu", k=10)
    sources = _source_files(results, store)

    assert any("YAZ OKULU" in s.upper() for s in sources)


def test_sparse_search_nonsense_query_can_return_empty(sparse):
    # BM25 filters out zero-score matches (see SparseRetriever.search) — a query with
    # no lexical overlap at all against the corpus should not fabricate results.
    results = sparse.search("xyzzyplonkqwerty123nonexistentterm", k=10)
    assert results == []


# --- Reranker (cross-encoder) --------------------------------------------------

def test_reranker_empty_candidates_returns_empty(reranker, store):
    assert reranker.rerank("herhangi bir soru", [], store) == []


def test_reranker_scores_all_candidates_and_sorts_descending(reranker, store):
    candidate_indices = list(range(10))
    reranked = reranker.rerank("Sınav sonuçlarına nasıl itiraz edebilirim?", candidate_indices, store)

    assert len(reranked) == len(candidate_indices)
    assert {idx for idx, _score in reranked} == set(candidate_indices)
    scores = [score for _idx, score in reranked]
    assert scores == sorted(scores, reverse=True)


def test_reranker_prefers_topically_relevant_chunk(dense, reranker, store):
    query = "Sınav sonuçlarına kaç iş günü içinde itiraz edebilirim?"

    # Find one clearly relevant chunk (from dense search) and one clearly unrelated
    # chunk (a distant match) rather than hardcoding indices, so this stays valid
    # if the corpus changes later.
    dense_results = dense.search(query, k=1)
    relevant_idx = dense_results[0][0]
    far_results = dense.search(query, k=len(store))
    unrelated_idx = far_results[-1][0]

    reranked = reranker.rerank(query, [relevant_idx, unrelated_idx], store)
    top_idx, _top_score = reranked[0]

    assert top_idx == relevant_idx
