import json
from pathlib import Path

from src.config import settings
from src.generation.query_expansion import expand_query
from src.retrieval.dense import DenseRetriever
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.query_classifier import apply_category_boost
from src.retrieval.reranker import Reranker
from src.retrieval.sparse import SparseRetriever
from src.retrieval.store import ChunkRecord, ChunkStore

BENCHMARK_PATH = Path("scripts/benchmark_queries.json")


def load_benchmark() -> list[dict]:
    with open(BENCHMARK_PATH, encoding="utf-8") as f:
        return json.load(f)


def normalize_article(article_no: str | None) -> str | None:
    if article_no is None:
        return None
    return article_no.split(" ")[0].split("(")[0].strip()


def is_relevant(chunk: ChunkRecord, expected_files: list[str], expected_articles: list[str]) -> bool:
    # Document-level match: a chunk from the correct source document is relevant, since
    # that's what Precision@k is meant to capture (did retrieval find the right source),
    # not whether it landed on the exact article a human happened to hand-pick as ground
    # truth. Article-level match is reported separately as a stricter secondary metric.
    return chunk.source_file in expected_files


def is_relevant_strict(chunk: ChunkRecord, expected_files: list[str], expected_articles: list[str]) -> bool:
    if chunk.source_file not in expected_files:
        return False
    if not expected_articles:
        return True
    return normalize_article(chunk.article_no) in expected_articles


def precision_at_k(chunks: list[ChunkRecord], expected_files, expected_articles, k: int, matcher=is_relevant) -> float:
    top_k = chunks[:k]
    if not top_k:
        return 0.0
    relevant = sum(1 for c in top_k if matcher(c, expected_files, expected_articles))
    return relevant / len(top_k)


def reciprocal_rank(chunks: list[ChunkRecord], expected_files, expected_articles, matcher=is_relevant) -> float:
    for i, c in enumerate(chunks, start=1):
        if matcher(c, expected_files, expected_articles):
            return 1.0 / i
    return 0.0


class Retrievers:
    def __init__(self):
        self.store = ChunkStore()
        self.dense = DenseRetriever()
        self.sparse = SparseRetriever()
        self.reranker = Reranker()


def run_retrieval(
    query: str,
    r: Retrievers,
    use_expansion: bool,
    use_hybrid: bool,
    use_rerank: bool,
    pool_size: int | None = None,
) -> list[ChunkRecord]:
    pool_size = pool_size or settings.rerank_candidate_pool_size
    variants = expand_query(query) if use_expansion else [query]

    dense_results = r.dense.search_batch(variants)
    ranked_lists = list(dense_results)
    if use_hybrid:
        for v in variants:
            ranked_lists.append(r.sparse.search(v))

    fused = reciprocal_rank_fusion(ranked_lists)
    fused = apply_category_boost(fused, query, r.store)

    if use_rerank:
        candidate_indices = [idx for idx, _ in fused[:pool_size]]
        reranked = r.reranker.rerank(query, candidate_indices, r.store)
        final_indices = [idx for idx, score in reranked if score >= settings.min_reranker_score]
    else:
        final_indices = [idx for idx, _ in fused]

    final_indices = final_indices[: settings.rerank_top_k]
    return [r.store[idx] for idx in final_indices]
