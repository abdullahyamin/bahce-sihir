import time

from src.config import settings
from src.generation.generator import generate_answer
from src.generation.query_expansion import expand_query
from src.retrieval.dense import DenseRetriever
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.reranker import Reranker
from src.retrieval.sparse import SparseRetriever
from src.retrieval.store import ChunkStore


def timed(label, fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    print(f"{label}: {elapsed:.2f}s")
    return result, elapsed


def main():
    query = "Yaz okulunda kaç ders alabilirim?"

    print("Loading models...")
    store, _ = timed("  ChunkStore load", ChunkStore)
    dense, _ = timed("  DenseRetriever load", DenseRetriever)
    sparse, _ = timed("  SparseRetriever load", SparseRetriever)
    reranker, _ = timed("  Reranker load", Reranker)

    print("\n--- Query pipeline ---")
    variants, t_expand = timed("Query expansion (Gemini)", expand_query, query)
    print(f"  {len(variants)} variants: {variants}")

    start = time.perf_counter()
    ranked_lists = list(dense.search_batch(variants))
    for v in variants:
        ranked_lists.append(sparse.search(v))
    t_retrieve = time.perf_counter() - start
    print(f"Hybrid retrieval ({len(variants)} variants x 2 paths): {t_retrieve:.2f}s")

    fused, t_fuse = timed("RRF fusion", reciprocal_rank_fusion, ranked_lists)
    candidate_indices = [idx for idx, _ in fused[: settings.rerank_candidate_pool_size]]

    reranked, t_rerank = timed(
        f"Reranking ({len(candidate_indices)} candidates)",
        reranker.rerank, query, candidate_indices, store,
    )
    filtered = [
        (idx, score) for idx, score in reranked if score >= settings.min_reranker_score
    ][: settings.rerank_top_k]
    chunks = [store[idx] for idx, _ in filtered]

    answer, t_generate = timed("Answer generation (Gemini)", generate_answer, query, chunks)

    total = t_expand + t_retrieve + t_fuse + t_rerank + t_generate
    print(f"\n--- TOTAL (excluding model load): {total:.2f}s ---")
    print(f"  Query expansion : {t_expand:6.2f}s ({t_expand/total*100:4.1f}%)")
    print(f"  Hybrid retrieval: {t_retrieve:6.2f}s ({t_retrieve/total*100:4.1f}%)")
    print(f"  RRF fusion      : {t_fuse:6.2f}s ({t_fuse/total*100:4.1f}%)")
    print(f"  Reranking       : {t_rerank:6.2f}s ({t_rerank/total*100:4.1f}%)")
    print(f"  Generation      : {t_generate:6.2f}s ({t_generate/total*100:4.1f}%)")


if __name__ == "__main__":
    main()
