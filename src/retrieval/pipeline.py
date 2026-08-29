from src.config import settings
from src.generation.generator import generate_answer
from src.generation.query_expansion import expand_query
from src.retrieval.dense import DenseRetriever
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.query_classifier import apply_category_boost
from src.retrieval.reranker import Reranker
from src.retrieval.sparse import SparseRetriever
from src.retrieval.store import ChunkStore


class RAGPipeline:
    def __init__(self):
        self.store = ChunkStore()
        self.dense = DenseRetriever()
        self.sparse = SparseRetriever()
        self.reranker = Reranker()

    def answer(self, query: str) -> dict:
        variants = expand_query(query)

        dense_results = self.dense.search_batch(variants)
        ranked_lists = list(dense_results)
        for variant in variants:
            ranked_lists.append(self.sparse.search(variant))

        fused = reciprocal_rank_fusion(ranked_lists)
        fused = apply_category_boost(fused, query, self.store)
        candidate_indices = [idx for idx, _score in fused[: settings.rerank_candidate_pool_size]]

        reranked = self.reranker.rerank(query, candidate_indices, self.store)
        filtered = [
            (idx, score) for idx, score in reranked if score >= settings.min_reranker_score
        ][: settings.rerank_top_k]

        chunks = [self.store[idx] for idx, _score in filtered]
        answer_text = generate_answer(query, chunks)

        sources = [
            {
                "source_file": c.source_file,
                "category": c.category,
                "section": c.section,
                "article_no": c.article_no,
            }
            for c in chunks
        ]
        return {"answer": answer_text, "sources": sources}
