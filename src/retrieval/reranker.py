from sentence_transformers import CrossEncoder

from src.config import settings
from src.retrieval.store import ChunkStore


class Reranker:
    def __init__(self):
        self.model = CrossEncoder(settings.reranker_model_name)

    def rerank(
        self, query: str, candidate_indices: list[int], store: ChunkStore
    ) -> list[tuple[int, float]]:
        if not candidate_indices:
            return []
        pairs = [(query, store[idx].text) for idx in candidate_indices]
        scores = self.model.predict(pairs)
        scored = list(zip(candidate_indices, (float(s) for s in scores)))
        return sorted(scored, key=lambda item: item[1], reverse=True)
