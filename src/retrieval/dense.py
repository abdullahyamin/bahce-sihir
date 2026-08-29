import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import settings


class DenseRetriever:
    def __init__(self):
        self.model = SentenceTransformer(settings.embedding_model_name)
        self.embeddings = np.load(settings.embeddings_path)

    def _search_from_embedding(self, query_embedding: np.ndarray, k: int) -> list[tuple[int, float]]:
        scores = self.embeddings @ query_embedding
        top_indices = np.argsort(scores)[::-1][:k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]

    def search(self, query: str, k: int | None = None) -> list[tuple[int, float]]:
        k = k or settings.dense_top_k
        query_embedding = self.model.encode([query], normalize_embeddings=True).astype("float32")[0]
        return self._search_from_embedding(query_embedding, k)

    def search_batch(
        self, queries: list[str], k: int | None = None
    ) -> list[list[tuple[int, float]]]:
        k = k or settings.dense_top_k
        query_embeddings = self.model.encode(queries, normalize_embeddings=True).astype("float32")
        return [self._search_from_embedding(emb, k) for emb in query_embeddings]
