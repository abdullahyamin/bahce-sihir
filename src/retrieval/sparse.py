import pickle

import numpy as np

from src.config import settings
from src.text_utils import tokenize_turkish


class SparseRetriever:
    def __init__(self):
        with open(settings.bm25_index_path, "rb") as f:
            self.bm25 = pickle.load(f)

    def search(self, query: str, k: int | None = None) -> list[tuple[int, float]]:
        k = k or settings.sparse_top_k
        scores = self.bm25.get_scores(tokenize_turkish(query))
        top_indices = np.argsort(scores)[::-1][:k]
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]
