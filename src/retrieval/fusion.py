from src.config import settings


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[int, float]]], k: int | None = None
) -> list[tuple[int, float]]:
    k = k or settings.rrf_k
    combined_scores: dict[int, float] = {}

    for ranked_list in ranked_lists:
        for rank, (chunk_idx, _score) in enumerate(ranked_list, start=1):
            combined_scores[chunk_idx] = combined_scores.get(chunk_idx, 0.0) + 1.0 / (k + rank)

    return sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)
