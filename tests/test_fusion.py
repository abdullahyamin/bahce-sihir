from src.retrieval.fusion import reciprocal_rank_fusion


def test_single_list_preserves_relative_order():
    ranked_list = [(10, 0.9), (20, 0.5), (30, 0.1)]
    fused = reciprocal_rank_fusion([ranked_list])

    ids_in_order = [idx for idx, _score in fused]
    assert ids_in_order == [10, 20, 30]


def test_item_ranked_high_in_multiple_lists_wins():
    list_a = [(1, 0.0), (2, 0.0), (3, 0.0)]
    list_b = [(2, 0.0), (1, 0.0), (3, 0.0)]

    fused = reciprocal_rank_fusion([list_a, list_b])

    top_idx, _ = fused[0]
    # item 1 is rank-1 in list_a and rank-2 in list_b; item 2 is rank-2 then rank-1;
    # both should tie for the top RRF score, but either must strictly beat item 3.
    assert top_idx in (1, 2)
    assert fused[-1][0] == 3


def test_item_only_in_one_list_still_included():
    list_a = [(1, 0.0)]
    list_b = [(2, 0.0)]

    fused = reciprocal_rank_fusion([list_a, list_b])
    ids = {idx for idx, _score in fused}
    assert ids == {1, 2}


def test_empty_input_returns_empty():
    assert reciprocal_rank_fusion([]) == []


def test_scores_are_monotonically_non_increasing():
    list_a = [(1, 0.0), (2, 0.0), (3, 0.0), (4, 0.0)]
    fused = reciprocal_rank_fusion([list_a])
    scores = [score for _idx, score in fused]
    assert scores == sorted(scores, reverse=True)
