from moderngl_docs_mcp.retrieval.fusion import ( 
    fuse_rankings,
    rank_by_fused_score,
)

def test_fuse_rankings_single_list_matches_hand_calculation():
    # k=1 for easy arithmetic: rank 0 -> 1/(1+0)=1.0, rank 1 -> 1/(1+1)=0.5
    scores = fuse_rankings([100, 200], k=1)
    assert scores[100] == 1.0
    assert scores[200] == 0.5

def test_fuse_rankings_two_lists_with_ties():
    # k=1 for easy arithmetic
    scores = fuse_rankings([100, 200], [100, 300], k=1)
    assert scores[100] == 2.0  # 1.0 + 1.0
    assert scores[200] == 0.5  # 0.5 + 0.0
    assert scores[300] == 0.5  # 0.0 + 0.5

def test_fuse_rankings_empty_input():
    scores = fuse_rankings()
    assert scores == {}

def test_fuse_rankings_item_in_one_list_only():
    scores = fuse_rankings([100, 200], [300, 400], k=1)
    assert scores[100] == 1.0
    assert scores[200] == 0.5
    assert scores[300] == 1.0
    assert scores[400] == 0.5

def test_fuse_rankings_item_in_neither_list():
    scores = fuse_rankings([100, 200], [300, 400], k=1)
    assert 500 not in scores

def test_rank_by_fused_score_sorts_correctly():
    scores = {100: 2.0, 200: 0.5, 300: 1.0}
    ranked = rank_by_fused_score(scores)
    assert ranked == [100, 300, 200]
def test_rank_by_fused_score_tie_breaks_by_id():
    scores = {100: 1.0, 200: 1.0, 300: 0.5}
    ranked = rank_by_fused_score(scores)
    assert ranked == [100, 200, 300]