"""Reciprocal Rank Fusion (RRF) for combining keyword and vector search rankings.

Pure logic — takes two ranked lists of IDs, returns a fused score per ID.
No database, no embedding model, no row data. This is deliberately the
*only* thing this function does: the original prototype mixed rank-scoring
math with row data (title/content/type) in the same dict, which made it
impossible to test the scoring math without a real database and a real
embedding model in the loop. Separating them means the part most likely to
have a subtle bug (off-by-one ranks, wrong k, wrong tie-break order) can be
tested with nothing but plain Python lists of integers.

RRF itself: each result list contributes 1/(k + rank) to an item's score,
where rank is 0-indexed position in that list and k is a smoothing constant
(60 is the conventional default from the original RRF paper -- Cormack et
al. 2009 -- and is what the prototype already used). An item appearing in
both lists sums both contributions; an item appearing in only one list gets
only that list's contribution. Higher total score = better fused rank.
"""
from __future__ import annotations

DEFAULT_RRF_K = 60


def fuse_rankings(
    *ranked_id_lists: list[int],
    k: int = DEFAULT_RRF_K,
) -> dict[int, float]:
    """Fuse any number of ranked ID lists into a single score-per-ID mapping.

    Args:
        *ranked_id_lists: One or more lists of IDs, each already sorted
            best-match-first (rank 0 = best). The original prototype only
            ever fused exactly two lists (FTS + vector); this accepts any
            number so a future third signal (e.g. a freshness or popularity
            ranking) can be added without changing the function's shape.
        k: RRF smoothing constant. Larger k flattens the score curve
            (rank position matters less); smaller k makes top ranks
            dominate more sharply. 60 is the conventional default.

    Returns:
        Dict mapping each ID that appeared in at least one input list to
        its summed RRF score. IDs that never appeared in any list are not
        present in the result (there is nothing to score). Does not sort —
        sorting by score is the caller's job, since "what to do with ties"
        and "how many results to keep" are policy decisions, not fusion math.

    Note: an ID's score depends only on its rank position in each list, not
    on any other property of the result (no row data is touched here) --
    this is intentional; see module docstring.
    """
    scores: dict[int, float] = {}
    for id_list in ranked_id_lists:
        for rank, doc_id in enumerate(id_list):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


def rank_by_fused_score(scores: dict[int, float]) -> list[int]:
    """Sort IDs by fused score, best first.

    Ties are broken by ID ascending, purely so sort order is deterministic
    and tests can assert an exact list rather than "any order among ties."
    This is a stand-in tie-break, not a meaningful ranking signal — IDs are
    assigned at insert time and have no relationship to result quality.
    """
    return sorted(scores.keys(), key=lambda doc_id: (-scores[doc_id], doc_id))