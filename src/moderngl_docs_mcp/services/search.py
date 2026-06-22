"""Search service — orchestrates FTS5 + vector search through RRF fusion."""
from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass

import sqlite_vec

from moderngl_docs_mcp.retrieval.fusion import fuse_rankings, rank_by_fused_score
from moderngl_docs_mcp.retrieval.query import build_match_expression

Embedder = Callable[[str], list[float]]

_CANDIDATES_PER_SIGNAL = 20


@dataclass(frozen=True)
class SearchHit:
    """One ranked, fully-populated search result."""

    id: int
    title: str
    content_text: str
    doc_type: str
    score: float


def _fts_candidate_ids(
    conn: sqlite3.Connection, query: str, limit: int, doc_type: str | None = None
) -> list[int]:
    """Return section ids ranked by BM25, best first. Empty list on no matches
    or on a query with no escapable tokens (build_match_expression('') -> '""',
    which is valid FTS5 syntax that matches nothing, not an error).

    doc_type, when given, restricts candidates to that type BEFORE ranking
    (not as a post-hoc filter on the results) -- this matters because
    filtering after the fact would waste the `limit` candidate budget on
    rows that get thrown away, weakening recall for no reason. See
    docs/benchmarks/BENCHMARKS.md "Same API, two doc_types" for the
    motivating case: a "code" row and an "info" row sharing a title used to
    silently compete against each other in fused ranking.
    """
    match_expr = build_match_expression(query)
    if doc_type is not None:
        rows = conn.execute(
            """
            SELECT s.id
            FROM sections_fts f
            JOIN sections s ON f.rowid = s.id
            WHERE sections_fts MATCH ? AND s.doc_type = ?
            ORDER BY bm25(sections_fts, 10.0, 1.0)
            LIMIT ?
            """,
            (match_expr, doc_type, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT s.id
            FROM sections_fts f
            JOIN sections s ON f.rowid = s.id
            WHERE sections_fts MATCH ?
            ORDER BY bm25(sections_fts, 10.0, 1.0)
            LIMIT ?
            """,
            (match_expr, limit),
        ).fetchall()
    return [row["id"] for row in rows]


def _vec_candidate_ids(
    conn: sqlite3.Connection,
    query: str,
    limit: int,
    embedder: Embedder,
    doc_type: str | None = None,
) -> list[int]:
    """Return section ids ranked by vector similarity, best first.

    doc_type, when given, restricts the k-nearest-neighbor search to that
    type via a join + WHERE on sections_vec's MATCH query (sqlite-vec
    supports auxiliary WHERE clauses alongside the embedding MATCH/k
    clause), same SQL-level-filtering rationale as _fts_candidate_ids.
    """
    query_vector = embedder(query)
    embedding_bytes = sqlite_vec.serialize_float32(query_vector)
    if doc_type is not None:
        rows = conn.execute(
            """
            SELECT v.section_id
            FROM sections_vec v
            JOIN sections s ON s.id = v.section_id
            WHERE v.embedding MATCH ? AND v.k = ? AND s.doc_type = ?
            """,
            (embedding_bytes, limit, doc_type),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT section_id
            FROM sections_vec
            WHERE embedding MATCH ? AND k = ?
            """,
            (embedding_bytes, limit),
        ).fetchall()
    return [row["section_id"] for row in rows]


def _fetch_rows_in_order(conn: sqlite3.Connection, ids_in_order: list[int]) -> dict[int, sqlite3.Row]:
    """Fetch section rows for the given ids, as a lookup dict (order is the caller's job).

    A single `WHERE id IN (...)` query, not one query per id -- this is the
    fix for a real (if minor at this corpus size) inefficiency in the
    original prototype, which fetched full row data as a side effect of
    each individual FTS/vec query rather than once for only the final,
    already-trimmed result set.
    """
    if not ids_in_order:
        return {}
    placeholders = ",".join("?" for _ in ids_in_order)
    rows = conn.execute(
        f"SELECT id, title, content_text, doc_type FROM sections WHERE id IN ({placeholders})",
        ids_in_order,
    ).fetchall()
    return {row["id"]: row for row in rows}


def search(
    conn: sqlite3.Connection,
    query: str,
    embedder: Embedder,
    limit: int = 5,
    doc_type: str | None = None,
) -> list[SearchHit]:
    """Hybrid search: FTS5 + vector, fused via RRF, top `limit` results.

    doc_type, when given (e.g. "code" or "info"), restricts results to that
    type. Default None searches across all doc_types -- in that mode, two
    rows sharing a title under different doc_types (a real case in this
    corpus: a "code" example and an "info" explanation for the same API)
    are still independent candidates that can fragment each other's fused
    score rather than being merged. doc_type filtering is the documented
    workaround a caller can opt into; it is not a fix to that fragmentation
    in the default cross-type search. See docs/benchmarks/BENCHMARKS.md.

    Returns an empty list for a query that matches nothing on either side
    (not an error) -- mirrors the original prototype's "no results" path,
    just without the string formatting baked in here. Formatting the
    MCP-facing response text is the server layer's job, not this service's.
    """
    fts_ids = _fts_candidate_ids(conn, query, _CANDIDATES_PER_SIGNAL, doc_type)
    vec_ids = _vec_candidate_ids(conn, query, _CANDIDATES_PER_SIGNAL, embedder, doc_type)

    scores = fuse_rankings(fts_ids, vec_ids)
    ranked_ids = rank_by_fused_score(scores)[:limit]

    rows_by_id = _fetch_rows_in_order(conn, ranked_ids)

    hits = []
    for doc_id in ranked_ids:
        row = rows_by_id.get(doc_id)
        if row is None:
            continue
        hits.append(
            SearchHit(
                id=doc_id,
                title=row["title"],
                content_text=row["content_text"],
                doc_type=row["doc_type"],
                score=scores[doc_id],
            )
        )
    return hits