#!/usr/bin/env python3
"""Benchmark harness for moderngl-docs-mcp's hybrid search.

Runs every query in docs/benchmarks/queries.yaml through three retrieval
modes against the real, ingested database:

  fts_only    -- _fts_candidate_ids alone (no fusion, no vector signal)
  vector_only -- _vec_candidate_ids alone (no fusion, no keyword signal)
  hybrid      -- the real search() function (RRF fusion of both)

For each (query, mode) pair, records whether and where the labeled correct
answer appeared in the ranked results, plus wall-clock latency. Writes:

  - a per-query CSV (docs/benchmarks/results_raw.csv) for further analysis
  - a summary Markdown table to stdout, suitable for pasting into
    docs/benchmarks/BENCHMARKS.md

Usage:
    python scripts/run_benchmarks.py [--db PATH] [--queries PATH] [--k 1 3 5]

Requires a real, already-ingested database -- this harness does not ingest
anything itself. It also loads the REAL embedding model (not a test fake),
since benchmark numbers must reflect production behavior, not a stand-in.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from moderngl_docs_mcp.services.search import (  # noqa: E402
    _fts_candidate_ids,
    _vec_candidate_ids,
    rank_by_fused_score,
    search,
)
from moderngl_docs_mcp.retrieval.fusion import fuse_rankings  # noqa: E402
from moderngl_docs_mcp.storage.db import get_readonly_connection  # noqa: E402

_CANDIDATES = 20


@dataclass
class QueryResult:
    query: str
    category: str
    mode: str
    expected_title: str
    found_rank: int | None  # 1-indexed position of the expected doc, or None if absent from results
    latency_ms: float
    top_titles: list[str] = field(default_factory=list)


def _title_for_id(conn, doc_id: int) -> str:
    row = conn.execute("SELECT title FROM sections WHERE id = ?", (doc_id,)).fetchone()
    return row["title"] if row else f"<missing id {doc_id}>"


def _id_for_title(conn, title: str) -> int | None:
    row = conn.execute("SELECT id FROM sections WHERE title = ?", (title,)).fetchone()
    return row["id"] if row else None


def _rank_of(target_id: int | None, ranked_ids: list[int]) -> int | None:
    if target_id is None:
        return None
    try:
        return ranked_ids.index(target_id) + 1  # 1-indexed for human readability
    except ValueError:
        return None


def run_fts_only(conn, query: str, expected_id: int | None) -> tuple[int | None, float, list[int]]:
    start = time.perf_counter()
    ids = _fts_candidate_ids(conn, query, _CANDIDATES)
    latency_ms = (time.perf_counter() - start) * 1000
    return _rank_of(expected_id, ids), latency_ms, ids


def run_vector_only(conn, query: str, expected_id: int | None, embedder) -> tuple[int | None, float, list[int]]:
    start = time.perf_counter()
    ids = _vec_candidate_ids(conn, query, _CANDIDATES, embedder)
    latency_ms = (time.perf_counter() - start) * 1000
    return _rank_of(expected_id, ids), latency_ms, ids


def run_hybrid(conn, query: str, expected_id: int | None, embedder) -> tuple[int | None, float, list[int]]:
    start = time.perf_counter()
    hits = search(conn, query, embedder, limit=_CANDIDATES)
    latency_ms = (time.perf_counter() - start) * 1000
    ids = [h.id for h in hits]
    return _rank_of(expected_id, ids), latency_ms, ids


def load_queries(path: Path) -> list[dict]:
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must be a YAML list of query entries")
    return data


def run_benchmark(db_path: Path, queries_path: Path, embedder) -> list[QueryResult]:
    conn = get_readonly_connection(db_path)
    queries = load_queries(queries_path)
    results: list[QueryResult] = []

    for entry in queries:
        query = entry["query"]
        category = entry.get("category", "uncategorized")
        expected_title = entry["expected_title"]
        expected_id = _id_for_title(conn, expected_title)

        if expected_id is None:
            print(
                f"WARNING: expected_title {expected_title!r} for query {query!r} "
                f"not found in database -- this entry's results will show as misses "
                f"for every mode. Check docs/benchmarks/queries.yaml against your "
                f"real ingested data.",
                file=sys.stderr,
            )

        for mode, run_fn in (
            ("fts_only", run_fts_only),
            ("vector_only", run_vector_only),
            ("hybrid", run_hybrid),
        ):
            if mode == "fts_only":
                rank, latency_ms, ids = run_fn(conn, query, expected_id)
            else:
                rank, latency_ms, ids = run_fn(conn, query, expected_id, embedder)
            top_titles = [_title_for_id(conn, i) for i in ids[:5]]
            results.append(
                QueryResult(
                    query=query,
                    category=category,
                    mode=mode,
                    expected_title=expected_title,
                    found_rank=rank,
                    latency_ms=latency_ms,
                    top_titles=top_titles,
                )
            )

    conn.close()
    return results


def write_raw_csv(results: list[QueryResult], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query", "category", "mode", "expected_title", "found_rank", "latency_ms", "top_titles"])
        for r in results:
            writer.writerow(
                [r.query, r.category, r.mode, r.expected_title, r.found_rank, f"{r.latency_ms:.2f}", " | ".join(r.top_titles)]
            )


def recall_at_k(results: list[QueryResult], mode: str, k: int) -> float:
    mode_results = [r for r in results if r.mode == mode]
    if not mode_results:
        return 0.0
    hits = sum(1 for r in mode_results if r.found_rank is not None and r.found_rank <= k)
    return hits / len(mode_results)


def latency_stats(results: list[QueryResult], mode: str) -> dict[str, float]:
    values = sorted(r.latency_ms for r in results if r.mode == mode)
    if not values:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0}
    p95_index = min(len(values) - 1, int(len(values) * 0.95))
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p95": values[p95_index],
    }


def print_summary(results: list[QueryResult], ks: list[int]) -> None:
    modes = ["fts_only", "vector_only", "hybrid"]

    print("\n## Recall@k\n")
    header = "| Mode | " + " | ".join(f"Recall@{k}" for k in ks) + " |"
    sep = "|---|" + "---|" * len(ks)
    print(header)
    print(sep)
    for mode in modes:
        cells = " | ".join(f"{recall_at_k(results, mode, k):.0%}" for k in ks)
        print(f"| {mode} | {cells} |")

    print("\n## Latency (ms)\n")
    print("| Mode | Mean | Median | p95 |")
    print("|---|---|---|---|")
    for mode in modes:
        stats = latency_stats(results, mode)
        print(f"| {mode} | {stats['mean']:.2f} | {stats['median']:.2f} | {stats['p95']:.2f} |")

    print("\n## Per-query breakdown\n")
    print("| Query | Category | Mode | Found rank | Top hit |")
    print("|---|---|---|---|---|")
    for r in results:
        found = str(r.found_rank) if r.found_rank is not None else "MISS"
        top = r.top_titles[0] if r.top_titles else "(none)"
        print(f"| {r.query} | {r.category} | {r.mode} | {found} | {top} |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("moderngl_docs.db"))
    parser.add_argument("--queries", type=Path, default=Path("docs/benchmarks/queries.yaml"))
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument(
        "--out-csv", type=Path, default=Path("docs/benchmarks/results_raw.csv")
    )
    args = parser.parse_args()

    if not args.db.exists():
        print(f"No database found at {args.db}. Run ingestion first.", file=sys.stderr)
        sys.exit(1)
    if not args.queries.exists():
        print(f"No query file found at {args.queries}.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading embedding model (this may take a moment)...", file=sys.stderr)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("mixedbread-ai/mxbai-embed-xsmall-v1")

    def embedder(text: str) -> list[float]:
        return model.encode(text).tolist()

    results = run_benchmark(args.db, args.queries, embedder)
    write_raw_csv(results, args.out_csv)
    print(f"Raw results written to {args.out_csv}", file=sys.stderr)
    print_summary(results, args.k)


if __name__ == "__main__":
    main()