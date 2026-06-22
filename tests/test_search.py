import sqlite3

import pytest
import sqlite_vec

from moderngl_docs_mcp.ingestion.ingest import ingest_dump
from moderngl_docs_mcp.services.search import search
from moderngl_docs_mcp.storage.db import bootstrap_schema

SAMPLE_DUMP = """### Buffer
Source: https://moderngl.readthedocs.io/buffer

Buffers store vertex data on the GPU.
--------------------------------
### Fake Buffer
Source: https://moderngl.readthedocs.io/not-buffer

fake info.
--------------------------------

### VertexArray Buffer
Source: https://moderngl.readthedocs.io/really-not-buffer

fake info.
--------------------------------

### VertexArray
Source: https://moderngl.readthedocs.io/vertex-array

VertexArray objects bind buffers to shader attributes.
"""

def fake_embedder(text: str) -> list[float]:
    return [float(len(text)), float(sum(ord(c) for c in text))] * 192

@pytest.fixture
def populated_db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    bootstrap_schema(conn)

    ingest_dump(conn, SAMPLE_DUMP, doc_type="info", source_file="test.txt", embedder=fake_embedder)

    yield conn
    conn.close()

def test_search_returns_results_after_ingestion(populated_db):
    results = search(conn=populated_db, query="Buffer", embedder=fake_embedder)
    assert isinstance(results, list)
    assert len(results) > 0
    hit = results[0]
    assert hasattr(hit, "id")
    assert hasattr(hit, "title")
    assert hasattr(hit, "content_text")
    assert hasattr(hit, "doc_type")
    assert hasattr(hit, "score")

def test_fts_side_finds_no_candidates_for_unmatched_term(populated_db):
    from moderngl_docs_mcp.services.search import _fts_candidate_ids

    ids = _fts_candidate_ids(populated_db, "NonExistentTerm", limit=20)
    assert ids == []


def test_fts_side_finds_no_candidates_for_empty_query(populated_db):
    from moderngl_docs_mcp.services.search import _fts_candidate_ids

    ids = _fts_candidate_ids(populated_db, "", limit=20)
    assert ids == []


def test_search_does_not_error_on_unmatched_or_empty_query(populated_db):
    no_match_results = search(conn=populated_db, query="NonExistentTerm", embedder=fake_embedder)
    empty_query_results = search(conn=populated_db, query="", embedder=fake_embedder)
    assert isinstance(no_match_results, list)
    assert isinstance(empty_query_results, list)

def test_search_limits_results(populated_db):
    results = search(conn=populated_db, query="Buffer", embedder=fake_embedder, limit=1)
    assert len(results) == 1

def test_search_results_are_sorted_by_score(populated_db):
    results = search(conn=populated_db, query="Buffer", embedder=fake_embedder)
    scores = [hit.score for hit in results]
    assert scores == sorted(scores, reverse=True)

def test_search_includes_docs_with_mixed_signal_strength(populated_db):
    results = search(conn=populated_db, query="Buffer", embedder=fake_embedder)
    titles = [hit.title for hit in results]
    assert "Buffer" in titles


# ── doc_type filtering ───────────────────────────────────────────────
#
# Real bug found via benchmarking (docs/benchmarks/BENCHMARKS.md): the same
# API documented as both a "code" example and an "info" explanation creates
# two separate sections sharing a title. In unfiltered hybrid search, RRF
# fusion treats them as independent candidates and can split/fragment the
# correct answer's score across both ids, letting a worse, undivided
# document outrank it. doc_type filtering lets a caller sidestep this by
# restricting the candidate pool to one type up front -- it does NOT fix
# the fragmentation in the default (doc_type=None) cross-type search; these
# tests cover only the filtering feature itself.

SAME_TITLE_DUMP_CODE = """### Context.copy_framebuffer()
Source: https://example.com/code

ctx.copy_framebuffer(dst, src)
"""

SAME_TITLE_DUMP_INFO = """### Context.copy_framebuffer()
Source: https://example.com/info

Copies framebuffer content from one framebuffer to another, useful for downsampling.
"""


@pytest.fixture
def db_with_duplicate_title_across_doc_types(populated_db):
    # Reuses the populated_db fixture's existing Buffer/VertexArray sections
    # as background noise, then adds the same-title-different-doc_type case.
    ingest_dump(
        populated_db, SAME_TITLE_DUMP_CODE, doc_type="code", source_file="dup_code.txt", embedder=fake_embedder
    )
    ingest_dump(
        populated_db, SAME_TITLE_DUMP_INFO, doc_type="info", source_file="dup_info.txt", embedder=fake_embedder
    )
    return populated_db


def test_doc_type_filter_excludes_other_types(db_with_duplicate_title_across_doc_types):
    db = db_with_duplicate_title_across_doc_types
    results = search(conn=db, query="copy_framebuffer", embedder=fake_embedder, doc_type="code", limit=10)
    doc_types_seen = {hit.doc_type for hit in results}
    assert doc_types_seen <= {"code"}
    assert "info" not in doc_types_seen


def test_doc_type_filter_finds_correct_type(db_with_duplicate_title_across_doc_types):
    db = db_with_duplicate_title_across_doc_types
    results = search(conn=db, query="copy_framebuffer", embedder=fake_embedder, doc_type="info", limit=10)
    titles = [hit.title for hit in results]
    assert "Context.copy_framebuffer()" in titles
    matching_hit = next(h for h in results if h.title == "Context.copy_framebuffer()")
    assert matching_hit.doc_type == "info"


def test_doc_type_none_searches_both_types(db_with_duplicate_title_across_doc_types):
    db = db_with_duplicate_title_across_doc_types
    results = search(conn=db, query="copy_framebuffer", embedder=fake_embedder, doc_type=None, limit=10)
    doc_types_seen = {hit.doc_type for hit in results}
    # Both the code and info rows are legitimate candidates when no filter
    # is applied -- this is the documented, not-yet-fixed fragmentation case.
    assert "code" in doc_types_seen
    assert "info" in doc_types_seen


def test_doc_type_filter_with_unknown_type_returns_empty(db_with_duplicate_title_across_doc_types):
    db = db_with_duplicate_title_across_doc_types
    results = search(conn=db, query="copy_framebuffer", embedder=fake_embedder, doc_type="nonexistent_type")
    assert results == []