# tests/test_ingest.py
#
# This file tests code that touches a real database and calls an embedding
# model. The trick for both: don't use the REAL versions in tests.
#
#   - Database  -> use sqlite3.connect(":memory:") instead of a real .db
#                  file. It's a real, fully working SQLite database that
#                  only exists in RAM and disappears when the test ends.
#                  No real file is ever touched.
#
#   - Embedder  -> ingest_dump() takes `embedder` as a PARAMETER (look at
#                  its signature). That's deliberate: in production you'd
#                  pass the real SentenceTransformer.encode, but in tests
#                  you pass a tiny fake function instead. This is called
#                  a "fake" or "test double" -- it's not testing whether
#                  sentence-transformers works.

import sqlite3

import pytest
import sqlite_vec

from moderngl_docs_mcp.ingestion.ingest import ingest_dump
from moderngl_docs_mcp.storage.db import bootstrap_schema

# A fake dump with two sections, used by several tests below.
SAMPLE_DUMP = """### Buffer
Source: https://moderngl.readthedocs.io/buffer

Buffers store vertex data on the GPU.
--------------------------------
### VertexArray
Source: https://moderngl.readthedocs.io/vertex-array

VertexArray objects bind buffers to shader attributes.
"""


def fake_embedder(text: str) -> list[float]:
    """Stand-in for the real SentenceTransformer model.

    Returns a fixed-length list of zeros -- the actual numbers don't matter
    for ingestion tests, only that *something* of the right shape (384
    floats, matching the schema's FLOAT[384] column) gets stored.
    Instant, deterministic, no model download required.
    """
    return [0.0] * 384


@pytest.fixture
def db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    bootstrap_schema(conn)

    yield conn 

    conn.close() 


# ── Tests ────────────────────────────────────────────────────────────

def test_ingest_adds_all_sections_on_first_run(db):
    stats = ingest_dump(db, SAMPLE_DUMP, doc_type="info", source_file="test.txt", embedder=fake_embedder)

    assert stats.sections_seen == 2
    assert stats.sections_added == 2
    assert stats.sections_updated == 0
    assert stats.sections_skipped == 0

    row_count = db.execute("SELECT COUNT(*) FROM sections").fetchone()[0]
    assert row_count == 2


def test_ingest_is_idempotent_on_unchanged_dump(db):
    ingest_dump(db, SAMPLE_DUMP, doc_type="info", source_file="test.txt", embedder=fake_embedder)

    # ...then run the EXACT same dump again. This is the core promise of
    # the whole content_hash design: nothing should be re-added.
    stats = ingest_dump(db, SAMPLE_DUMP, doc_type="info", source_file="test.txt", embedder=fake_embedder)

    assert stats.sections_added == 0
    assert stats.sections_updated == 0
    assert stats.sections_skipped == 2

    # Most important check: still only 2 rows, not 4.
    row_count = db.execute("SELECT COUNT(*) FROM sections").fetchone()[0]
    assert row_count == 2


def test_ingest_updates_changed_section_without_duplicating(db):
    ingest_dump(db, SAMPLE_DUMP, doc_type="info", source_file="test.txt", embedder=fake_embedder)

    edited_dump = SAMPLE_DUMP.replace(
        "Buffers store vertex data on the GPU.",
        "Buffers store vertex data on the GPU. Updated description.",
    )
    stats = ingest_dump(db, edited_dump, doc_type="info", source_file="test.txt", embedder=fake_embedder)

    assert stats.sections_added == 0
    assert stats.sections_updated == 1   # only "Buffer" changed
    assert stats.sections_skipped == 1   # "VertexArray" did not change

    row_count = db.execute("SELECT COUNT(*) FROM sections").fetchone()[0]
    assert row_count == 2  # still 2, not 3 -- updated in place

    new_text = db.execute(
        "SELECT content_text FROM sections WHERE title = 'Buffer'"
    ).fetchone()[0]
    assert "Updated description" in new_text


def test_ingest_keeps_section_id_stable_across_update(db):
    # This protects the docs://{doc_id} resource: editing a section's
    # content must NOT change its id, or every saved/bookmarked doc link
    # silently breaks on the next ingestion run.
    ingest_dump(db, SAMPLE_DUMP, doc_type="info", source_file="test.txt", embedder=fake_embedder)
    original_id = db.execute(
        "SELECT id FROM sections WHERE title = 'Buffer'"
    ).fetchone()[0]

    edited_dump = SAMPLE_DUMP.replace("on the GPU.", "on the GPU, fast.")
    ingest_dump(db, edited_dump, doc_type="info", source_file="test.txt", embedder=fake_embedder)

    new_id = db.execute(
        "SELECT id FROM sections WHERE title = 'Buffer'"
    ).fetchone()[0]
    assert new_id == original_id


def test_ingest_writes_one_vec_row_per_section(db):
    # Every section in `sections` must have exactly one matching row in
    # sections_vec -- if ingestion ever inserts a section without an
    # embedding (or leaves a stale embedding behind after an update),
    # vector search would silently return wrong/missing results.
    ingest_dump(db, SAMPLE_DUMP, doc_type="info", source_file="test.txt", embedder=fake_embedder)

    sections_count = db.execute("SELECT COUNT(*) FROM sections").fetchone()[0]
    vec_count = db.execute("SELECT COUNT(*) FROM sections_vec").fetchone()[0]
    assert sections_count == vec_count == 2


def test_ingest_records_a_dump_run(db):
    ingest_dump(db, SAMPLE_DUMP, doc_type="code", source_file="my_dump.txt", embedder=fake_embedder)

    run = db.execute("SELECT * FROM dump_runs WHERE source_file = 'my_dump.txt'").fetchone()
    assert run is not None
    assert run["doc_type"] == "code"
    assert run["sections_added"] == 2
    assert run["status"] == "complete"


def test_ingest_same_title_different_doc_type_are_separate_sections(db):
    # (title, doc_type) together are the identity key -- the same title
    # under two different doc_types must NOT collide/overwrite each other.
    ingest_dump(db, SAMPLE_DUMP, doc_type="info", source_file="a.txt", embedder=fake_embedder)
    ingest_dump(db, SAMPLE_DUMP, doc_type="code", source_file="b.txt", embedder=fake_embedder)

    row_count = db.execute("SELECT COUNT(*) FROM sections").fetchone()[0]
    assert row_count == 4  # 2 sections x 2 doc_types, not deduped across types