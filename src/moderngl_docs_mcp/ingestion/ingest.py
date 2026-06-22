"""Idempotent ingestion of dump files into the sections/FTS/vec tables.

Orchestration only — text parsing lives in parser.py (pure, testable without
a DB), embedding generation is injected as a callable (testable without
loading a real model). This module's job is solely: given parsed sections
and an embedder, make the database match the dump, exactly once per
content_hash, regardless of how many times this function runs.
"""
from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass

import sqlite_vec

from moderngl_docs_mcp.ingestion.parser import ParsedSection, parse_dump, semantic_title

logger = logging.getLogger(__name__)

Embedder = Callable[[str], list[float]]


@dataclass
class IngestStats:
    """Counts returned by ingest_dump, also written to dump_runs."""

    sections_seen: int = 0
    sections_added: int = 0
    sections_updated: int = 0
    sections_skipped: int = 0


def _existing_hash(conn: sqlite3.Connection, title: str, doc_type: str) -> str | None:
    row = conn.execute(
        "SELECT content_hash FROM sections WHERE title = ? AND doc_type = ?",
        (title, doc_type),
    ).fetchone()
    return row["content_hash"] if row is not None else None


def _upsert_section(
    conn: sqlite3.Connection,
    section: ParsedSection,
    doc_type: str,
    dump_id: int,
    embedder: Embedder,
) -> int:
    """Insert or replace one section plus its embedding. Returns the section id.

    Replace-in-place (not insert-new-row) keeps the section's id stable
    across re-ingestion when only content changes, which matters because
    `docs://{doc_id}` resource URIs should not silently break on re-ingest.
    """
    cursor = conn.execute(
        """
        INSERT INTO sections (title, doc_type, source_url, content_text, content_hash, dump_id)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(title, doc_type) DO UPDATE SET
            source_url = excluded.source_url,
            content_text = excluded.content_text,
            content_hash = excluded.content_hash,
            dump_id = excluded.dump_id
        RETURNING id
        """,
        (
            section.title,
            doc_type,
            section.source_url,
            section.content_text,
            section.content_hash,
            dump_id,
        ),
    )
    section_id = cursor.fetchone()["id"]

    embedding = embedder(semantic_title(section.title, section.content_text))
    conn.execute("DELETE FROM sections_vec WHERE section_id = ?", (section_id,))
    conn.execute(
        "INSERT INTO sections_vec (section_id, embedding) VALUES (?, ?)",
        (section_id, sqlite_vec.serialize_float32(embedding)),
    )
    return section_id


def ingest_dump(
    conn: sqlite3.Connection,
    raw_text: str,
    doc_type: str,
    source_file: str,
    embedder: Embedder,
) -> IngestStats:
    """Ingest one dump's text into the database, idempotently.

    For each parsed section:
      - unseen (title, doc_type) -> insert + embed (added)
      - seen with identical content_hash -> no-op, no re-embed (skipped)
      - seen with different content_hash -> replace + re-embed (updated)

    Re-running this function with an unchanged dump file is a true no-op
    for sections_fts and sections_vec — no duplicate rows, no wasted
    embedding calls. The FTS index is rebuilt once at the end of the run
    (rebuild is a single bulk operation; doing it per-row would be wasteful
    and FTS5's external-content rebuild command does not support partial
    rebuilds).
    """
    sections = parse_dump(raw_text)
    stats = IngestStats(sections_seen=len(sections))

    cursor = conn.execute(
        "INSERT INTO dump_runs (source_file, doc_type) VALUES (?, ?)",
        (source_file, doc_type),
    )
    dump_id = cursor.lastrowid
    assert dump_id is not None

    for section in sections:
        existing_hash = _existing_hash(conn, section.title, doc_type)

        if existing_hash == section.content_hash:
            stats.sections_skipped += 1
            continue

        _upsert_section(conn, section, doc_type, dump_id, embedder)
        if existing_hash is None:
            stats.sections_added += 1
        else:
            stats.sections_updated += 1

    conn.execute("INSERT INTO sections_fts(sections_fts) VALUES('rebuild')")
    conn.execute(
        """
        UPDATE dump_runs SET
            finished_at = CURRENT_TIMESTAMP,
            sections_seen = ?,
            sections_added = ?,
            sections_updated = ?,
            sections_skipped = ?,
            status = 'complete'
        WHERE id = ?
        """,
        (
            stats.sections_seen,
            stats.sections_added,
            stats.sections_updated,
            stats.sections_skipped,
            dump_id,
        ),
    )
    conn.commit()

    logger.info(
        "Ingested %s: %d seen, %d added, %d updated, %d skipped",
        source_file,
        stats.sections_seen,
        stats.sections_added,
        stats.sections_updated,
        stats.sections_skipped,
    )
    return stats