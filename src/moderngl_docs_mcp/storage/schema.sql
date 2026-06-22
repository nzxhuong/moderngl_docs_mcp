-- schema.sql — moderngl-docs-mcp storage schema
--
-- Design decisions:
--   - sections.content_hash (SHA256 of title+content_text+source_url) makes
--     ingestion idempotent: re-running ingest on the same dump is a no-op
--     for unchanged sections and a clean replace for changed ones, instead
--     of accumulating duplicate rows on every run.
--   - sections.dump_id + dump_runs table record provenance: which source
--     file, ingested when, how many sections. This is what makes "manual
--     text dump" a *documented, versioned* source rather than an untracked
--     one-off paste.
--   - FTS5 tokenizer: unicode61 remove_diacritics 2, tokenchars '._'
--     (matches python-docs-mcp-server's convention) so identifiers like
--     ctx.buffer or VertexArray.render index as single tokens. No Porter
--     stemming: exact API name matches should not get stemmed away.
--   - sections_vec uses sqlite-vec's vec0 virtual table, 384-dim float
--     vectors matching mixedbread-ai/mxbai-embed-xsmall-v1's output size.
--     section_id is NOT INTEGER PRIMARY KEY here on purpose (vec0 quirk:
--     it must reference sections.id but is not a real foreign key) — kept
--     as a plain rowid-aliased column for join clarity.

CREATE TABLE IF NOT EXISTS dump_runs (
    id              INTEGER PRIMARY KEY,
    source_file     TEXT NOT NULL,
    doc_type        TEXT NOT NULL,
    started_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    finished_at     TEXT,
    sections_seen   INTEGER NOT NULL DEFAULT 0,
    sections_added  INTEGER NOT NULL DEFAULT 0,
    sections_updated INTEGER NOT NULL DEFAULT 0,
    sections_skipped INTEGER NOT NULL DEFAULT 0,
    status          TEXT NOT NULL DEFAULT 'running'
);

CREATE TABLE IF NOT EXISTS sections (
    id              INTEGER PRIMARY KEY,
    title           TEXT NOT NULL,
    doc_type        TEXT NOT NULL,
    source_url      TEXT,
    content_text    TEXT NOT NULL,
    content_hash    TEXT NOT NULL,
    dump_id         INTEGER REFERENCES dump_runs(id) ON DELETE SET NULL,
    -- Stable identity key independent of row id: same (title, doc_type)
    -- pair is treated as "the same section" across re-ingestion, so
    -- updates replace in place instead of duplicating.
    UNIQUE(title, doc_type)
);

-- No separate index needed for (title, doc_type) lookups in _existing_hash:
-- the UNIQUE(title, doc_type) constraint above already creates one
-- automatically (SQLite always backs a UNIQUE constraint with an index).
-- idx_sections_doc_type covers doc_type-only lookups (e.g. filtering
-- search results or browsing docs://index by type) that don't include
-- title -- that access pattern can't use the UNIQUE(title, doc_type)
-- index, since title is the leading column there.
CREATE INDEX IF NOT EXISTS idx_sections_doc_type ON sections(doc_type);

CREATE VIRTUAL TABLE IF NOT EXISTS sections_fts USING fts5(
    title,
    content_text,
    content='sections',
    content_rowid='id',
    tokenize="unicode61 remove_diacritics 1 tokenchars '._'"
);

CREATE VIRTUAL TABLE IF NOT EXISTS sections_vec USING vec0(
    section_id INTEGER PRIMARY KEY,
    embedding FLOAT[384]
);