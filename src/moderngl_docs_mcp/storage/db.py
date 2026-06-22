"""SQLite connection management and schema bootstrap."""
from __future__ import annotations

import importlib.resources
import sqlite3
from pathlib import Path

import sqlite_vec


def default_db_path() -> Path:
    """Default database location, relative to the current working directory.

    Kept simple (no platformdirs) since this is a single-corpus, single-user
    project rather than a multi-corpus installed package — but isolated in
    one function so that decision is easy to revisit.
    """
    return Path("moderngl_docs.db")


def _load_vec_extension(conn: sqlite3.Connection) -> None:
    """Load sqlite-vec for this connection only, then lock extension loading back down.

    Extension loading must be re-enabled/disabled per-connection; it is not
    a global SQLite setting. Leaving it enabled after use is an unnecessary
    attack surface (arbitrary .so/.dll loading) for a server that has no
    other reason to load extensions at runtime.
    """
    conn.enable_load_extension(True)
    try:
        sqlite_vec.load(conn)
    finally:
        conn.enable_load_extension(False)


def _set_common_pragmas(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA foreign_keys = ON")


def get_readwrite_connection(path: str | Path) -> sqlite3.Connection:
    """Open a read-write connection for ingestion."""
    path = Path(path)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    _load_vec_extension(conn)
    conn.execute("PRAGMA journal_mode = WAL")
    _set_common_pragmas(conn)
    conn.row_factory = sqlite3.Row
    return conn


def get_readonly_connection(path: str | Path) -> sqlite3.Connection:
    """Open a connection for serving.

    Not opened in true SQLite ?mode=ro because sqlite-vec's extension load
    step itself does a write-capable handshake on some platforms; the
    application layer (services) treats this connection as read-only by
    convention/contract instead. This mirrors a real constraint you'll want
    to call out explicitly in the architecture doc rather than paper over.
    """
    path = Path(path)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    _load_vec_extension(conn)
    _set_common_pragmas(conn)
    conn.row_factory = sqlite3.Row
    return conn


def bootstrap_schema(conn: sqlite3.Connection) -> None:
    """Create all tables and indexes from schema.sql (idempotent).

    FTS5 and vec0 virtual tables are dropped and recreated every time, same
    rationale as python-docs-mcp-server: there is no ALTER for a virtual
    table's tokenizer/dimension config, so DROP + recreate is the only way
    to guarantee the schema on disk matches schema.sql. This is safe because
    both are derived data — sections_fts rebuilds from `sections` via the
    FTS5 'rebuild' command, and sections_vec is rebuilt by re-running
    ingestion (the canonical source is the dump file, not the vec table).
    """
    _VIRTUAL_TABLES = ("sections_fts", "sections_vec")
    for table in _VIRTUAL_TABLES:
        assert table.isidentifier(), f"Invalid table name: {table}"
        conn.execute(f"DROP TABLE IF EXISTS {table}")

    ref = importlib.resources.files("moderngl_docs_mcp.storage") / "schema.sql"
    with importlib.resources.as_file(ref) as schema_path:
        schema_sql = schema_path.read_text()
    conn.executescript(schema_sql)