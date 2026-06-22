"""Test-only stdio entry point: runs the real server with a fake embedder.

Used exclusively by tests/test_server_stdio.py to spawn a real MCP server
subprocess over stdio without requiring network access to download the
real SentenceTransformer model. The point of this file existing at all is
that the actual protocol-framing, lifespan-startup, and tool-dispatch code
in server.py is exercised completely unmodified -- only the *embedder*
differs from production, the same single seam every other test in this
project has used (ingest_dump, search) to keep model loading optional.
"""
from __future__ import annotations

from pathlib import Path

from moderngl_docs_mcp.server import create_server


def _fake_embedder():
    def embed(text: str) -> list[float]:
        return [float(len(text)), float(sum(ord(c) for c in text))] * 192

    return embed


def main() -> None:
    import os

    db_path = Path(os.environ["MODERNGL_DOCS_TEST_DB_PATH"])
    server = create_server(db_path=db_path, embedder_factory=_fake_embedder)
    server.run()


if __name__ == "__main__":
    main()
