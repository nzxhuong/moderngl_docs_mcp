"""Real stdio smoke test: spawns the actual server as a subprocess and talks
to it over the real MCP protocol, exactly as a real client (Claude Desktop,
Cursor, MCP Inspector) would.

Every other test in this project calls Python functions directly (in-process,
fast, no protocol framing). This is the one test that proves those functions
are actually wired together correctly behind real tool/resource decorators
and a real stdio transport -- the class of bug a unit test structurally
cannot catch (e.g. a typo in a @mcp.tool() name, a lifespan that doesn't
actually yield, wrong argument names in a tool signature).

Uses tests/_stdio_test_entrypoint.py rather than server.py's own __main__,
so the real SentenceTransformer model never needs to download -- the fake
embedder is the one deliberate substitution; everything else this test
exercises is the unmodified production server code.
"""
from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

import pytest
import sqlite_vec

from moderngl_docs_mcp.ingestion.ingest import ingest_dump
from moderngl_docs_mcp.storage.db import bootstrap_schema

SAMPLE_DUMP = """### Buffer
Source: https://moderngl.readthedocs.io/buffer

Buffers store vertex data on the GPU.
--------------------------------
### VertexArray
Source: https://moderngl.readthedocs.io/vertex-array

VertexArray objects bind buffers to shader attributes.
"""


def _fake_embedder(text: str) -> list[float]:
    return [float(len(text)), float(sum(ord(c) for c in text))] * 192


@pytest.fixture
def test_db_path(tmp_path: Path) -> Path:
    """A real on-disk (not :memory:) database -- the stdio subprocess needs
    a file path it can open itself; an in-memory DB cannot be shared across
    the process boundary between this test and the spawned server.
    """
    db_path = tmp_path / "test_moderngl_docs.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    bootstrap_schema(conn)
    ingest_dump(conn, SAMPLE_DUMP, doc_type="info", source_file="test.txt", embedder=_fake_embedder)
    conn.close()
    return db_path


@pytest.mark.asyncio
async def test_stdio_server_lists_search_docs_tool(test_db_path):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    entrypoint = str(Path(__file__).parent / "_stdio_test_entrypoint.py")
    env = dict(os.environ)
    env["MODERNGL_DOCS_TEST_DB_PATH"] = str(test_db_path)

    params = StdioServerParameters(command=sys.executable, args=[entrypoint], env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            tool_names = [t.name for t in tools.tools]
            assert "search_docs" in tool_names


@pytest.mark.asyncio
async def test_stdio_server_search_docs_returns_real_content(test_db_path):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    entrypoint = str(Path(__file__).parent / "_stdio_test_entrypoint.py")
    env = dict(os.environ)
    env["MODERNGL_DOCS_TEST_DB_PATH"] = str(test_db_path)

    params = StdioServerParameters(command=sys.executable, args=[entrypoint], env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("search_docs", {"query": "Buffer", "limit": 3})

            assert len(result.content) == 1
            text = result.content[0].text
            # Real ingested content must appear -- not a stub, not an error
            # string, not "No documentation found."
            assert "Buffer" in text
            assert "GPU" in text


@pytest.mark.asyncio
async def test_stdio_server_get_document_resource(test_db_path):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from pydantic import AnyUrl

    entrypoint = str(Path(__file__).parent / "_stdio_test_entrypoint.py")
    env = dict(os.environ)
    env["MODERNGL_DOCS_TEST_DB_PATH"] = str(test_db_path)

    params = StdioServerParameters(command=sys.executable, args=[entrypoint], env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            # Section id 1 is "Buffer" -- first row inserted by ingest_dump
            # against a freshly bootstrapped (empty) database.
            result = await session.read_resource(AnyUrl("docs://1"))
            assert len(result.contents) == 1
            text = result.contents[0].text
            assert "Buffer" in text
            assert "moderngl.readthedocs.io" in text
