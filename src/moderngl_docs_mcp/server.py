"""moderngl-docs-mcp server."""
from __future__ import annotations

import logging
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from collections.abc import Callable

from mcp.server.fastmcp import Context, FastMCP

from moderngl_docs_mcp.services.search import Embedder, SearchHit, search
from moderngl_docs_mcp.storage.db import default_db_path, get_readonly_connection

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger("moderngl_docs_mcp")

EMBEDDING_MODEL_NAME = "mixedbread-ai/mxbai-embed-xsmall-v1"


def _default_embedder_factory() -> Embedder:
    """Load the SentenceTransformer model and return an Embedder callable."""
    logger.info("Loading embedding model %s...", EMBEDDING_MODEL_NAME)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    def embed(text: str) -> list[float]:
        return model.encode(text).tolist()

    return embed


@dataclass
class AppContext:
    """Typed lifespan context for the MCP server."""
    db: object  
    embed: Embedder


def _make_app_lifespan(
    db_path_override=None,
    embedder_factory: Callable[[], Embedder] = _default_embedder_factory,
):
    @asynccontextmanager
    async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
        db_path = db_path_override if db_path_override is not None else default_db_path()
        if not db_path.exists():
            msg = (
                f"No database found at {db_path.resolve()}\n"
                f"Run ingestion first, e.g.:\n"
                f"  python -m moderngl_docs_mcp.ingestion.cli moderngl_code_docs.txt code "
                f"moderngl_info_docs.txt info"
            )
            logger.error(msg)
            print(msg, file=sys.stderr)
            raise SystemExit(1)

        embed = embedder_factory()

        logger.info("Opening database at %s...", db_path)
        conn = get_readonly_connection(db_path)

        try:
            yield AppContext(db=conn, embed=embed)
        finally:
            conn.close()

    return app_lifespan


def _format_hit(hit: SearchHit) -> str:
    return f"### {hit.title} [{hit.doc_type.upper()}]\n{hit.content_text}"


def create_server(db_path=None, embedder_factory: Callable[[], Embedder] = _default_embedder_factory) -> FastMCP:
    mcp = FastMCP("ModernGL_doc", lifespan=_make_app_lifespan(db_path, embedder_factory))

    @mcp.tool()
    def search_docs(
        query: str, limit: int = 5, doc_type: str | None = None, ctx: Context = None  # type: ignore[assignment]
    ) -> str:
        """Search ModernGL documentation for code examples and API reference info.

        Combines keyword (FTS5) and semantic search, fused via reciprocal rank.

        QUERY SYNTAX:
        - A bare multi-word query like "vertex array object setup" is an IMPLICIT
          AND of every term.
        - Prefer 2-4 essential keywords over a full sentence, e.g. "vertex array"
          rather than "how do I set up a vertex array object".
        - To broaden recall, join terms with "or" (any case), e.g.
          "vertex or array or buffer".
        - Semantic search handles natural-language phrasing well regardless of keywords.
        - If a query returns few/no results, retry with fewer, more generic terms.

        doc_type filter: 
        Pass doc_type="code" for runnable examples only, doc_type="info" for 
        explanations only, or omit it to search both.
        """
        app_ctx: AppContext = ctx.request_context.lifespan_context
        hits = search(conn=app_ctx.db, query=query, embedder=app_ctx.embed, limit=limit, doc_type=doc_type)
        if not hits:
            return "No documentation found."
        return "\n---\n".join(_format_hit(hit) for hit in hits)

    @mcp.resource("docs://index")
    def get_docs_index(ctx: Context = None) -> str:  # type: ignore[assignment]
        """Get a complete list of all available ModernGL documentation topics and their IDs."""
        app_ctx: AppContext = ctx.request_context.lifespan_context
        rows = app_ctx.db.execute(
            "SELECT id, title, doc_type FROM sections ORDER BY doc_type, title"
        ).fetchall()

        lines = ["# ModernGL Documentation Index\n"]
        for row in rows:
            lines.append(f"- [{row['id']}] {row['title']} ({row['doc_type'].upper()})")
        return "\n".join(lines)

    @mcp.resource("docs://{doc_id}")
    def get_document(doc_id: str, ctx: Context = None) -> str:  # type: ignore[assignment]
        """Fetch a specific ModernGL document by its exact ID."""
        app_ctx: AppContext = ctx.request_context.lifespan_context

        try:
            numeric_id = int(doc_id)
        except ValueError:
            return f"Error: {doc_id!r} is not a valid document ID (expected an integer)."

        row = app_ctx.db.execute(
            "SELECT title, content_text, source_url, doc_type FROM sections WHERE id = ?",
            (numeric_id,),
        ).fetchone()

        if row is None:
            return f"Error: Document ID {doc_id} not found."

        return f"### {row['title']} [{row['doc_type'].upper()}]\nSource: {row['source_url']}\n\n{row['content_text']}"

    @mcp.prompt("build_moderngl_app")
    def build_moderngl_app_prompt(objective: str) -> list:
        """A guided workflow for the LLM to research and write a ModernGL script."""
        return [
            {
                "role": "user",
                "content": f"""I want you to write a ModernGL Python script for the following objective:
{objective}

Please follow this strict workflow before you start coding:
1. Use `search_docs` with doc_type="code" to look up relevant implementation
   examples.
2. Use `search_docs` with doc_type="info" to find exact API definitions, flag
   meanings, or object property semantics.
3. If a doc_type-filtered search returns nothing useful, retry the same query
   without doc_type (search both types).
4. Write the complete, runnable Python code with comments explaining the OpenGL concepts.""",
            },
            {
                "role": "assistant",
                "content": (
                    "I understand. I will aggressively search the documentation first to "
                    "ensure I am using the correct ModernGL API patterns before writing any "
                    "code. I'll pass doc_type=\"code\" when I want a working snippet to adapt, "
                    "and doc_type=\"info\" when I need to understand what a flag or parameter "
                    "actually does. When searching, I'll use short, focused keyword phrases. "
                    "If a search returns few results, I'll retry with general terms or without "
                    "a doc_type filter."
                ),
            },
        ]

    return mcp


def main() -> None:
    server = create_server()
    server.run()


if __name__ == "__main__":
    main()