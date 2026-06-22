# moderngl-docs-mcp

A hybrid (FTS5 keyword + vector semantic) MCP search server for [ModernGL](https://github.com/moderngl/moderngl) documentation (legaly take from context7 👀 , pls don't sue me). Combines SQLite FTS5 and `sqlite-vec` k-NN search, fused via Reciprocal Rank Fusion, behind a single `search_docs` MCP tool.

## Why hybrid search

A keyword index (FTS5/BM25) finds documents that share literal words with the query. A vector index (embedding similarity) finds documents that are *semantically* close, even with no shared vocabulary. Real documentation questions span both. This project runs both signals on every query and fuses their rankings.

## What you get

- `search_docs(query, limit=5, doc_type=None)` — hybrid search tool. `doc_type="code"` restricts to runnable examples, `doc_type="info"` restricts to conceptual/API explanations.
- `docs://index` — a resource listing every indexed section and its id.
- `docs://{doc_id}` — a resource fetching one section's full content by id.
- `build_moderngl_app` — a guided prompt that walks an agent through searching docs before writing code.

## Install

```bash
pip install -e .

```

This installs the `moderngl-docs-mcp` command (the MCP server itself) and the ingestion CLI as a Python module.

## First run: build the database

The server needs an ingested SQLite database before it can answer queries. Point the ingestion CLI at your dump file(s):

```bash
python -m moderngl_docs_mcp.ingestion.cli moderngl_code_docs.txt code moderngl_info_docs.txt info

```

This bootstraps `moderngl_docs.db` in the current directory, downloads the embedding model on first run (`mixedbread-ai/mxbai-embed-xsmall-v1`, ~30MB, cached locally afterward), and ingests both dump files.

## Run the server directly (for testing)

```bash
moderngl-docs-mcp

```

MCP servers communicate over stdin/stdout with a client. Use an MCP client or [MCP Inspector](https://github.com/modelcontextprotocol/inspector) for manual testing:

```bash
npx @modelcontextprotocol/inspector moderngl-docs-mcp

```

## Configure your MCP client

### VS Code

**Workspace-scoped**: create `.vscode/mcp.json` in the project root:

```json
{
  "servers": {
    "moderngl-docs": {
      "type": "stdio",
      "command": "moderngl-docs-mcp"
    }
  }
}

```

If `moderngl-docs-mcp` isn't on your system `PATH` (e.g. installed inside a virtualenv), use the full interpreter path instead:

```json
{
  "servers": {
    "moderngl-docs": {
      "type": "stdio",
      "command": "/full/path/to/your/venv/bin/moderngl-docs-mcp"
    }
  }
}

```

### Claude Desktop

Add to your Claude Desktop config file:

* **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
* **Linux:** `~/.config/Claude/claude_desktop_config.json`
* **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "moderngl-docs": {
      "command": "moderngl-docs-mcp"
    }
  }
}

```

### Cursor

Add to `.cursor/mcp.json` (project) or your global Cursor MCP settings:

```json
{
  "mcpServers": {
    "moderngl-docs": {
      "command": "moderngl-docs-mcp"
    }
  }
}

```

## Development

```bash
pip install -e .
pip install pytest pytest-asyncio sqlite-vec sentence-transformers pyyaml
pytest -v

```

### Run benchmarks

```bash
python scripts/run_benchmarks.py

```

Re-runs the labeled query set in `docs/benchmarks/queries.yaml` against your real, ingested database.

```bash
python scripts/run_benchmarks.py --diagnose "your query" --expected-title "Exact Section Title"

```

## Project structure

```
src/moderngl_docs_mcp/
├── ingestion/    parser.py + ingest.py + cli.py
├── retrieval/    query.py + fusion.py
├── services/     search.py
├── storage/      schema.sql + db.py
└── server.py     FastMCP tool/resource/prompt registration

docs/
├── architecture/ARCHITECTURE.md
├── benchmarks/BENCHMARKS.md
└── benchmarks/FUTURE-RERANKING.md

scripts/run_benchmarks.py
tests/

```

## License

MIT
