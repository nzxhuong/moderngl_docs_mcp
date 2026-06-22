# Architecture

This document records the design decisions behind `moderngl-docs-mcp`'s
retrieval architecture: what was chosen, what the alternatives were, and
what the actual measured tradeoffs are. Where a decision is backed by real
benchmark data rather than reasoning alone, this doc links to
[`docs/benchmarks/BENCHMARKS.md`](../benchmarks/BENCHMARKS.md) rather than
restating the numbers.

## 1. Layered design

```
ingestion/   parser.py (pure text parsing) + ingest.py (DB writes, embedding orchestration)
retrieval/   query.py (FTS5 escaping) + fusion.py (RRF scoring math)
services/    search.py (wires retrieval + storage + embedder together)
storage/     schema.sql + db.py (connection management, schema bootstrap)
server.py    FastMCP tool/resource/prompt registration, lifespan DI
```

### Decision: pure logic separated from I/O at every layer

`parser.py` and `fusion.py` take and return plain Python values (strings,
lists, dicts of floats) — no `sqlite3.Connection`, no embedding model, no
filesystem access. `ingest.py` and `services/search.py` do the I/O, but
accept the embedding model as an injected `Callable[[str], list[float]]`
(`Embedder`) rather than importing `sentence_transformers` directly.

**Why:** this is a testability decision, not a purity exercise. Every pure
function in this codebase is unit-tested with plain values and no fixtures;
every I/O function is tested against a real (if `:memory:` or `tmp_path`)
SQLite connection with a fake, instantly-computed embedder. The one
genuinely expensive resource in this project — `SentenceTransformer` model
loading — is never required by the test suite. `tests/test_server_stdio.py`
spawns the *real* server binary as a subprocess and exercises the *real*
MCP protocol path, and even that test substitutes a fake embedder via
`create_server(embedder_factory=...)`, the same seam used everywhere else.

**Alternative considered:** module-level globals for the DB connection and
model, as in the original prototype (`mcp_server.py`'s lazy `get_db()` /
module-level `model = SentenceTransformer(...)`). Rejected because it makes
every consumer implicitly depend on a real model and a real on-disk
database with no way to substitute either in a test, and because lazy
module-level state is harder to reason about than an explicit FastMCP
lifespan context passed through `ctx.request_context.lifespan_context`.

## 2. Hybrid retrieval: FTS5 + vector, fused via Reciprocal Rank Fusion

### Decision: combine two retrieval signals rather than picking one

`search()` runs a query through both SQLite FTS5 (`sections_fts`, BM25-
ranked) and `sqlite-vec` (`sections_vec`, k-nearest-neighbor cosine
similarity over `mxbai-embed-xsmall-v1` embeddings), then fuses the two
ranked candidate lists with Reciprocal Rank Fusion:

```
score(doc) = sum over each signal's ranked list containing doc of: 1 / (k + rank)
```

(`retrieval/fusion.py`, `k=60` — the conventional default from the
original RRF literature, Cormack et al. 2009, and what the original
prototype already used.)

**Why RRF specifically, not a weighted sum of raw scores:** BM25 and cosine
similarity are not on comparable scales — BM25 is an unbounded, corpus-
dependent score; cosine similarity is bounded to roughly [-1, 1]. A
weighted sum (`0.5 * bm25_score + 0.5 * cosine_score`) would require
normalizing both onto a shared scale, and that normalization is itself a
tuning decision with its own failure modes (e.g. min-max normalization is
sensitive to outliers in either ranked list). RRF sidesteps the
comparable-scale problem entirely by operating only on **rank position**,
not the underlying score magnitude — `1/(k+rank)` only needs to know
"this doc was 3rd-best in this list," not "this doc scored 14.2 in this
list." This is also why it composes cleanly: `fuse_rankings()` accepts any
number of ranked ID lists (`*ranked_id_lists`), not just exactly two,
should a third signal (e.g. a popularity or freshness ranking) be added
later.

**Measured tradeoff, not assumed:** this design was validated empirically,
and the empirical answer is more nuanced than "hybrid is strictly better."
[`BENCHMARKS.md`](../benchmarks/BENCHMARKS.md) found hybrid's recall@1
(38%) and recall@5 (75%) both *below* vector-only alone (44%, 81%) on a
16-query adversarial benchmark, with only recall@3 favoring hybrid. The
root cause was traced to a real, proven mechanism (see §4 below and
`BENCHMARKS.md` Finding 3) — not a flaw in RRF as an algorithm, but a
specific assumption violation in how this corpus's data interacts with it.
This is documented here because an architecture doc that only states "we
combined two signals for better recall" without the caveat would overclaim
relative to what was actually measured.

### Decision: candidate width wider than the result limit

`_CANDIDATES_PER_SIGNAL = 20` in `services/search.py` — each individual
signal contributes up to 20 candidates to fusion, even when the caller asks
for `limit=5` results. A document strong on one signal but absent from the
other's top-`limit` should still get a chance to be pulled up by fusion;
if each signal only contributed exactly `limit` candidates, a doc ranked
6th by FTS but 1st by vector search (with `limit=5`) would never enter the
fused pool at all. No formal sweep over this constant was performed; 20
was chosen as a reasonable multiple of typical `limit` values (5) and
matches the `k` parameter sqlite-vec's k-NN query already needs.

## 3. Idempotent ingestion via content hashing

### Decision: `content_hash` (not row id) determines insert vs. skip vs. update

`sections.content_hash` is `SHA256(title + content_text + source_url)`
(`ingestion/parser.py::content_hash`). `ingest_dump()` looks up the
existing hash for a `(title, doc_type)` pair before writing: identical hash
→ skip (no DB write, no re-embed); different hash → upsert in place,
preserving the row's `id`; new `(title, doc_type)` → insert.

**Why:** re-embedding is the expensive operation in this pipeline (a real
`SentenceTransformer.encode()` call per changed section); comparing two
64-character hash strings is effectively free. Re-running ingestion on an
unchanged dump file is therefore a true no-op for `sections_vec` and
`sections_fts` — not "re-insert and rely on `INSERT OR REPLACE` to dedupe,"
which would still pay the re-embedding cost on every run regardless of
whether content actually changed.

**Why id is preserved on update, not regenerated:** `docs://{doc_id}` is an
MCP resource URI keyed on `sections.id`. If editing a section's content
allocated a new id, every previously-saved or client-cached
`docs://{old_id}` reference would silently start 404ing after the next
ingestion run. The upsert uses `ON CONFLICT(title, doc_type) DO UPDATE`
specifically to keep the row (and its id) stable across content edits.

**Tested:** `tests/test_ingest.py` covers all three branches (insert,
no-op skip, in-place update) plus the id-stability guarantee directly, not
just as an incidental side effect of another test.

## 4. Known limitation: same-title sections across `doc_type` fragment fusion

This corpus contains, by design, both a `doc_type="code"` (runnable
example) and a `doc_type="info"` (conceptual explanation) section for the
same API, sharing a title (`schema.sql`'s `UNIQUE(title, doc_type)`
constraint explicitly allows this — it is not a data bug). Default,
unfiltered hybrid search treats these as two independent fusion
candidates. [`BENCHMARKS.md`](../benchmarks/BENCHMARKS.md) Finding 3 proves
(with real candidate-list arithmetic, not a guess) that this can fragment
the correct answer's RRF score across both ids, letting a different,
genuinely-less-correct document win.

**Mitigation shipped:** `search(doc_type=...)` restricts both the FTS and
vector candidate queries to one type *before* ranking (SQL-level filtering,
not post-hoc result filtering — see `services/search.py::_fts_candidate_ids`
/ `_vec_candidate_ids` docstrings for why SQL-level filtering specifically
was chosen over filtering the already-fused result list). This is a real,
tested (`tests/test_search.py`), opt-in fix **for callers who use it** —
not a fix to default (`doc_type=None`) ranking, which is what every query
in the benchmark report actually used. The `build_moderngl_app` MCP prompt
was updated to tell agents to use `doc_type="code"` / `doc_type="info"`
explicitly, with an instruction to retry unfiltered if a filtered search
comes up empty (`server.py`, `tests/test_prompt.py`).

**Not yet implemented:** grouping same-title candidates before fusion
(summing or max-ing their per-signal scores into one fused candidate),
which would fix the default cross-type path rather than requiring the
caller to opt into a type filter. Tracked as the top recommended next step
in `BENCHMARKS.md`.

## 5. Storage: SQLite + FTS5 + sqlite-vec, no separate vector database

**Why not a dedicated vector database** (e.g. Qdrant, Pinecone, Chroma):
this corpus is small (~600 sections) and the project's entire premise is a
local, dependency-light MCP server an end user runs alongside their editor
— a separate running database service is a meaningfully heavier
operational footprint for a marginal-at-this-scale performance gain.
`sqlite-vec`'s `vec0` virtual table gives approximate-enough k-NN search
inside the same file as the FTS5 index and the canonical `sections` table,
so there is exactly one file to back up, version, or delete to reset state.

**Tradeoff accepted:** `sqlite-vec` is a loadable SQLite extension, not a
built-in — every connection must call `enable_load_extension(True)` /
`sqlite_vec.load(conn)` / `enable_load_extension(False)` individually
(`storage/db.py::_load_vec_extension`), and the embedding dimension
(`FLOAT[384]`) is hardcoded in `schema.sql`, tied to the specific
embedding model in use. Swapping embedding models requires a schema change
and a full re-ingestion, not just a config flag — an explicit, accepted
constraint of this approach, not an oversight.

## 6. Deferred: cross-encoder reranking

See [`docs/benchmarks/FUTURE-RERANKING.md`](../benchmarks/FUTURE-RERANKING.md)
for a full writeup of what a reranking stage would and would not address
in this corpus's measured failure modes, and why it was designed but not
implemented. Summarized: it is a different and complementary mechanism to
RRF fusion (joint query-document scoring vs. independent embedding
comparison) and would plausibly help borderline ranking judgment calls, but
would **not** fix the §4 fragmentation issue, since reranking operates on
whatever candidate set fusion already produced — it does not merge or
deduplicate candidates.

## Source of truth

This document describes the architecture as implemented. For the empirical
evidence behind the retrieval-design claims in §2 and §4, see
[`docs/benchmarks/BENCHMARKS.md`](../benchmarks/BENCHMARKS.md) and its raw
data, [`docs/benchmarks/results_raw.csv`](../benchmarks/results_raw.csv).
For test coverage backing each decision, see the referenced test files
directly rather than trusting this document's claims about test coverage
at face value.
