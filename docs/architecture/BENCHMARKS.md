# Benchmarks

This report measures `search()` — moderngl-docs-mcp's hybrid (FTS5 + vector,
fused via Reciprocal Rank Fusion) retrieval — against a hand-labeled query
set, comparing it to each individual signal in isolation.

**Reproduce this report:**
```bash
python scripts/run_benchmarks.py
```
Raw per-query results: [`results_raw.csv`](results_raw.csv).
Query set and labeling methodology: [`queries.yaml`](queries.yaml).

## Method

16 queries were hand-written against the real, ingested ModernGL
documentation corpus (~600 sections). For each query, a human (not a model)
inspected the corpus and recorded the one section title considered the
correct answer (`expected_title` in `queries.yaml`). Queries were
deliberately designed to include:

- **symbol** queries: exact API names as a user would type them
  (`Context.buffer()`)
- **concept** / **concept_paraphrase** queries: natural-language questions
  with little or no literal keyword overlap with the correct section's
  title or body (`downsampling a framebuffer` → `Context.copy_framebuffer()`)
- **troubleshooting** queries: a described symptom, not the underlying
  mechanism (`why are my samplers bleeding over to the next frame` →
  `Context.clear_samplers()`)
- **tricky_overlap** queries: phrased to share keywords with a *plausible
  but wrong* section (`change buffer size after creating it` → `Resize a
  Buffer`, not one of several other buffer-creation sections)

Each query was run three ways against the same database and embedding
model: `fts_only` (BM25 candidates alone), `vector_only` (k-NN candidates
alone), and `hybrid` (the real `search()`, RRF fusion of both). Recorded
per (query, mode): the rank of the expected section in the result list (or
a miss), and wall-clock latency.

## Results

### Recall@k

| Mode | Recall@1 | Recall@3 | Recall@5 |
|---|---|---|---|
| FTS-only | 1/16 (6%) | 2/16 (12%) | 3/16 (19%) |
| Vector-only | 7/16 (44%) | 10/16 (62%) | 13/16 (81%) |
| Hybrid | 6/16 (38%) | 11/16 (69%) | 12/16 (75%) |

### Latency

| Mode | Mean | Median | Max |
|---|---|---|---|
| FTS-only | 0.62 ms | 0.43 ms | 2.24 ms |
| Vector-only | 8.51 ms | 8.20 ms | 12.47 ms |
| Hybrid | 10.05 ms | 9.06 ms | 14.59 ms |

(One vector-only run recorded 2958 ms — almost certainly the embedding
model's first-call initialization cost inside that process, not steady-state
latency. Excluded from the table above as a measurement artifact rather than
a real result; the raw, unfiltered number is preserved in
[`results_raw.csv`](results_raw.csv).)

### Per-query breakdown

Rank shown where the expected section was found; **MISS** = not in the top
20 candidates for that mode.

| Query | Category | FTS | Vector | Hybrid |
|---|---|---|---|---|
| Context.memory_barrier() | symbol | 2 | 1 | 1 |
| Texture.build_mipmaps | symbol | 1 | 2 | 2 |
| how do I prevent z-fighting when drawing a wireframe | concept | MISS | 2 | 2 |
| using threads with python and opengl context | concept_paraphrase | MISS | 1 | 1 |
| downsampling a framebuffer | concept_paraphrase | MISS | 4 | 6 |
| running code on the GPU for arbitrary calculations without rendering to the screen | concept_paraphrase | MISS | MISS | MISS |
| what format string is used for 32-bit floats | concept | MISS | 2 | 2 |
| ModernGL crashes on exit when using auto gc with window library | troubleshooting | MISS | 1 | 1 |
| Context.buffer() | symbol | 4 | 4 | 3 |
| z-fighting fix | concept_paraphrase | MISS | 16 | 16 |
| how to do GPGPU and compute tasks | natural_language | MISS | 1 | 1 |
| change buffer size after creating it | tricky_overlap | MISS | 1 | 1 |
| multiple render targets MRT | concept | MISS | 1 | 2 |
| why are my samplers bleeding over to the next frame | troubleshooting | MISS | 5 | 5 |
| prevent rendering to the alpha channel | concept_paraphrase | MISS | MISS | MISS |
| how to run tests in github actions without a monitor | natural_language | MISS | 1 | 1 |

## Finding 1: FTS-only is nearly useless on this query mix, by design

FTS-only recall@5 is 19%. This is not a tuning failure — 13 of 16 queries
were deliberately written as natural-language questions or symptom
descriptions with no literal keyword match to the correct section, exactly
the gap a keyword index cannot close on its own. FTS wins decisively only on
the two queries that are themselves the API's exact name
(`Texture.build_mipmaps`, where the doc title literally is the query).

This corpus and query mix is not a fair fight for FTS-only — it was designed
to demonstrate where keyword search categorically fails, not to estimate FTS
quality on a more keyword-natural query distribution. A user typing exact
API names would see a very different number.

## Finding 2: hybrid does not strictly dominate vector-only here

This is the result worth not glossing over. Hybrid's recall@1 (38%) and
recall@5 (75%) are both **lower** than vector-only alone (44%, 81%). Only
recall@3 favors hybrid (69% vs 62%). A report that only stated "hybrid
search combines the best of both worlds" would be contradicted by this
project's own data.

## Finding 3: root cause — same-title sections across `doc_type` fragment RRF's score

Diagnosing the `downsampling a framebuffer` query (full candidate lists,
not just the printed top-5) found the underlying mechanism:

```
FTS candidates:    [548] Context.copy_framebuffer()   <- expected, rank 1
                    [308] Copy Framebuffer Content
                    [254] Copy Framebuffer to Texture (Blit)

Vector candidates: ... 20 candidates, including BOTH:
                    [189] Context.copy_framebuffer()   <- rank 4
                    [548] Context.copy_framebuffer()   <- rank 11 (same title, different row!)
```

`id 189` (`doc_type="code"`) and `id 548` (`doc_type="info"`) are two real,
legitimately distinct database rows — a runnable code example and a
conceptual explanation for the same API, ingested separately by design (the
schema's `UNIQUE(title, doc_type)` constraint explicitly allows this; see
`docs/architecture/`). They are not a data bug.

The bug is in fusion's blindness to that relationship. RRF sums
`1/(k+rank)` **per id**, and these are two different ids. The correct
answer's relevance signal is split between them, while a different,
genuinely-less-correct document (`Copy Framebuffer Content`, id 308) keeps
its full, undivided score from both signals — and can outrank the
fragmented pair. Computed directly from the real candidate lists above:

| Scenario | `Context.copy_framebuffer()` score | `Copy Framebuffer Content` score | Winner |
|---|---|---|---|
| As fragmented across ids 189+548 (real, current behavior) | 0.0310 (id 548 alone) | 0.0325 | **Wrong doc wins** |
| Counterfactual: same content merged into one id | 0.0468 | 0.0325 | **Correct doc wins** |

This is not a flaw in RRF as an algorithm — it's a missing assumption in how
this project applies it: RRF assumes each id is one independent answer, and
this corpus has cases where that assumption doesn't hold.

## Mitigation shipped: `doc_type` filter (scoped, not a full fix)

`search()` and the `search_docs` MCP tool now accept an optional `doc_type`
parameter. Passing `doc_type="info"` or `doc_type="code"` restricts both the
FTS and vector candidate queries to that type *before* ranking — so a
same-title pair across types cannot fragment within a single filtered call,
since only one of the two rows is ever a candidate.

**What this does not claim:** the recall numbers in this report were all
measured with `doc_type=None` (the default, cross-type search) — the same
mode every benchmark query above actually used. The filter was not applied
retroactively to regenerate "improved" numbers; doing so would conflate "a
caller who happens to know to use the filter" with "default behavior got
better," which would not be true. **Default, unfiltered hybrid search is
still subject to the fragmentation described in Finding 3.** The filter is
a real, tested (`tests/test_search.py`), opt-in mitigation for callers who
know in advance which doc_type they want — not a fix to the default
cross-type ranking.

A real fix to the default case — grouping same-title candidates before
fusion rather than after — is documented as a concrete next step below, not
yet implemented.

## Other observed failure modes

- **`z-fighting fix` → rank 16 (vector), rank 16 (hybrid).** The colloquial
  phrase "z-fighting fix" found `Context.polygon_offset` only at the very
  edge of the k=20 candidate window in both modes — much worse than the
  more fully-phrased `how do I prevent z-fighting when drawing a wireframe`
  (rank 2 in both). The embedding model appears sensitive to query length/
  context, not just topic — a 3-word colloquial query embeds less reliably
  near its answer than the same concept phrased as a full question.
- **Two complete misses in every mode**
  (`running code on the GPU for arbitrary calculations without rendering to
  the screen` → `ComputeShader`, and `prevent rendering to the alpha
  channel` → `Control Framebuffer Color Mask`). FTS found 0 candidates for
  both (expected — no keyword overlap). More notably, vector search also
  missed both within the k=20 budget despite the underlying section
  existing and other, more weakly-paraphrased queries succeeding (e.g.
  `how to do GPGPU and compute tasks` → rank 1 for the *same* `ComputeShader`
  family of content, just phrased differently). This suggests these two
  specific query phrasings sit unusually far from the correct section in
  embedding space — a reranking pass (see below) operating on a wider
  candidate set, or a larger `_CANDIDATES_PER_SIGNAL`, are the two most
  promising next experiments, not yet run.

## What this benchmark does not establish

- **Not a comparison against any other docs-MCP or retrieval system** —
  this measures this project's three internal modes against each other
  only.
- **Not a claim about FTS quality on keyword-natural queries** — see
  Finding 1; the query set was deliberately adversarial to keyword search.
- **Small sample (16 queries)** — individual query results (e.g. the single
  rank-16 case) carry disproportionate weight in the aggregate percentages.
  Treat the qualitative findings (the fragmentation mechanism, the
  colloquial-phrasing sensitivity) as more load-bearing than the precise
  recall percentages, which would likely shift with a larger query set.
- **Single embedding model, single k constant (k=60), single candidate
  width (20 per signal)** — no sweep over these was performed; the observed
  recall numbers are specific to this configuration, not necessarily
  representative of hybrid search's ceiling on this corpus.

## Recommended next steps

1. **Title-aware grouping before fusion** — merge candidates sharing a
   title (summing or max-ing their per-signal scores) before RRF, so the
   `doc_type` fragmentation in Finding 3 is fixed for the *default* search
   path, not only the opt-in filtered path. This is the highest-leverage
   fix identified by this benchmark and is not yet implemented.
2. **Cross-encoder reranking** — see
   [`FUTURE-RERANKING.md`](FUTURE-RERANKING.md) for a scoped writeup of
   what it would and would not address here, and why it's deferred rather
   than rejected.
3. **Re-run with a larger, less hand-curated query set** once a wider
   corpus and a less adversarially-designed query mix are available, to get
   a less sample-size-sensitive recall estimate.
