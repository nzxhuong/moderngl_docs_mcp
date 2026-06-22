# tests/test_parser.py
#
# This is your FIRST test file. Read every comment once, then delete the
# comments in your own copy once it makes sense -- they're training wheels.
#
# pytest's only rule: any function in a file named test_*.py whose name
# starts with test_ gets run automatically. No registration, no config.

# This imports the real function from your real source code.
# If this line fails with "ModuleNotFoundError", it means the package
# isn't installed -- see the README section "Installing for development".
from moderngl_docs_mcp.ingestion.parser import (
    content_hash,
    parse_dump,
    semantic_title,
)


# ── Test 1: the simplest possible test ──────────────────────────────────
# "assert X" means: if X is False, this test FAILS and pytest tells you
# exactly which line and what the values were. If X is True, nothing
# happens -- the test just passes silently.

def test_content_hash_is_deterministic():
    # Calling the same function with the same inputs twice...
    h1 = content_hash("Buffer", "some text", "https://example.com")
    h2 = content_hash("Buffer", "some text", "https://example.com")
    # ...must produce the same hash. This is the property the whole
    # idempotent-ingestion design depends on -- if this test ever fails,
    # ingestion would silently start re-embedding everything on every run.
    assert h1 == h2


def test_content_hash_changes_when_content_changes():
    h1 = content_hash("Buffer", "version one", "https://example.com")
    h2 = content_hash("Buffer", "version two", "https://example.com")
    # Different body text -> different hash. This is what lets the
    # ingester detect "this section was edited" vs "nothing changed".
    assert h1 != h2


def test_content_hash_changes_when_title_changes():
    # Same body, different title -- also must produce a different hash,
    # since (title, doc_type) is the section's identity key in the DB.
    h1 = content_hash("Buffer", "same body", "https://example.com")
    h2 = content_hash("VertexArray", "same body", "https://example.com")
    assert h1 != h2


# ── Test 2: testing a function that returns a more complex value ───────
# parse_dump() returns a LIST of ParsedSection objects, so we test it by
# checking how many sections came out, and what's inside the first one.

def test_parse_dump_extracts_title_and_body():
    # This string is a tiny, fake version of your real dump file format.
    # Writing a small fake input by hand (instead of using a real 500-line
    # dump file) is itself a deliberate testing technique: it's called a
    # "fixture" or "minimal reproduction" -- the smallest input that still
    # exercises the behavior you're checking.
    raw = """### Buffer
Source: https://moderngl.readthedocs.io/buffer

This is the buffer documentation body.
"""
    sections = parse_dump(raw)

    # len(sections) == 1 means: exactly one section was found in our
    # fake dump. If parse_dump had a bug that produced 0 or 2, this
    # assert catches it immediately.
    assert len(sections) == 1

    section = sections[0]
    assert section.title == "Buffer"
    assert section.source_url == "https://moderngl.readthedocs.io/buffer"
    assert "buffer documentation body" in section.content_text


def test_parse_dump_handles_multiple_chunks():
    # The real separator between sections is a line of 32 dashes.
    raw = (
        "### First Section\n"
        "Source: https://example.com/a\n"
        "\n"
        "First body text.\n"
        "--------------------------------\n"
        "### Second Section\n"
        "Source: https://example.com/b\n"
        "\n"
        "Second body text.\n"
    )
    sections = parse_dump(raw)

    assert len(sections) == 2
    assert sections[0].title == "First Section"
    assert sections[1].title == "Second Section"


def test_parse_dump_skips_empty_chunks():
    # Two separators in a row produce an empty chunk in between --
    # parse_dump should silently skip it, not crash or return a junk entry.
    raw = (
        "### Real Section\n"
        "Source: https://example.com\n"
        "\n"
        "Real body.\n"
        "--------------------------------\n"
        "--------------------------------\n"
        "### Another Real Section\n"
        "Source: https://example.com\n"
        "\n"
        "Another body.\n"
    )
    sections = parse_dump(raw)
    assert len(sections) == 2


def test_parse_dump_skips_chunk_with_no_title():
    # A chunk with a body but no "### Title" line has no usable identity
    # (sections are keyed by title), so it should be dropped, not crash.
    raw = "Just some text with no title heading at all.\n"
    sections = parse_dump(raw)
    assert len(sections) == 0


# ── Test 3: a function with a "what should happen on weird input" angle ─
# This is the kind of test that catches real bugs -- not "does the happy
# path work" (you already eyeballed that) but "does an edge case break it".

def test_semantic_title_strips_code_fences():
    content = "```python\nctx.buffer(data)\n```\n\nMore explanation here."
    result = semantic_title("Buffer", content)
    # The ``` markers should be gone -- they add no semantic meaning for
    # the embedding model and would otherwise pollute every code example.
    assert "```" not in result
    assert "Buffer" in result


def test_semantic_title_falls_back_to_title_when_body_is_only_whitespace():
    # If the "first paragraph" is empty/whitespace, semantic_title should
    # not crash or return "Buffer - " with a dangling separator.
    result = semantic_title("Buffer", "   \n\n   ")
    assert result == "Buffer"


def test_source_url_stops_at_first_whitespace():
    # If the source line has extra text after the URL, that extra text
    # should not become part of the source_url field or the content hash.
    raw = (
        "### Section Title\n"
        "Source: https://example.com/section some extra text here\n"
        "\n"
        "Section body.\n"
    )
    sections = parse_dump(raw)
    assert len(sections) == 1
    assert sections[0].source_url == "https://example.com/section"