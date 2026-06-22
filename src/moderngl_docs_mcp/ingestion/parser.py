"""Parse the context7-style dump format into structured sections.

Pure logic — no sqlite, no embedding model, no filesystem I/O beyond the
text already being in memory. This separation is what lets the parser be
unit-tested with in-memory strings instead of a real ingestion run.

Dump format (one or more chunks separated by a literal line of 32 dashes):

    ### Section Title
    Source: https://example.com/page

    body text, possibly multiple paragraphs, possibly containing
    ```python
    code fences
    ```
    --------------------------------
    ### Next Section Title
    ...
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

_CHUNK_SEP = "--------------------------------"
_TITLE_RE = re.compile(r"^###\s+(.*)", re.MULTILINE)
_SOURCE_RE = re.compile(r"^Source:\s+(\S+)", re.MULTILINE)
_TITLE_LINE_RE = re.compile(r"^###\s+.*$", re.MULTILINE)
_SOURCE_LINE_RE = re.compile(r"^Source:\s+\S+.*$", re.MULTILINE)


@dataclass(frozen=True)
class ParsedSection:
    """One parsed section from a dump file, before embedding/persistence."""

    title: str
    source_url: str
    content_text: str
    content_hash: str


def content_hash(title: str, content_text: str, source_url: str) -> str:
    """Stable hash over the fields that define "this section changed".

    Used by the ingester to decide insert vs. no-op vs. update without
    comparing full text in application code. Excludes doc_type, since the
    same (title, content) ingested under a different doc_type is a
    conflicting input the ingester should reject, not silently re-hash past.
    """
    payload = f"{title}\x00{content_text}\x00{source_url}".encode()
    return hashlib.sha256(payload).hexdigest()


def parse_dump(raw_text: str) -> list[ParsedSection]:
    """Parse a context7-style dump into a list of sections.

    Skips chunks with no usable title or empty body. Does not deduplicate
    (that is the ingester's job, since dedup requires DB state); a dump
    containing two chunks with the same title produces two ParsedSection
    entries here.
    """
    sections: list[ParsedSection] = []

    for raw_chunk in raw_text.split(_CHUNK_SEP):
        chunk = raw_chunk.strip()
        if not chunk:
            continue

        title_match = _TITLE_RE.search(chunk)
        title = title_match.group(1).strip() if title_match else ""

        source_match = _SOURCE_RE.search(chunk)
        source_url = source_match.group(1).strip() if source_match else ""

        body = _TITLE_LINE_RE.sub("", chunk)
        body = _SOURCE_LINE_RE.sub("", body).strip()

        if not title or not body:
            continue

        sections.append(
            ParsedSection(
                title=title,
                source_url=source_url,
                content_text=body,
                content_hash=content_hash(title, body, source_url),
            )
        )

    return sections


def semantic_title(title: str, content: str) -> str:
    """Build the text actually fed to the embedding model.

    Prepends the title to the first paragraph of the body so short,
    code-heavy sections (which embed poorly on their own) get useful
    semantic signal from their heading. Code fence markers are stripped
    since ``` adds no semantic content and can dominate short embeddings.
    """
    first_paragraph = content.strip().split("\n\n")[0]
    first_paragraph = first_paragraph.replace("```python", "").replace("```", "")
    first_paragraph = first_paragraph.replace("\n", " ").strip()
    return f"{title} - {first_paragraph}" if first_paragraph else title