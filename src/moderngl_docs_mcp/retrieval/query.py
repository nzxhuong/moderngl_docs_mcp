"""FTS5 query construction and escaping.

The original prototype passed `query` straight into MATCH, which means any
input containing FTS5 syntax (an unquoted AND/OR/NOT, a quote character,
a trailing *, a leading - before a colon, etc.) is interpreted according to
FTS5's query grammar rather than treated as literal text. That's a
correctness bug (surprising results on ordinary text like "ten-pin bowling
setup", where the hyphen sits in an unrelated position) as much as a
hygiene one.

This module makes the choice explicit instead of accidental: by default,
escape everything to literal-token matching (matches python-docs-mcp-server's
approach). A separate `build_match_expression` helper exists for the
documented "OR" broadening feature mentioned in the search_docs docstring,
so OR-joining is an intentional, parsed feature rather than raw passthrough.
"""
from __future__ import annotations

import re

# Matches a standalone "or" as a whole word, any casing (OR, or, Or, oR),
# surrounded by whitespace. \b ensures "orbit" or "color" don't match --
# only "or" as its own word does. This is deliberately MORE lenient than
# real FTS5 (which requires exact-case "OR"): the broadening feature here
# is a convenience for end users typing natural queries, not a literal
# passthrough of FTS5 operator syntax, so leniency is a product choice,
# not a correctness bug.
_OR_SPLIT_RE = re.compile(r"\s+or\s+", re.IGNORECASE)


def fts5_escape(query: str) -> str:
    """Escape user input for safe FTS5 MATCH queries.

    FTS5 has exactly three case-sensitive boolean operators: AND, OR, NOT
    (plus the NEAR(...) function). A bare, unquoted occurrence of one of
    these words is interpreted as an operator, not literal text -- e.g. a
    query containing the standalone word "OR" splits the query in two.

    Separately, a handful of *characters* (not words) are also syntactically
    special in specific positions: a trailing "*" marks a prefix query, a
    leading "-" before a column filter excludes that column, ":" separates
    a column filter from its term, and unescaped '"' delimits a quoted
    string. None of these are boolean operators -- they're each a distinct
    piece of syntax with its own narrow trigger condition -- but they are
    still not safe to pass through unescaped, since a user's literal text
    (e.g. a hyphenated word, or a trailing asterisk typed for emphasis)
    can accidentally trigger them.

    Wrapping every whitespace-separated token in double quotes neutralizes
    all of the above at once: per FTS5's own string-quoting rule, anything
    inside double quotes is treated as literal text, operators and special
    characters included. Internal double quotes in the input are escaped
    by doubling, also per FTS5's quoting rule.

    Empty/whitespace-only input returns '""' (matches nothing rather than
    raising or matching everything).
    """
    query = query.replace("\x00", "").strip()
    if not query:
        return '""'
    tokens = []
    for token in query.split():
        token = token.replace("\x00", "")
        if not token:
            continue
        safe = token.replace('"', '""')
        tokens.append(f'"{safe}"')
    return " ".join(tokens) if tokens else '""'


def build_match_expression(query: str) -> str:
    """Build a safe FTS5 MATCH expression, honoring a standalone "or" separator.

    This is the one piece of FTS5 syntax intentionally exposed to callers
    (matching the search_docs docstring's documented "vertex OR array OR
    buffer" broadening feature) — each OR-separated clause is escaped
    independently, then re-joined with a real (unescaped) FTS5 OR operator.
    Any other FTS5 syntax in the input (AND, NOT, *, quotes, parens) is
    escaped to a literal token, not interpreted.

    Splitting is case-insensitive ("or", "Or", "OR" all split a clause) —
    a deliberate leniency decision for end-user convenience, distinct from
    real FTS5's own case-sensitive OR operator. A future "AND" broadening
    feature would need the same explicit decision made for it; today only
    "or"-shaped tokens split a clause, "AND" stays literal within one.
    """
    query = query.strip()
    if not query:
        return '""'

    clauses = _OR_SPLIT_RE.split(query)
    escaped_clauses = [fts5_escape(clause) for clause in clauses if clause.strip()]
    if not escaped_clauses:
        return '""'
    return " OR ".join(escaped_clauses)