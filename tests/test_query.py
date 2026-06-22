from moderngl_docs_mcp.retrieval.query import (
    build_match_expression,
    fts5_escape,
)


def test_fts5_escape_multi_word_query():
    # A multi-word query should have each word wrapped in quotes, and the
    # whole thing should be a single MATCH token (no operators or grouping).
    input = "vertex buffer"
    expected = '"vertex" "buffer"'
    assert fts5_escape(input) == expected


def test_fts5_escape_double_quotes():
    # Internal double quotes are escaped by DOUBLING (per FTS5's own
    # quoting rule), and the whole token also gets an outer pair of quotes
    # from fts5_escape itself. So a 7-char token '"hello"' (2 literal
    # quote chars) becomes '""hello""' after doubling, then '"""hello"""'
    # once wrapped -- two layers, not one. Easy to undercount by hand.
    input = 'He said "hello" to the buffer'
    expected = '"He" "said" """hello""" "to" "the" "buffer"'
    assert fts5_escape(input) == expected


def test_fts5_escape_boolean_operators():
    # FTS5's three real boolean operators (AND, OR, NOT) -- plus characters
    # like * that are only special in certain positions -- all get escaped
    # to literal tokens here, since fts5_escape wraps every whitespace-
    # separated word with no exceptions.
    input = "vertex AND buffer-thing OR NOT *array"
    expected = '"vertex" "AND" "buffer-thing" "OR" "NOT" "*array"'
    assert fts5_escape(input) == expected


def test_fts5_escape_whitespace_only():
    input = "   "
    expected = '""'
    assert fts5_escape(input) == expected


def test_fts5_escape_duplicate_whitespace():
    input = " vertex   buffer  "
    expected = '"vertex" "buffer"'
    assert fts5_escape(input) == expected


def test_build_match_expression_or():
    input = "vertex OR array OR buffer"
    expected = '"vertex" OR "array" OR "buffer"'
    assert build_match_expression(input) == expected


def test_build_match_expression_mixed_or_and():
    # AND-broadening is NOT a supported feature (only OR splits a clause),
    # so "array AND buffer" is NOT kept as one clause -- it's escaped word
    # by word like anything else, with "AND" becoming a literal token.
    # This documents current scope, not a future promise.
    input = "vertex OR array AND buffer"
    expected = '"vertex" OR "array" "AND" "buffer"'
    assert build_match_expression(input) == expected


def test_build_match_expression_no_or():
    input = "vertex buffer"
    expected = '"vertex" "buffer"'
    assert build_match_expression(input) == expected


def test_build_match_expression_or_in_word_is_not_split():
    # "mirror" and "orbit" both contain the letters "or" mid-word -- the
    # split must require "or" as its OWN word (whitespace on both sides),
    # not just a substring match, or ordinary words would be silently
    # mangled.
    input = "mirror orbit reflection"
    expected = '"mirror" "orbit" "reflection"'
    assert build_match_expression(input) == expected


def test_build_match_expression_or_as_standalone_word_does_split():
    input = "color or shading"
    expected = '"color" OR "shading"'
    assert build_match_expression(input) == expected


def test_build_match_expression_empty():
    input = "   "
    expected = '""'
    assert build_match_expression(input) == expected


def test_build_match_expression_accepts_lowercase_or():
    # Product decision: lenient casing. Real FTS5's OR operator is
    # case-sensitive (uppercase only), but this broadening feature is a
    # convenience layer for end users, not a literal passthrough of FTS5
    # syntax -- so "or" / "Or" / "OR" all trigger the split here.
    input = "vertex or array or buffer"
    expected = '"vertex" OR "array" OR "buffer"'
    assert build_match_expression(input) == expected


def test_build_match_expression_accepts_mixed_case_or():
    input = "vertex Or array oR buffer"
    expected = '"vertex" OR "array" OR "buffer"'
    assert build_match_expression(input) == expected