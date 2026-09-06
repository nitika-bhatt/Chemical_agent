"""
Unit tests for query parsing — run with: pytest test_query_parsing.py -v

Covers the edge cases discussed while reviewing the pipeline: 10-char
accessions, the motif-vs-sequence keyword collision, lowercase input,
and bare amino-acid strings.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# ── orchestrator.parse_query ────────────────────────────────────────────────
# Imported lazily inside a fixture-less function so this file can still be
# collected even if agents/* aren't importable in a bare test environment —
# only Orchestrator's parse_query is a @staticmethod-like pure function, so
# we exercise it directly without instantiating the (heavy) full pipeline.

def _parse_query():
    from agents.orchestrator import Orchestrator
    # parse_query doesn't touch self.* state, but it's an instance method,
    # so build a minimal stand-in that only has the method bound.
    return Orchestrator.parse_query


def test_motif_keyword():
    parse_query = _parse_query()
    result = parse_query(None, "Analyze motif: FTLQIRGRERF")
    assert result["mode"] == "motif"
    assert result["sequence"] == "FTLQIRGRERF"


def test_sequence_keyword_full_protein():
    parse_query = _parse_query()
    result = parse_query(None, "Analyze sequence: MKRGGKPLRTAMKRGGKPLRTA")
    assert result["mode"] == "sequence"
    assert result["sequence"] == "MKRGGKPLRTAMKRGGKPLRTA"


def test_sample_index():
    parse_query = _parse_query()
    result = parse_query(None, "Analyze sample #125 from dataset")
    assert result["mode"] == "sample"
    assert result["sample_index"] == 125


def test_accession_6char():
    parse_query = _parse_query()
    result = parse_query(None, "Analyze P04637 for methylation")
    assert result["mode"] == "accession"
    assert result["accession"] == "P04637"


def test_accession_10char():
    """Regression test: this used to fail before the regex was widened."""
    parse_query = _parse_query()
    result = parse_query(None, "Analyze A0A024RBG1 for methylation")
    assert result["mode"] == "accession"
    assert result["accession"] == "A0A024RBG1"


def test_accession_embedded_in_sequence_prefix():
    """
    Regression test: web_app.py used to always prefix with 'Analyze
    sequence: ', which historically still worked for accessions only
    because of the unanchored regex search — verify that still holds.
    """
    parse_query = _parse_query()
    result = parse_query(None, "Analyze sequence: P04637")
    assert result["mode"] == "accession"
    assert result["accession"] == "P04637"


def test_motif_vs_sequence_keyword_collision():
    """
    An 11-letter motif wrapped as 'Analyze sequence: ...' (the old
    web_app.py behavior) gets classified as mode=sequence, not mode=motif.
    This is expected given the current regex priority (documents the
    behavior rather than asserting it's ideal) — build_query() in
    web_app.py is what actually avoids this by sending the right prefix.
    """
    parse_query = _parse_query()
    result = parse_query(None, "Analyze sequence: FTLQIRGRERF")
    assert result["mode"] == "sequence"  # documents current fallback behavior
    assert result["sequence"] == "FTLQIRGRERF"


def test_bare_amino_acid_string():
    parse_query = _parse_query()
    result = parse_query(None, "MKRGGKPLRTA")
    assert result["mode"] == "sequence"
    assert result["sequence"] == "MKRGGKPLRTA"


def test_lowercase_motif():
    parse_query = _parse_query()
    result = parse_query(None, "analyze motif: ftlqirgrerf")
    assert result["mode"] == "motif"
    assert result["sequence"] == "FTLQIRGRERF"


def test_unparseable_query_returns_error():
    parse_query = _parse_query()
    result = parse_query(None, "???")
    assert "error" in result


# ── web_app.build_query ─────────────────────────────────────────────────────

def _build_query():
    from query_utils import build_query
    return build_query


def test_build_query_accession_6char():
    build_query = _build_query()
    assert "P04637" in build_query("P04637")
    assert "for methylation" in build_query("P04637")


def test_build_query_accession_10char():
    build_query = _build_query()
    q = build_query("A0A024RBG1")
    assert "A0A024RBG1" in q
    assert "for methylation" in q


def test_build_query_motif():
    build_query = _build_query()
    q = build_query("FTLQIRGRERF")
    assert q.startswith("Analyze motif:")
    assert "FTLQIRGRERF" in q


def test_build_query_long_sequence():
    build_query = _build_query()
    q = build_query("MKRGGKPLRTAMKRGGKPLRTAMKRGGKPLRTA")
    assert q.startswith("Analyze sequence:")


def test_build_query_strips_whitespace():
    build_query = _build_query()
    q = build_query("  P04637  ")
    assert "P04637" in q
    assert "  " not in q
