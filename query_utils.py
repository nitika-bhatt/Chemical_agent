"""
Query intent detection — shared logic for turning raw user input (an
accession, a motif, or a raw sequence) into the query string the
Orchestrator expects. Kept in its own module (not inline in web_app.py)
so it can be unit-tested without importing Streamlit / triggering
st.set_page_config() and friends.
"""

import re

# Matches Orchestrator.parse_query's own accession pattern — the official
# UniProt format, covering both 6-char (P04637) and 10-char (A0A024RBG1) IDs.
_ACCESSION_RE = re.compile(
    r"^([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})$",
    re.IGNORECASE,
)
_AA_SET = set("ACDEFGHIKLMNPQRSTVWY")


def build_query(raw_input: str) -> str:
    """
    Build the orchestrator query string that matches the *actual* input
    type, instead of always wrapping everything as "Analyze sequence: ...".
    This keeps query_mode / query_id / output-folder naming correct for
    accessions and motifs, matching what the README documents.
    """
    cleaned = (raw_input or "").strip()
    bare = re.sub(r"\s+", "", cleaned).upper()

    if _ACCESSION_RE.match(bare):
        return f"Analyze {bare} for methylation"
    if len(bare) == 11 and all(c in _AA_SET for c in bare):
        return f"Analyze motif: {bare}"
    return f"Analyze sequence: {bare}"
