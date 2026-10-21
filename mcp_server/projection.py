"""Row projection for tool responses.

Agents pay for every field they receive. The database rows carry internal
bookkeeping (content hashes, deletion markers, embeddings) that is noise for
the caller. This module decides what a tool shows by default and lets the
caller narrow the selection further.
"""

from __future__ import annotations

# Fields that describe a memory and are useful to an agent.
PUBLIC_FIELDS = (
    "id",
    "content",
    "mem_type",
    "scope",
    "entity",
    "importance",
    "metadata",
    "access_count",
    "superseded_by",
    "created_at",
    "updated_at",
)

# Scoring annotations added by the search layer.
SCORE_FIELDS = (
    "_similarity",
    "_distance",
    "_rrf_score",
    "_rrf_norm",
    "_decay_score",
    "_decay_norm",
    "_final_score",
    "_graph_boost",
    "_graph_source",
    "_graph_weight",
)

# Fields that are never shown unless explicitly requested.
INTERNAL_FIELDS = (
    "content_hash",
    "embedding",
    "deleted_at",
)

# Even slimmer view for the session bootstrap.
BOOTSTRAP_FIELDS = (
    "id",
    "content",
    "entity",
    "mem_type",
    "scope",
    "importance",
    "updated_at",
)


def project(row: dict, fields: list[str] | None = None) -> dict:
    """Reduce a database row to what the caller should see.

    With `fields` given, only those keys are returned (missing keys are
    skipped). Without it, the public fields plus score annotations are
    returned; internal fields are dropped.
    """
    if fields is not None:
        return {f: row[f] for f in fields if f in row}
    keep = PUBLIC_FIELDS + SCORE_FIELDS
    return {f: row[f] for f in keep if f in row}


def project_bootstrap(row: dict) -> dict:
    """Reduce a row to the fields a session bootstrap needs."""
    return {f: row[f] for f in BOOTSTRAP_FIELDS if f in row}


def _stem(word: str) -> str:
    """Crude plural stemming so 'monday' and 'mondays' count as the same."""
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _tokens(content: str) -> set[str]:
    text = (content or "").lower()
    chars = [c if c.isalnum() or c.isspace() else " " for c in text]
    return {_stem(w) for w in "".join(chars).split() if w}


def normalize_content(content: str) -> str:
    """Key for near-duplicate detection: case, punctuation and plurals
    insensitive."""
    return " ".join(sorted(_tokens(content)))


def _similarity(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def dedupe_near_duplicates(rows: list[dict], threshold: float = 0.9) -> list[dict]:
    """Drop rows that say the same thing in slightly different words.

    Keeps the first occurrence, so ordering (by score or recency) decides
    which variant survives.
    """
    kept: list[dict] = []
    kept_tokens: list[set[str]] = []
    for row in rows:
        tokens = _tokens(str(row.get("content", "")))
        if not tokens:
            kept.append(row)
            kept_tokens.append(tokens)
            continue
        if any(_similarity(tokens, other) >= threshold for other in kept_tokens
               if other):
            continue
        kept.append(row)
        kept_tokens.append(tokens)
    return kept
