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
    "project",
)

# Scoring annotations added by the search layer.
SCORE_FIELDS = (
    "_similarity",
    "_distance",
    "_text_score",
    "_rrf_score",
    "_rrf_norm",
    "_relevance_score",
    "_scope_boost",
    "_decay_score",
    "_decay_norm",
    "_final_score",
    "_graph_boost",
    "_graph_source",
    "_graph_weight",
)

SELECTABLE_FIELDS = frozenset(PUBLIC_FIELDS + SCORE_FIELDS
                              + ("_project", "deleted_at"))


def validate_fields(fields: list[str] | None) -> str | None:
    if fields is None:
        return None
    invalid = sorted(set(fields) - SELECTABLE_FIELDS)
    if invalid:
        return f"unknown field(s): {', '.join(invalid)}"
    return None

# Fields that are never shown unless explicitly requested.
INTERNAL_FIELDS = (
    "content_hash",
    "embedding",
    "deleted_at",
    "_project",
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
    returned; internal fields are dropped. The scope of a memory is always
    reported as `project` (the project name, or "global").
    """
    if fields is not None:
        projected = {}
        for field in fields:
            if field in ("project", "_project"):
                value = row.get("_project")
                if value is None:
                    value = row.get("scope")
                if value is not None:
                    projected[field] = value
            elif field in row:
                projected[field] = row[field]
        return projected
    out = {f: row[f] for f in PUBLIC_FIELDS if f in row}
    if "project" not in out:
        value = row.get("_project") or row.get("scope")
        if value is not None:
            out["project"] = value
    out.update({f: row[f] for f in SCORE_FIELDS if f in row})
    return out


def project_bootstrap(row: dict) -> dict:
    """Reduce a row to the fields a session bootstrap needs.

    The scope of a memory is exposed as `project` (the project name, or
    "global") so an agent can tell where a memory comes from even when it
    bootstraps several projects at once.
    """
    out = {f: row[f] for f in BOOTSTRAP_FIELDS if f in row}
    scope = row.get("_project")
    if scope is None:
        scope = row.get("scope")
    if scope is not None:
        out["project"] = scope
    return out


def project_full(row: dict) -> dict:
    """Full column view for `detail="full"`, minus internal bookkeeping.

    Internal fields (content hash, embedding, deletion marker) are never
    exposed, not even in the full bootstrap. The search score annotations
    (`_decay_score`, `_final_score`, ...) are dropped as well: they describe
    a ranking that the bootstrap does not use - the list is sorted by the
    last update, not by a score.
    The scope is reported as `project`, like everywhere else.
    """
    out = {k: v for k, v in row.items()
           if k not in INTERNAL_FIELDS and k not in SCORE_FIELDS}
    if "project" not in out:
        scope = row.get("_project") or row.get("scope")
        if scope is not None:
            out["project"] = scope
    return out


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
