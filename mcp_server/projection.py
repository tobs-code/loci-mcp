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
    "confidence",
    "source",
    "agent_id",
    "tags",
    "metadata",
    "access_count",
    "superseded_by",
    "valid_from",
    "valid_to",
    "created_at",
    "updated_at",
    "expires_at",
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
    "_rerank_score",
    "_scope_boost",
    "_decay_score",
    "_decay_norm",
    "_final_score",
    "_graph_boost",
    "_graph_source",
    "_graph_weight",
    "_graph_relation_context",
    "_exact_match",
    "_history_penalty",
)

SELECTABLE_FIELDS = frozenset(PUBLIC_FIELDS + SCORE_FIELDS
                              + ("_project", "deleted_at", "forgotten",
                                 "expired", "expires_at"))


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
