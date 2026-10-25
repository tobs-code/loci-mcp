import json
import logging
import math
import re
from dataclasses import dataclass
from typing import Any

from mcp_server import config
from mcp_server.config import (
    FTS_MIN_COVERAGE,
    FTS_WEIGHT,
    PROJECT_SCOPE_BOOST,
    RERANK_ALWAYS,
    RERANK_ENABLED,
    RERANK_LOGIT_BIAS,
    RERANK_TOP_K,
    RERANK_TRIGGER_MARGIN,
    RERANK_TRIGGER_SCORE,
    RRF_K,
)
from mcp_server.store import MemoryStore
from mcp_server.duplicates import is_cross_scope_duplicate
from mcp_server.filters import (
    feature_conditions,
    parse_iso_timestamp,
    validity_condition,
)
from mcp_server.graph_store import GraphStore
from mcp_server.inference_gateway import rerank
from mcp_server.scope_catalog import (
    _ag_str,
    _all_mem_tables,
    _project_names,
    _retrieval_tables,
    _scope_label,
)
from mcp_server.scoring import (
    _blend_scores,
    _clamp01,
    _decay_norm,
    _decay_score,
    _fts_query,
    _now,
    _relevance_score,
)
from mcp_server.lifecycle import (
    is_expired as _is_expired_value,
    is_visible as _is_current_row,
    visibility_sql as visibility_condition,
)

logger = logging.getLogger(__name__)


BEST_DROPPED_PREVIEW_CHARS = 160

# An outdated row stays visible with the history flags but must lose against
# the current fact that replaced it. 0.5 is strong enough to flip a near-tie
# without hiding an old fact that is the only match.
HISTORY_RELEVANCE_WEIGHT = 0.5


@dataclass(frozen=True)
class SearchRequest:
    """Named search intent, so an internal call never relies on argument order."""

    query_embedding: list[float] | None
    query_text: str | None
    mem_type: str | None = None
    entity: str | None = None
    limit: int = 10
    project: str | None = None
    include_decay: bool = False
    include_embedding: bool = False
    include_superseded: bool = False
    min_similarity: float | None = None
    hybrid: bool = True
    tables: list[str] | None = None
    updated_after: str | None = None
    metadata_filter: dict | None = None
    as_of: str | None = None
    tags: list[str] | None = None
    tags_any: list[str] | None = None
    source: str | None = None
    agent_id: str | None = None
    min_confidence: float | None = None
    recent_first: bool = False
    exclude_relation_context: bool = False
    include_entity_relation_context: bool = False
    stats: dict[str, Any] | None = None
    include_expired: bool = False
    include_deleted: bool = False

# An exact substring hit is stronger evidence than a stemmed full-text hit:
# a file name, an id or a token must outrank a memory that merely shares a
# word with it. The weight is an RRF contribution, not a probability.
EXACT_MATCH_WEIGHT = 1.5

# A single token that looks like an identifier (separators, digits or inner
# capitals) is what FTS tokenizers and embeddings handle badly. Quoted input
# is always treated as an exact needle.
_IDENTIFIER_RE = re.compile(r"^\S{2,120}$")
_IDENTIFIER_SEPARATORS = set("_-./:\\@#")


def _exact_needle(text: str | None) -> str | None:
    """The literal substring a query asks for, or None.

    Only identifier-like queries and quoted phrases use the exact leg. A
    natural-language question would match arbitrary substrings and turn the
    leg into noise; those keep the vector/FTS fusion.
    """
    if not text:
        return None
    value = text.strip()
    if len(value) >= 4 and value[0] == value[-1] == '"':
        value = value[1:-1].strip()
        return value or None
    if not _IDENTIFIER_RE.match(value):
        return None
    if (any(c.isdigit() for c in value)
            or any(c in _IDENTIFIER_SEPARATORS for c in value)
            or any(c.isupper() for c in value[1:])):
        return value
    return None


def _escape_like(value: str) -> str:
    """Escape LIKE/ILIKE wildcards so the needle is matched literally."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _hidden_exact_matches(cur, tables: list[str], needle: str,
                          include_deleted: bool, include_superseded: bool,
                          include_expired: bool,
                          where: str = "",
                          where_params: tuple = ()) -> dict[str, Any] | None:
    """Count exact matches that the visibility flags keep out of the results.

    A hidden row that contains the exact identifier is the difference between
    "this memory does not exist" and "it is expired/superseded/forgotten".
    The count is reported with the flags that would reveal it, so the caller
    does not have to guess.

    `where`/`where_params` are the *non-visibility* filters of the same search
    (mem_type, entity, metadata, relation context). They must be applied here
    too: otherwise a hidden row that the query would never have returned is
    advertised as a match behind a reveal flag.
    """
    pattern = f"%{_escape_like(needle)}%"
    where_sql = f" AND {where}" if where else ""
    params = [pattern, *where_params]
    count = 0
    flags: set[str] = set()
    scopes: dict[str, int] = {}
    truncated = False
    names = _project_names(cur)
    for table in tables:
        try:
            # One extra row tells us whether the LIMIT cut the set, so the
            # count is not silently wrong for a very common identifier.
            cur.execute(
                f"SELECT deleted_at, superseded_by, expires_at FROM {table} "
                f"WHERE content ILIKE %s ESCAPE '\\'{where_sql} LIMIT 101",
                params)
            rows = cur.fetchall()
        except Exception:  # a repair call must not break the search
            logger.debug("hidden exact lookup failed for %s", table,
                         exc_info=True)
            continue
        if len(rows) > 100:
            truncated = True
            rows = rows[:100]
        hidden_here = 0
        for deleted_at, superseded_by, expires_at in rows:
            row_flags: list[str] = []
            expired = _is_expired_value(expires_at)
            if expired and not include_expired:
                row_flags.append("include_expired")
            if deleted_at is not None and not include_deleted:
                row_flags.append("include_deleted")
            # A row that is both deleted and requested as history is visible
            # even when `superseded_by` is set (the visibility flags are
            # orthogonal, deleted wins over superseded in the SQL predicate).
            if (deleted_at is None and superseded_by is not None
                    and not include_superseded):
                row_flags.append("include_superseded")
            if not row_flags:
                continue
            hidden_here += 1
            flags.update(row_flags)
        if hidden_here:
            count += hidden_here
            scope = ("global" if table == "memories"
                     else names.get(table[len("mem_"):], table[len("mem_"):]))
            scopes[scope] = scopes.get(scope, 0) + hidden_here
    if not count:
        return None
    result = {"count": count,
              "flags": sorted(flags),
              "scopes": scopes}
    if truncated:
        result["truncated"] = True
    return result


def _note_dropped_candidate(
    dropped: dict[str, Any] | None,
    row_id: str,
    table: str,
    similarity: float,
) -> None:
    """Keep the strongest vector hit that `min_similarity` rejected.

    A count alone (`dropped_by_similarity: 10`) hides the agent's best match:
    the natural-language question that barely missed the threshold. Reporting
    the strongest rejected candidate (id + similarity + scope) lets the
    caller decide whether one call with `min_similarity=null` is worth it.
    """
    if dropped is None:
        return
    if "similarity" in dropped and similarity <= dropped["similarity"]:
        return
    dropped.clear()
    dropped.update({
        "id": row_id,
        "similarity": round(similarity, 4),
        # The raw table is kept for the preview lookup; `scope` is replaced by
        # the readable project name in `_decorate_best_dropped`.
        "scope": "global" if table == "memories" else table[4:] or "global",
        "_table": table,
    })


def _decorate_best_dropped(cur, dropped: dict[str, Any] | None,
                           result_keys: set[tuple[str, str]] | None = None) -> None:
    """Report the project name and drop a candidate that is already visible.

    The internal table suffix (`test_project_05f71b6a`) means nothing to a
    caller; the registry name (`test-project`) does. A hit that ended up in
    the result set is not "dropped" and must not be advertised as a rejected
    match, otherwise the hint contradicts the response.

    `result_keys` is keyed by `(project, id)`: a memory id is unique per scope,
    so the same id in another scope is a different row - dropping the hint just
    because that other row is visible hid a real near-miss.
    """
    if not dropped:
        return
    names = _project_names(cur)
    table = dropped.pop("_table", None)
    if table and table != "memories":
        suffix = table[len("mem_"):]
        dropped["scope"] = names.get(suffix, dropped.get("scope") or suffix)
    else:
        dropped["scope"] = "global"
    if result_keys and (dropped["scope"], dropped.get("id")) in result_keys:
        dropped.clear()



def _fill_dropped_preview(cur, table: str, row_id: str,
                          dropped: dict[str, Any]) -> None:
    """Fetch the content preview of the strongest rejected candidate."""
    try:
        cur.execute(f"SELECT content FROM {table} WHERE id = %s", (row_id,))
        row = cur.fetchone()
    except Exception:
        return
    if row and row[0]:
        dropped["content_preview"] = str(row[0])[:BEST_DROPPED_PREVIEW_CHARS]


def _table_of_dropped(dropped: dict[str, Any]) -> str:
    table = dropped.get("_table")
    if table:
        return table
    scope = dropped.get("scope") or "global"
    return "memories" if scope == "global" else f"mem_{scope}"


def _apply_history_penalty(row: dict) -> None:
    """Rank history below current facts when it is included on purpose.

    `include_superseded`/`include_deleted`/`include_expired` widen the result
    set; they do not make an outdated row as good as the fact that replaced
    it. Without the penalty the old wording often wins on vector similarity
    alone and the agent reads the stale answer first.
    """
    relevance = row.get("_relevance_score")
    if relevance is None or _is_current_row(row):
        return
    penalised = float(relevance) * HISTORY_RELEVANCE_WEIGHT
    row["_relevance_score"] = round(penalised, 4)
    row["_history_penalty"] = round(1.0 - HISTORY_RELEVANCE_WEIGHT, 4)
    if "_final_score" in row:
        row["_final_score"] = round(
            float(row["_final_score"]) * HISTORY_RELEVANCE_WEIGHT, 4)


def _fts_lexemes(cur, query_text: str | None) -> list[str]:
    """The query lexemes after stemming and stop word removal.

    They measure how much of the query a full-text hit actually covers: an
    OR query makes a single common word look like a perfect match otherwise.
    """
    if not query_text:
        return []
    cur.execute(
        f"SELECT tsvector_to_array(to_tsvector('{config.FTS_LANGUAGE}', %s))",
        (query_text,),
    )
    row = cur.fetchone()
    return list(row[0] or []) if row else []


def _metadata_conditions(metadata_filter: dict) -> tuple[list[str], list[Any]]:
    """Compare metadata values tolerantly.

    `metadata @> %s::jsonb` is JSON-type strict: a filter {"port": "8001"}
    does not match the stored integer 8001, and the caller gets zero hits
    with no hint why. Scalar values are therefore compared as text (numbers
    also numerically), while objects and arrays keep the containment check
    that nested structures need.
    """
    conditions: list[str] = []
    params: list[Any] = []
    for key, value in metadata_filter.items():
        if isinstance(value, (dict, list)):
            conditions.append("metadata @> %s::jsonb")
            params.append(json.dumps({key: value}))
        elif value is None:
            conditions.append("jsonb_typeof(metadata->%s) = 'null'")
            params.append(key)
        elif isinstance(value, bool):
            conditions.append("metadata->>%s = %s")
            params += [key, "true" if value else "false"]
        elif isinstance(value, (int, float)):
            conditions.append(
                "(metadata->>%s = %s OR (jsonb_typeof(metadata->%s) = "
                "'number' AND (metadata->>%s)::numeric = %s))")
            params += [key, str(value), key, key, value]
        else:
            conditions.append("metadata->>%s = %s")
            params += [key, str(value)]
    return conditions, params


def _extra_conditions(
    updated_after: str | None = None,
    metadata_filter: dict | None = None,
    *,
    as_of: str | None = None,
    tags: list[str] | None = None,
    tags_any: list[str] | None = None,
    source: str | None = None,
    agent_id: str | None = None,
    min_confidence: float | None = None,
) -> tuple[list[str], list[Any]]:
    """Build the optional filter conditions shared by all retrieval paths."""
    conditions: list[str] = []
    params: list[Any] = []
    if updated_after:
        stamp = _parse_timestamp(updated_after)
        if stamp is None:
            raise ValueError(
                f"invalid updated_after {updated_after!r}: use an ISO date "
                f"like 2026-10-01 or 2026-10-01T12:00:00Z"
            )
        conditions.append("updated_at >= %s")
        params.append(stamp)
    if metadata_filter:
        meta_conditions, meta_params = _metadata_conditions(metadata_filter)
        conditions.extend(meta_conditions)
        params.extend(meta_params)
    feature_sql, feature_params = feature_conditions(
        as_of=as_of, tags=tags, tags_any=tags_any, source=source,
        agent_id=agent_id, min_confidence=min_confidence,
    )
    conditions.extend(feature_sql)
    params.extend(feature_params)
    return conditions, params


def _count_outside_validity(
    cur,
    table: str,
    base_conditions: list[str],
    base_params: list[Any],
    as_of: Any,
    *,
    include_deleted: bool,
    include_superseded: bool,
    include_expired: bool,
) -> int:
    """Count the rows the query would return if the validity window matched.

    The base conditions must NOT already carry the `as_of` predicate: counting
    `valid AND NOT valid` is always zero and would hide exactly the fact the
    caller wanted to see ("it exists, but not at your `as_of`").
    """
    validity_sql, validity_params = validity_condition(as_of)
    if not validity_sql:
        return 0
    where = " AND ".join(list(base_conditions) + [
        visibility_condition(
            include_deleted=include_deleted,
            include_superseded=include_superseded,
            include_expired=include_expired,
        ),
        f"NOT {validity_sql}",
    ])
    cur.execute(f"SELECT COUNT(*) FROM {table} WHERE {where}",
                list(base_params) + validity_params)
    return int(cur.fetchone()[0])


def _entity_conditions(
    entity: str | None,
    include_relation_context: bool = False,
) -> tuple[list[str], list[Any]]:
    """Filter to one entity, optionally including its edge explanations.

    Relation-context sentences live under the synthetic entity
    ``<from>_<relation>_<to>``, so a plain ``entity = %s`` hides the "why" of
    every edge from an entity-scoped search. Their metadata names both
    endpoints, which lets the same filter find the sentences without also
    returning unrelated context rows.
    """
    if not entity:
        return [], []
    if not include_relation_context:
        return ["entity = %s"], [entity]
    return ([
        "(entity = %s OR ("
        "COALESCE(metadata->>'relation_context', 'false') = 'true' AND "
        "(metadata->>'from' = %s OR metadata->>'to' = %s)))"
    ], [entity, entity, entity])


def fetch_pinned_rows(
    cur,
    tables: list[str],
    *,
    mem_type: str | None = None,
    entity: str | None = None,
    include_entity_relation_context: bool = False,
    exclude_relation_context: bool = False,
    updated_after: str | None = None,
    metadata_filter: dict | None = None,
    as_of: str | None = None,
    tags: list[str] | None = None,
    tags_any: list[str] | None = None,
    source: str | None = None,
    agent_id: str | None = None,
    min_confidence: float | None = None,
    include_deleted: bool = False,
    include_superseded: bool = False,
    include_expired: bool = False,
    limit: int = 200,
) -> list[dict]:
    """The pinned, active rows of the searched scopes, newest first.

    These rows are the always-include candidates of `recall`: they satisfy the
    structural filters (scope, entity, type, tags, provenance, `as_of`, the
    visibility flags) but are deliberately NOT judged by the relevance or decay
    floors - a pin is meant to survive them. Callers merge the result into the
    page and mark it with `_pinned`.
    """
    if not tables:
        return []
    extra_conditions, extra_params = _extra_conditions(
        updated_after, metadata_filter, as_of=as_of, tags=tags,
        tags_any=tags_any, source=source, agent_id=agent_id,
        min_confidence=min_confidence)
    if exclude_relation_context:
        extra_conditions = extra_conditions + [
            "COALESCE(metadata->>'relation_context', 'false') <> 'true'"]
    entity_conditions, entity_params = _entity_conditions(
        entity, include_entity_relation_context)
    project_names = _project_names(cur)
    fetched: list[tuple[dict, str]] = []
    for table in tables:
        conditions = [
            "pinned",
            visibility_condition(include_deleted=include_deleted,
                                 include_superseded=include_superseded,
                                 include_expired=include_expired),
        ]
        params: list[Any] = []
        if mem_type:
            conditions.append("mem_type = %s")
            params.append(mem_type)
        conditions.extend(entity_conditions)
        params.extend(entity_params)
        conditions.extend(extra_conditions)
        params.extend(extra_params)
        where = " AND ".join(conditions)
        cur.execute(
            f"SELECT * FROM {table} WHERE {where} "
            f"ORDER BY updated_at DESC NULLS LAST, id ASC LIMIT %s",
            params + [limit])
        columns = [desc[0] for desc in cur.description]
        for raw in cur.fetchall():
            row = dict(zip(columns, raw))
            row["_project"] = ("global" if table == "memories"
                               else project_names.get(table[4:], table[4:]))
            row["_pinned"] = True
            # No query judged these rows: they carry no relevance. 0.0 keeps
            # the confidence computation honest instead of pretending a match.
            row["_relevance_score"] = 0.0
            fetched.append((row, table))

    def _pin_sort_key(item: dict) -> tuple:
        stamp = item.get("updated_at")
        value = stamp.timestamp() if hasattr(stamp, "timestamp") else 0.0
        return (-value, str(item.get("id") or ""))

    return [row for row, _ in sorted(fetched, key=lambda p: _pin_sort_key(p[0]))]


def fetch_embeddings(cur, tables: list[str],
                     ids: list[str]) -> dict[str, list[float]]:
    """The stored vectors of the given ids, keyed by id.

    `compile_context` needs the vectors to run MMR diversity selection; they
    are used server-side only and never projected into an answer. A missing or
    unparseable vector simply drops out of the map, so the caller falls back to
    relevance-only ordering for that row.
    """
    wanted = [str(i) for i in ids if i]
    if not wanted:
        return {}
    out: dict[str, list[float]] = {}
    for table in tables:
        try:
            cur.execute(
                f"SELECT id, embedding::text FROM {table} "
                f"WHERE id = ANY(%s) AND embedding IS NOT NULL",
                (wanted,))
        except Exception:
            continue
        for row_id, raw in cur.fetchall():
            try:
                out[str(row_id)] = json.loads(raw)
            except (TypeError, ValueError):
                continue
    return out


def _parse_timestamp(value: str):
    """Parse an ISO-8601 date/datetime, or None when it does not parse."""
    try:
        return parse_iso_timestamp(value)
    except ValueError:
        return None


def _sigmoid_scores(values: list[float]) -> list[float]:
    """Calibrate raw cross-encoder logits one by one.

    A softmax over the reranked block depends on the block size: a strong
    hit scored alone becomes ~1.0, the same hit in a block with one rival
    suddenly ~0.5. The published `_relevance_score` then means different
    things on page 1 and page 2. The sigmoid of a single logit is independent
    of the candidate block, so the score stays comparable across queries and
    pages (it is a per-pair confidence, not a share of the block).
    """
    if not values:
        return []
    scores: list[float] = []
    for value in values:
        try:
            logit = float(value)
        except (TypeError, ValueError):
            logit = 0.0
        # Keep exp() finite for a broken provider that returns huge logits.
        logit = max(-60.0, min(60.0, logit))
        scores.append(1.0 / (1.0 + math.exp(-logit)))
    return scores


def _rerank_rows(
    rows: list[dict],
    query_text: str | None,
    stats: dict[str, Any] | None,
    include_decay: bool,
) -> list[dict]:
    """Reorder the first-stage candidates with a local cross-encoder.

    The bi-encoder is fast but scores query and memory independently; a
    vocabulary-shifting paraphrase can therefore sit below `min_similarity`
    while an unrelated decoy survives. The cross-encoder sees both texts
    together and is only asked when the first stage is genuinely ambiguous:
    a confident winner (`>= RERANK_TRIGGER_SCORE`) or a clear margin
    (`>= RERANK_TRIGGER_MARGIN`) is trusted as is. That keeps the ~2s CPU
    cost out of the common path without hiding the hard case.

    The raw logits are not probabilities, so `_rerank_score` is the sigmoid
    of each logit: a per-pair confidence that does not change when another
    candidate joins or leaves the block. The sigmoid is not calibrated as an
    absolute relevance (a clearly relevant pair often scores around 0.4), so
    it is used to *cap* the first-stage relevance rather than replace it:
    `_relevance_score = min(first_stage_relevance, sigmoid(logit + bias))`.
    A strong reranker rejection lowers confidence; a merely modest positive
    logit never promotes a weak first-stage hit. `_similarity`, `_rrf_score`
    and `_rrf_norm` stay the first-stage values.
    """
    if not RERANK_ENABLED or not query_text or not rows:
        return rows
    candidates = [
        row for row in rows
        if row.get("content") and not row.get("_graph_source")
    ]
    candidates.sort(key=lambda row: (-row.get("_relevance_score", 0.0),
                                     -row.get("_rrf_score", 0.0)))
    candidates = candidates[:RERANK_TOP_K]
    if len(candidates) < 2:
        return rows
    if not RERANK_ALWAYS:
        best = float(candidates[0].get("_relevance_score") or 0.0)
        second = float(candidates[1].get("_relevance_score") or 0.0)
        if (best >= RERANK_TRIGGER_SCORE
                or (best - second) >= RERANK_TRIGGER_MARGIN):
            return rows
    try:
        logits = rerank(query_text, [row["content"] for row in candidates])
    except Exception as exc:
        logger.warning("reranking skipped: %s", exc)
        if stats is not None:
            stats["rerank_error"] = str(exc)
        return rows
    if len(logits) != len(candidates):
        message = ("rerank returned "
                   f"{len(logits)} scores for {len(candidates)} candidates")
        logger.warning("reranking skipped: %s", message)
        if stats is not None:
            stats["rerank_error"] = message
        return rows
    if any(not math.isfinite(score) for score in logits):
        message = "rerank returned a non-finite score"
        logger.warning("reranking skipped: %s", message)
        if stats is not None:
            stats["rerank_error"] = message
        return rows

    probabilities = _sigmoid_scores(logits)
    caps = _sigmoid_scores([logit + RERANK_LOGIT_BIAS for logit in logits])
    for row, probability, cap in zip(candidates, probabilities, caps):
        row["_rerank_score"] = round(probability, 4)
        base_relevance = float(row.get("_relevance_score") or 0.0)
        # The reranker can only lower the first-stage relevance: its absolute
        # logits are not calibrated, but a very negative logit is a strong
        # "not relevant" signal. Exact literal hits keep their floor.
        relevance = min(base_relevance, cap)
        if row.get("_exact_match") == "content":
            relevance = max(relevance, 0.95)
        elif row.get("_exact_match") == "substring":
            relevance = max(relevance, 0.85)
        row["_relevance_score"] = round(relevance, 4)

    candidate_ids = {id(row) for row in candidates}
    # Order by the capped relevance: a reranker may demote a candidate but a
    # high logit must not promote a weak first-stage hit above a stronger one.
    reranked_head = sorted(
        candidates,
        key=lambda row: (-row.get("_relevance_score", 0.0),
                         -row.get("_rerank_score", 0.0),
                         -row.get("_rrf_score", 0.0)))
    rows[:] = reranked_head + [row for row in rows
                               if id(row) not in candidate_ids]
    if stats is not None:
        stats["reranked"] = len(candidates)

    if include_decay:
        for row in rows:
            relevance = row.get("_relevance_score")
            if relevance is None:
                row["_final_score"] = row.get("_decay_norm", 0.0)
            else:
                row["_final_score"] = _clamp01(
                    _blend_scores(relevance, row.get("_decay_norm", 0.0)))
        rows.sort(key=lambda row: -row.get("_final_score", 0.0))
    return rows


def _search_memories_rrf(
    cur,
    query_embedding: list[float] | None,
    query_text: str | None,
    mem_type: str | None,
    entity: str | None,
    limit: int,
    project: str | None,
    include_decay: bool = False,
    include_embedding: bool = False,
    include_superseded: bool = False,
    min_similarity: float | None = None,
    include_expired: bool = False,
    include_deleted: bool = False,
    k: int = RRF_K,
    tables: list[str] | None = None,
    updated_after: str | None = None,
    metadata_filter: dict | None = None,
    exclude_relation_context: bool = False,
    include_entity_relation_context: bool = False,
    stats: dict[str, Any] | None = None,
    as_of: str | None = None,
    tags: list[str] | None = None,
    tags_any: list[str] | None = None,
    source: str | None = None,
    agent_id: str | None = None,
    min_confidence: float | None = None,
) -> list[dict]:
    """
    Reciprocal Rank Fusion over vector + FTS results.

    Retrieval runs against the project table (if any) and the global table;
    project rows win when both contain the same content. Scores are normalized
    so that decay can reorder results without drowning out relevance.

    `min_similarity` only applies to the vector retriever. A full-text hit is
    kept even when its embedding is a poor match: that is exactly the case
    the full-text leg exists for (an identifier, a file name, a foreign word
    that the embedding model handles badly). The number of candidates that
    the filter removed is reported in `stats`.
    """
    tables = tables if tables is not None else _retrieval_tables(cur, project)
    if not tables:
        return []

    candidate_cap = max(50, limit)
    max_distance = None if min_similarity is None else 1.0 - min_similarity
    fts_query = _fts_query(query_text)
    exact_needle = _exact_needle(query_text)
    # Query lexemes after stemming/stopword removal. They are used to measure
    # how much of the query a full-text hit actually covers: an OR query makes
    # a single common word look like a perfect match otherwise.
    fts_lexemes: list[str] = _fts_lexemes(cur, query_text)
    extra_conditions, extra_params = _extra_conditions(
        updated_after, metadata_filter, as_of=as_of, tags=tags,
        tags_any=tags_any, source=source, agent_id=agent_id,
        min_confidence=min_confidence)
    if exclude_relation_context:
        # The relation-context sentences repeat what the graph edge already
        # says; leave them out of a recall ranking when asked to.
        extra_conditions = extra_conditions + [
            "COALESCE(metadata->>'relation_context', 'false') <> 'true'"]

    # (score, id, table) tuples, merged across tables before ranking
    vec_candidates: list[tuple[float, str, str]] = []
    # (rank, matched_lexemes, id, table)
    text_candidates: list[tuple[float, int, str, str]] = []
    # (is_full_content_match, id, table)
    exact_candidates: list[tuple[bool, str, str]] = []
    dropped_by_similarity = 0
    dropped_by_expired = 0
    dropped_by_validity = 0
    dropped_by_cross_scope_duplicate = 0
    best_dropped: dict[str, Any] | None = {}
    entity_conditions, entity_params = _entity_conditions(
        entity, include_entity_relation_context)
    # The same filters *without* the visibility predicate: the hidden-match
    # hint must describe rows this query would return if the flag were set.
    hidden_where = " AND ".join(
        ["TRUE"]
        + (["mem_type = %s"] if mem_type else [])
        + entity_conditions
        + extra_conditions)
    hidden_params: list[Any] = []
    if mem_type:
        hidden_params.append(mem_type)
    hidden_params.extend(entity_params)
    hidden_params.extend(extra_params)

    # The same filters without the `as_of` predicate: the validity counter has
    # to invert the window, not intersect it with itself.
    validity_free_extra, validity_free_params = _extra_conditions(
        updated_after, metadata_filter, as_of=None, tags=tags,
        tags_any=tags_any, source=source, agent_id=agent_id,
        min_confidence=min_confidence)
    if exclude_relation_context:
        validity_free_extra = validity_free_extra + [
            "COALESCE(metadata->>'relation_context', 'false') <> 'true'"]
    base_conditions = (["TRUE"] + (["mem_type = %s"] if mem_type else [])
                       + entity_conditions + validity_free_extra)
    base_params: list[Any] = []
    if mem_type:
        base_params.append(mem_type)
    base_params.extend(entity_params)
    base_params.extend(validity_free_params)

    for table in tables:
        conditions = ["TRUE"]
        params: list[Any] = []
        if mem_type:
            conditions.append("mem_type = %s")
            params.append(mem_type)
        conditions.extend(entity_conditions)
        params.extend(entity_params)
        conditions.extend(extra_conditions)
        params.extend(extra_params)
        # Expired rows are excluded by the visibility predicate. Count them
        # separately so an empty answer can say "the fact exists but its TTL
        # passed" instead of "nothing matches".
        if not include_expired:
            expired_where = " AND ".join(conditions + [
                visibility_condition(
                    include_deleted=include_deleted,
                    include_superseded=include_superseded,
                    include_expired=True,
                ),
                "expires_at IS NOT NULL AND expires_at <= NOW()",
            ])
            cur.execute(
                f"SELECT COUNT(*) FROM {table} WHERE {expired_where}",
                params,
            )
            dropped_by_expired += int(cur.fetchone()[0])
        if as_of is not None:
            # Rows this query would return except for the point in time: the
            # same filters, the same visibility, only the validity predicate
            # inverted. Reported like the expired counter so an empty answer
            # cannot hide "the fact is newer than your as_of".
            dropped_by_validity += _count_outside_validity(
                cur, table, base_conditions, base_params, as_of,
                include_deleted=include_deleted,
                include_superseded=include_superseded,
                include_expired=include_expired)
        conditions.append(visibility_condition(
            include_deleted=include_deleted,
            include_superseded=include_superseded,
            include_expired=include_expired,
        ))
        where = " AND ".join(conditions)

        # 1. Vector Search (Top 50 per table)
        if query_embedding is not None:
            sql = (
                f"SELECT id, embedding <=> %s::vector AS distance FROM {table} "
                f"WHERE {where} AND embedding IS NOT NULL"
            )
            vec_params: list[Any] = [query_embedding] + params
            # The distance filter is applied below, not in SQL: only then the
            # number of rejected candidates can be reported to the caller.
            sql += " ORDER BY distance ASC LIMIT %s"
            vec_params.append(candidate_cap)
            cur.execute(sql, vec_params)
            for row_id, dist in cur.fetchall():
                dist = float(dist)
                if max_distance is not None and dist > max_distance:
                    dropped_by_similarity += 1
                    _note_dropped_candidate(
                        best_dropped, row_id, table, 1.0 - dist)
                    continue
                vec_candidates.append((dist, row_id, table))

        # 2. FTS Search (Top 50 per table), OR-joined terms
        if fts_query:
            sql = (
                f"SELECT id, ts_rank(to_tsvector('{config.FTS_LANGUAGE}', content), "
                f"websearch_to_tsquery('{config.FTS_LANGUAGE}', %s)) AS rank, "
                f"(SELECT COUNT(*) FROM unnest(%s::text[]) AS lex "
                f"WHERE to_tsvector('{config.FTS_LANGUAGE}', content) @@ "
                f"plainto_tsquery('{config.FTS_LANGUAGE}', lex)) AS matched "
                f"FROM {table} "
                f"WHERE to_tsvector('{config.FTS_LANGUAGE}', content) @@ "
                f"websearch_to_tsquery('{config.FTS_LANGUAGE}', %s)"
            )
            fts_params: list[Any] = [fts_query, fts_lexemes, fts_query]
            # No similarity filter here: a full-text hit stays a hit even when
            # the embedding model scores it poorly.
            sql += f" AND {where} ORDER BY rank DESC LIMIT %s"
            cur.execute(sql, fts_params + params + [candidate_cap])
            text_candidates.extend(
                (float(rank), int(matched or 0), row_id, table)
                for row_id, rank, matched in cur.fetchall()
            )

        # 3. Exact substring search (identifier-like queries only). FTS
        # tokenizes `ZEBRAFISH_TOKEN` or `app/config.py` in ways that can miss
        # the literal token; ILIKE does not. The result is a candidate, not a
        # proof: the ranking still combines it with the other legs.
        if exact_needle:
            pattern = f"%{_escape_like(exact_needle)}%"
            sql = (
                f"SELECT id, (lower(content) = lower(%s)) AS exact_content "
                f"FROM {table} "
                f"WHERE content ILIKE %s ESCAPE '\\' AND {where} "
                f"ORDER BY exact_content DESC, length(content) ASC LIMIT %s"
            )
            cur.execute(sql, [exact_needle, pattern] + params + [candidate_cap])
            exact_candidates.extend(
                (bool(full), row_id, table)
                for row_id, full in cur.fetchall()
            )

    vec_candidates.sort(key=lambda c: c[0])
    text_candidates.sort(key=lambda c: -c[0])

    # 3. RRF Fusion
    # The key is (table, id): a memory id is unique per scope, not globally,
    # and an import can copy the same id into another scope. Keying by id
    # alone would let the two unrelated rows overwrite each other.
    Key = tuple[str, str]
    vector_results: dict[Key, float] = {}
    text_results: dict[Key, float] = {}
    exact_results: dict[Key, float] = {}
    exact_modes: dict[Key, str] = {}
    similarities: dict[Key, float] = {}
    text_scores: dict[Key, float] = {}
    coverages: dict[Key, float] = {}

    for rank, (distance, row_id, table) in enumerate(vec_candidates[:candidate_cap], 1):
        key = (table, row_id)
        vector_results[key] = 1.0 / (k + rank)
        similarities[key] = 1.0 - distance
    for rank, (_, _, row_id, table) in enumerate(text_candidates[:candidate_cap], 1):
        text_results[(table, row_id)] = FTS_WEIGHT / (k + rank)
    for rank, (full, row_id, table) in enumerate(exact_candidates[:candidate_cap], 1):
        key = (table, row_id)
        exact_results[key] = EXACT_MATCH_WEIGHT / (k + rank)
        exact_modes[key] = "content" if full else "substring"

    max_text_rank = max((r for r, _, _, _ in text_candidates[:candidate_cap]), default=0.0)
    lexeme_count = max(1, len(fts_lexemes))
    for rank, (ts_rank, matched, row_id, table) in enumerate(text_candidates[:candidate_cap], 1):
        # Normalized full-text evidence: the best text hit is 1.0, weaker
        # hits fall off with their text rank. Multiply by query coverage so a
        # hit that matched only one word of a multi-word query cannot look
        # like a perfect match.
        rank_norm = (ts_rank / max_text_rank) if max_text_rank else 0.0
        coverage = min(1.0, matched / lexeme_count) if fts_lexemes else 1.0
        key = (table, row_id)
        text_scores[key] = rank_norm * coverage
        coverages[key] = coverage

    fused_scores = {
        key: (vector_results.get(key, 0.0) + text_results.get(key, 0.0)
              + exact_results.get(key, 0.0))
        for key in (set(vector_results) | set(text_results)
                    | set(exact_results))
    }
    # A hit that only the full-text leg found has no similarity to vouch for
    # it. If it covers only a sliver of the query it is noise, and if it is
    # the *only* hit it would be handed to the agent as the answer. Hits the
    # vector retriever found are never dropped here.
    dropped_by_coverage = 0
    if fts_lexemes:
        weak = [key for key in fused_scores
                if key not in similarities and key not in exact_results
                and coverages.get(key, 1.0) < FTS_MIN_COVERAGE]
        for key in weak:
            del fused_scores[key]
        dropped_by_coverage = len(weak)
    hidden_exact = (
        _hidden_exact_matches(cur, tables, exact_needle, include_deleted,
                              include_superseded, include_expired,
                              hidden_where, tuple(hidden_params))
        if exact_needle else None)
    if not fused_scores:
        if stats is not None:
            if best_dropped:
                _fill_dropped_preview(
                    cur, _table_of_dropped(best_dropped),
                    best_dropped["id"], best_dropped)
                _decorate_best_dropped(cur, best_dropped)
            stats.update({"candidates": 0,
                          "dropped_by_similarity": dropped_by_similarity,
                          "dropped_by_expired": dropped_by_expired,
                          "dropped_by_coverage": dropped_by_coverage,
                          "dropped_by_cross_scope_duplicate":
                              dropped_by_cross_scope_duplicate,
                          "min_coverage": FTS_MIN_COVERAGE,
                          "fulltext_only": 0,
                          "exact_matches": 0})
            if as_of is not None:
                stats["dropped_by_validity"] = dropped_by_validity
            if hidden_exact:
                stats["hidden_exact_match"] = hidden_exact
            if best_dropped:
                stats["best_dropped_by_similarity"] = best_dropped
        return []

    max_rrf = max(fused_scores.values()) or 1.0
    # Do not cut before cross-table dedup: many duplicate contents could
    # otherwise hide unique rows that would fit into the limit.
    top_ids = sorted(fused_scores, key=lambda x: -fused_scores[x])

    # 4. Load details per table (project first), dropping cross-table duplicates
    rows: list[dict] = []
    seen_active_hashes: dict[str, str] = {}
    project_names = _project_names(cur)
    for table in tables:
        ids = [row_id for (cand_table, row_id) in top_ids if cand_table == table]
        if not ids:
            continue
        cur.execute(f"SELECT * FROM {table} WHERE id = ANY(%s)", (ids,))
        colnames = [desc[0] for desc in cur.description]
        by_id = {r[0]: dict(zip(colnames, r)) for r in cur.fetchall()}
        for row_id in ids:
            key = (table, row_id)
            d = by_id.get(row_id)
            if d is None:
                continue
            if table == "memories":
                d["_project"] = "global"
            else:
                d["_project"] = project_names.get(table[4:], table[4:])
            if is_cross_scope_duplicate(
                d, table, seen_active_hashes
            ):
                dropped_by_cross_scope_duplicate += 1
                continue
            d["_rrf_score"] = fused_scores.get(key, 0.0)
            d["_rrf_norm"] = d["_rrf_score"] / max_rrf
            d["_similarity"] = round(similarities[key], 4) if key in similarities else None
            d["_distance"] = round(1.0 - d["_similarity"], 4) if d["_similarity"] is not None else None
            if key in text_scores:
                d["_text_score"] = round(text_scores[key], 4)
            # Relevance combines both signals, so a memory found by the
            # full-text retriever alone is still ranked above a weak vector
            # match instead of being flattened into the same RRF band.
            relevance = _relevance_score(similarities.get(key, 0.0),
                                         text_scores.get(key, 0.0))
            if key in exact_results:
                # An exact literal hit is its own retriever signal (it enters
                # the fusion with a stronger RRF weight and is exempt from the
                # FTS-only noise floor). `_relevance_score` is left as the
                # genuine vector+FTS blend: overwriting it with a constant
                # flattened two exact hits of very different similarity into
                # the same score and made the field a lie.
                d["_exact_match"] = exact_modes[key]
            # A project-scoped row wins over an identical global one: the
            # global scope is the fallback, not an equal peer. The boost is
            # compressed towards 1 instead of added and clamped, so strong
            # hits stay comparable (0.97 vs 1.0 instead of flat 1.0) and the
            # value remains inside the documented 0..1 range. Published as
            # `_scope_boost` so the ranking stays explainable.
            if table != "memories":
                effective_boost = round(PROJECT_SCOPE_BOOST * (1.0 - relevance), 4)
                relevance = relevance + effective_boost
                d["_scope_boost"] = effective_boost
            d["_relevance_score"] = round(_clamp01(relevance), 4)
            _apply_history_penalty(d)
            if not include_embedding:
                d.pop("embedding", None)
            if include_decay:
                d["_decay_score"] = _decay_score(
                    d.get("created_at"), d.get("updated_at"),
                    d.get("access_count", 0), d.get("importance", 1.0),
                )
                d["_decay_norm"] = _decay_norm(
                    d.get("created_at"), d.get("updated_at"),
                    d.get("access_count", 0), d.get("importance", 1.0),
                )
            rows.append(d)

    if stats is not None:
        if best_dropped:
            _fill_dropped_preview(
                cur, _table_of_dropped(best_dropped),
                best_dropped["id"], best_dropped)
            _decorate_best_dropped(cur, best_dropped,
                                   {(r.get("_project") or "global", r.get("id"))
                                    for r in rows})
        stats.update({
            "candidates": len(fused_scores),
            "dropped_by_similarity": dropped_by_similarity,
            "dropped_by_expired": dropped_by_expired,
            "dropped_by_coverage": dropped_by_coverage,
            "dropped_by_cross_scope_duplicate":
                dropped_by_cross_scope_duplicate,
            "min_coverage": FTS_MIN_COVERAGE,
            "fulltext_only": sum(1 for key in fused_scores
                                 if key not in similarities),
            "exact_matches": sum(1 for key in fused_scores
                                 if key in exact_results),
        })
        if as_of is not None:
            stats["dropped_by_validity"] = dropped_by_validity
        if hidden_exact:
            stats["hidden_exact_match"] = hidden_exact
        if best_dropped:
            stats["best_dropped_by_similarity"] = best_dropped

    # 5. Re-rank
    if include_decay:
        for r in rows:
            r["_final_score"] = _clamp01(
                _blend_scores(r["_relevance_score"], r["_decay_norm"])
            )
        rows.sort(key=lambda x: -x.get("_final_score", 0))
    else:
        rows.sort(key=lambda x: (-x.get("_relevance_score", 0.0),
                                 -x.get("_rrf_score", 0.0)))

    return rows



def _search_memories(cur, request: SearchRequest) -> list[dict]:
    """
    Wrapper for search functions. Uses RRF when hybrid=True, otherwise legacy search.

    `tables` overrides the table set (default: project table + global).
    `recent_first` keeps the "newest first" order of a browse instead of
    re-ranking by decay score - the session bootstrap needs recency, not the
    most frequently accessed memories.
    """
    query_embedding = request.query_embedding
    query_text = request.query_text
    mem_type = request.mem_type
    entity = request.entity
    limit = request.limit
    project = request.project
    include_decay = request.include_decay
    include_embedding = request.include_embedding
    include_superseded = request.include_superseded
    min_similarity = request.min_similarity
    hybrid = request.hybrid
    tables = request.tables
    updated_after = request.updated_after
    metadata_filter = request.metadata_filter
    as_of = request.as_of
    tags = request.tags
    tags_any = request.tags_any
    source = request.source
    agent_id = request.agent_id
    min_confidence = request.min_confidence
    recent_first = request.recent_first
    exclude_relation_context = request.exclude_relation_context
    include_entity_relation_context = request.include_entity_relation_context
    stats = request.stats
    include_expired = request.include_expired
    include_deleted = request.include_deleted
    if hybrid and (query_embedding is not None or query_text):
        rows = _search_memories_rrf(
            cur, query_embedding, query_text, mem_type, entity, limit, project,
            include_decay, include_embedding, include_superseded, min_similarity,
            include_expired=include_expired,
            include_deleted=include_deleted,
            tables=tables, updated_after=updated_after,
            metadata_filter=metadata_filter,
            exclude_relation_context=exclude_relation_context,
            include_entity_relation_context=include_entity_relation_context,
            stats=stats,
            as_of=as_of, tags=tags, tags_any=tags_any, source=source,
            agent_id=agent_id, min_confidence=min_confidence,
        )
        _rerank_rows(rows, query_text, stats, include_decay)
        return rows[:limit]

    # Legacy fallback for individual retrievers
    tables = tables if tables is not None else _retrieval_tables(cur, project)
    if not tables:
        return []

    max_distance = None if min_similarity is None else 1.0 - min_similarity
    has_vec = query_embedding is not None
    fts_query = _fts_query(query_text)
    fts_lexemes: list[str] = _fts_lexemes(cur, query_text)
    extra_conditions, extra_params = _extra_conditions(
        updated_after, metadata_filter, as_of=as_of, tags=tags,
        tags_any=tags_any, source=source, agent_id=agent_id,
        min_confidence=min_confidence)
    if exclude_relation_context:
        extra_conditions = extra_conditions + [
            "COALESCE(metadata->>'relation_context', 'false') <> 'true'"]
    validity_free_extra, validity_free_params = _extra_conditions(
        updated_after, metadata_filter, as_of=None, tags=tags,
        tags_any=tags_any, source=source, agent_id=agent_id,
        min_confidence=min_confidence)
    if exclude_relation_context:
        validity_free_extra = validity_free_extra + [
            "COALESCE(metadata->>'relation_context', 'false') <> 'true'"]

    rows: list[dict] = []
    seen_active_hashes: dict[str, str] = {}
    dropped_by_similarity = 0
    dropped_by_expired = 0
    dropped_by_validity = 0
    dropped_by_coverage = 0
    dropped_by_cross_scope_duplicate = 0
    best_dropped: dict[str, Any] | None = {}

    project_names = _project_names(cur)
    # Fetch more than the caller asked for: the similarity filter is applied
    # in Python, so the limit must not cut the candidates before filtering.
    fetch_limit = max(limit, 50)
    for table in tables:
        filter_conditions: list[str] = ["TRUE"]
        filter_params: list[Any] = []
        params_vec: list[Any] = []

        if has_vec:
            col = "*, embedding <=> %s::vector AS distance"
            filter_conditions.append("embedding IS NOT NULL")
            order = "ORDER BY distance ASC"
            params_vec.append(query_embedding)
        elif fts_query:
            col = (f"*, ts_rank(to_tsvector('{config.FTS_LANGUAGE}', content), "
                   f"websearch_to_tsquery('{config.FTS_LANGUAGE}', %s)) AS fts_rank, "
                   f"(SELECT COUNT(*) FROM unnest(%s::text[]) AS lex "
                   f"WHERE to_tsvector('{config.FTS_LANGUAGE}', content) @@ "
                   f"plainto_tsquery('{config.FTS_LANGUAGE}', lex)) AS matched")
            params_vec.append(fts_query)
            params_vec.append(fts_lexemes)
            order = "ORDER BY fts_rank DESC"

        else:
            col = "*"
            order = "ORDER BY updated_at DESC"

        if fts_query:
            filter_conditions.append(
                f"to_tsvector('{config.FTS_LANGUAGE}', content) @@ "
                f"websearch_to_tsquery('{config.FTS_LANGUAGE}', %s)"
            )
            filter_params.append(fts_query)
        if mem_type:
            filter_conditions.append("mem_type = %s")
            filter_params.append(mem_type)
        entity_conditions, entity_params = _entity_conditions(
            entity, include_entity_relation_context)
        filter_conditions.extend(entity_conditions)
        filter_params.extend(entity_params)
        conditions = filter_conditions + extra_conditions
        params = filter_params + extra_params
        base_conditions = filter_conditions + validity_free_extra
        base_params = filter_params + validity_free_params
        if not include_expired:
            expired_where = " AND ".join(conditions + [
                visibility_condition(
                    include_deleted=include_deleted,
                    include_superseded=include_superseded,
                    include_expired=True,
                ),
                "expires_at IS NOT NULL AND expires_at <= NOW()",
            ])
            cur.execute(
                f"SELECT COUNT(*) FROM {table} WHERE {expired_where}",
                params,
            )
            dropped_by_expired += int(cur.fetchone()[0])
        if as_of is not None:
            dropped_by_validity += _count_outside_validity(
                cur, table, base_conditions, base_params, as_of,
                include_deleted=include_deleted,
                include_superseded=include_superseded,
                include_expired=include_expired)

        conditions.append(visibility_condition(
            include_deleted=include_deleted,
            include_superseded=include_superseded,
            include_expired=include_expired,
        ))

        where = " AND ".join(conditions)
        cur.execute(
            f"SELECT {col}, created_at, updated_at, access_count, importance "
            f"FROM {table} WHERE {where} {order} LIMIT %s",
            params_vec + params + [fetch_limit],
        )

        colnames = [desc[0] for desc in cur.description] if cur.description else []
        for row in cur.fetchall():
            d = dict(zip(colnames, row))
            if table == "memories":
                d["_project"] = "global"
            else:
                d["_project"] = project_names.get(table[4:], table[4:])
            if "distance" in d and d["distance"] is not None:
                d["_distance"] = float(d.pop("distance"))
                if max_distance is not None and d["_distance"] > max_distance:
                    dropped_by_similarity += 1
                    _note_dropped_candidate(
                        best_dropped, d["id"], table, 1.0 - d["_distance"])
                    continue
                d["_similarity"] = round(1.0 - d["_distance"], 4)
            else:
                d.pop("distance", None)
            if "fts_rank" in d:
                d["_fts_rank"] = float(d.pop("fts_rank") or 0.0)
            if "matched" in d:
                # The full-text-only path: a hit that covers just a sliver of
                # the query is noise, not an answer.
                matched = int(d.pop("matched") or 0)
                lexeme_count = max(1, len(fts_lexemes))
                coverage = min(1.0, matched / lexeme_count) if fts_lexemes else 1.0
                d["_coverage"] = round(coverage, 4)
                if coverage < FTS_MIN_COVERAGE:
                    dropped_by_coverage += 1
                    continue

            if is_cross_scope_duplicate(
                d, table, seen_active_hashes
            ):
                dropped_by_cross_scope_duplicate += 1
                continue
            if not include_embedding:
                d.pop("embedding", None)
            if include_decay:
                d["_decay_score"] = _decay_score(
                    d.get("created_at"), d.get("updated_at"),
                    d.get("access_count", 0), d.get("importance", 1.0),
                )
                d["_decay_norm"] = _decay_norm(
                    d.get("created_at"), d.get("updated_at"),
                    d.get("access_count", 0), d.get("importance", 1.0),
                )
            rows.append(d)

    if has_vec:
        rows.sort(key=lambda x: x.get("_distance", 1.0))
    elif fts_query:
        max_rank = max((r.get("_fts_rank", 0.0) for r in rows), default=0.0)
        for row in rows:
            rank = row.pop("_fts_rank", 0.0)
            row["_text_score"] = round(rank / max_rank, 4) if max_rank else 0.0
            row["_relevance_score"] = row["_text_score"]
            _apply_history_penalty(row)
        rows.sort(key=lambda x: -x.get("_relevance_score", 0.0))
    else:
        rows.sort(key=lambda x: x.get("updated_at") or _now(), reverse=True)

    if has_vec:
        for row in rows:
            row["_relevance_score"] = row.get("_similarity", 0.0)
            _apply_history_penalty(row)
        if not recent_first:
            rows.sort(key=lambda x: -x.get("_relevance_score", 0.0))
    if stats is not None:
        if best_dropped:
            if "content_preview" not in best_dropped:
                _fill_dropped_preview(
                    cur, _table_of_dropped(best_dropped),
                    best_dropped["id"], best_dropped)
            _decorate_best_dropped(cur, best_dropped,
                                   {(r.get("_project") or "global", r.get("id"))
                                    for r in rows})
        if best_dropped:
            stats["best_dropped_by_similarity"] = best_dropped
        stats.update({
            "candidates": len(rows),
            "dropped_by_similarity": dropped_by_similarity,
            "dropped_by_expired": dropped_by_expired,
            "dropped_by_coverage": dropped_by_coverage,
            "dropped_by_cross_scope_duplicate":
                dropped_by_cross_scope_duplicate,
            "min_coverage": FTS_MIN_COVERAGE,
            "fulltext_only": 0,
        })
        if as_of is not None:
            stats["dropped_by_validity"] = dropped_by_validity
    if include_decay:
        for r in rows:
            relevance = r.get("_relevance_score")
            if relevance is not None:
                r["_final_score"] = _clamp01(
                    _blend_scores(relevance, r["_decay_norm"])
                )
            else:
                # Nothing relevant to blend with: recency/frequency only.
                r["_final_score"] = r["_decay_norm"]
        if not recent_first:
            rows.sort(key=lambda x: -x.get("_final_score", 0))

    if not recent_first:
        _rerank_rows(rows, query_text, stats, include_decay)
    return rows[:limit]



def _search_memories_graph_boost(
    existing_results: list[dict],
    entity: str,
    limit: int,
    project: str | None,
    include_decay: bool = False,
    graph_boost_weight: float = 0.3,  # 30% boost for graph candidates
    include_global: bool = True,
    updated_after: str | None = None,
    metadata_filter: dict | None = None,
    exclude_relation_context: bool = False,
    all_projects: bool = False,
    include_expired: bool = False,
    include_deleted: bool = False,
    warnings: list[str] | None = None,
    cur=None,
    as_of: str | None = None,
    tags: list[str] | None = None,
    tags_any: list[str] | None = None,
    source: str | None = None,
    agent_id: str | None = None,
    min_confidence: float | None = None,
) -> list[dict]:
    """
    Augments search results with memories from connected entities via the graph.
    Merges graph candidates with RRF scores.

    A `graph_entity` that has no graph node and no memories is a silent no-op
    otherwise: the caller asked to expand a neighbourhood that does not exist
    and would read the unchanged result as a complete one. The warning names
    the entity and where it was looked up.
    """
    tables = None
    graph_entities: list[tuple[str, float]] = []
    # Relation-context sentences are stored under the synthetic entity
    # `<from>_<relation>_<to>`; they describe the very edge that was just
    # traversed, so a graph expansion that ignores them loses the "why" of
    # the connection. Both directions are tried because the match is
    # undirected.
    relation_entities: list[tuple[str, float]] = []
    graph_results: list[dict] = []

    # A Cypher failure must not abort the running recall transaction. The
    # caller can pass its cursor: the expansion then runs behind a savepoint,
    # so it still crosses the GraphStore seam on the same transaction.
    own_connection = cur is None
    db = None
    if own_connection:
        db = MemoryStore()
        cur = db.cur
    else:
        cur.execute("SAVEPOINT graph_boost")
    try:
        if all_projects:
            # Every scope has its own AGE label, so the traversal walks all
            # of them.
            tables = _all_mem_tables(cur)
            if not include_global:
                tables = [t for t in tables if t != "memories"]
            labels = ["Entity"] if include_global else []
            for table in tables:
                if table == "memories":
                    continue
                suffix = table[len("mem_"):]
                labels.append(f"Project_{suffix}")
        else:
            tables = _retrieval_tables(cur, project)
            if project and not include_global:
                tables = [table for table in tables if table != "memories"]
            labels = [_scope_label(project)]

        # 1. Find connected entities via Apache AGE (any edge direction/type)
        seen_nodes: set[str] = set()
        seen_relation_context: set[str] = set()
        for label in labels:
            for name, weight, rtype, edge_source in GraphStore(cur=cur).boost(label, entity):
                node = _ag_str(name)
                weight_value = float(str(weight))
                relation = _ag_str(rtype)
                source_name = _ag_str(edge_source)
                if relation:
                    for ctx_name in (f"{source_name}_{relation}_{node}",
                                     f"{node}_{relation}_{source_name}"):
                        if ctx_name not in seen_relation_context:
                            seen_relation_context.add(ctx_name)
                            relation_entities.append((ctx_name, weight_value))
                if node in seen_nodes:
                    continue
                seen_nodes.add(node)
                graph_entities.append((node, weight_value))

        # 2. Fetch memories of connected entities
        extra_conditions, extra_params = _extra_conditions(
            updated_after, metadata_filter, as_of=as_of, tags=tags,
            tags_any=tags_any, source=source, agent_id=agent_id,
            min_confidence=min_confidence)
        if exclude_relation_context:
            extra_conditions = extra_conditions + [
                "COALESCE(metadata->>'relation_context', 'false') <> 'true'"]
        extra_sql = "" if not extra_conditions else " AND " + " AND ".join(extra_conditions)
        # A memory id is unique per scope, so the same id in another scope is a
        # different row and must not suppress a genuine graph hit.
        existing_keys = {((r.get("_project") or "global"), r["id"])
                         for r in existing_results}
        project_names = _project_names(cur)

        def _collect(entity_name: str, weight: float,
                     relation_context: bool = False) -> None:
            for table in tables:
                cur.execute(
                    f"SELECT * FROM {table} "
                    f"WHERE entity = %s AND "
                    + visibility_condition(
                        include_deleted=include_deleted,
                        include_expired=include_expired)
                    + extra_sql
                    + " LIMIT 5",
                    [entity_name] + extra_params,
                )
                for row in cur.fetchall():
                    d = dict(zip([desc[0] for desc in cur.description], row))
                    d.pop("embedding", None)
                    scope = ("global" if table == "memories"
                             else project_names.get(table[4:], table[4:]))
                    if (scope, d["id"]) in existing_keys:
                        continue
                    # Report the project name (not the scope kind), like every
                    # other read path does.
                    d["_project"] = scope
                    d["_graph_weight"] = weight
                    d["_graph_source"] = entity_name
                    if relation_context:
                        d["_graph_relation_context"] = True
                    # Graph candidates are not retrieved by relevance: they have
                    # no similarity and no RRF contribution. Mark them as such
                    # instead of leaving the score fields undefined.
                    d["_similarity"] = None
                    d["_distance"] = None
                    d["_relevance_score"] = None
                    d["_rrf_score"] = 0.0
                    d["_rrf_norm"] = 0.0
                    if include_decay:
                        d["_decay_score"] = _decay_score(
                            d.get("created_at"), d.get("updated_at"),
                            d.get("access_count", 0), d.get("importance", 1.0),
                        )
                        d["_decay_norm"] = _decay_norm(
                            d.get("created_at"), d.get("updated_at"),
                            d.get("access_count", 0), d.get("importance", 1.0),
                        )
                    existing_keys.add((scope, d["id"]))
                    graph_results.append(d)

        for entity_name, weight in graph_entities:
            _collect(entity_name, weight)
        for ctx_name, weight in relation_entities:
            _collect(ctx_name, weight, relation_context=True)

        if not graph_entities:
            # Distinguish an isolated node (no neighbours) from a name that
            # has no node at all: only the latter is a meaningless expansion.
            node_exists = False
            for label in labels:
                if GraphStore(cur=cur).label_exists(label, entity):
                    node_exists = True
                    break
            if not node_exists and warnings is not None:
                scope_label = ("every scope" if all_projects
                               else (project or "the global scope"))
                warnings.append(
                    f"graph_boost: entity {entity!r} has no graph node in "
                    f"{scope_label}, so there was nothing to expand; the "
                    f"result contains query hits only (create it with relate)")
        if own_connection:
            db.commit()
        else:
            cur.execute("RELEASE SAVEPOINT graph_boost")
    except Exception as e:
        if own_connection:
            db.rollback()
        else:
            try:
                cur.execute("ROLLBACK TO SAVEPOINT graph_boost")
                cur.execute("RELEASE SAVEPOINT graph_boost")
            except Exception as rollback_error:
                logger.warning(
                    "Graph boost savepoint rollback failed: %s", rollback_error)
        logger.warning("Graph boost failed: %s", e)
        return existing_results
    finally:
        if own_connection:
            db.close()

    # 3. Fuse with existing results
    # Keep the boost relative to the best score so graph candidates cannot
    # jump to the top on weight alone.
    max_rrf = max((r.get("_rrf_score", 0.0) for r in existing_results), default=0.0)
    if max_rrf <= 0:
        max_rrf = 1.0 / 61.0

    all_results = existing_results + graph_results
    for r in all_results:
        rrf_norm = r.get("_rrf_norm")
        if rrf_norm is None:
            rrf_norm = r.get("_rrf_score", 0.0) / max_rrf
            r["_rrf_norm"] = rrf_norm
        if "_graph_weight" in r:
            r["_graph_boost"] = graph_boost_weight * r["_graph_weight"]
        graph_boost = r.get("_graph_boost", 0.0)
        # Graph candidates were not retrieved by relevance: they count as
        # relevance 0 and are lifted by the boost only.
        relevance = r.get("_relevance_score") or 0.0

        # Final score: relevance (blended with decay when asked for) plus the
        # graph boost, always inside the documented 0..1 range. Without decay
        # the blend must not shrink relevance to 70% - that made a boosted hit
        # score *lower* than the same hit without `graph_boost`.
        if include_decay:
            base = _blend_scores(relevance, r.get("_decay_norm", 0.0))
        else:
            base = relevance
        r["_final_score"] = _clamp01(base + graph_boost)

    # Query hits always rank before graph-only additions. A graph hit carries
    # no similarity to the query, so on a tie it must not push a weak but real
    # query match out of the page: it fills the remaining slots.
    query_hits = [r for r in all_results if not r.get("_graph_source")]
    graph_hits = [r for r in all_results if r.get("_graph_source")]
    query_hits.sort(key=lambda x: -x.get("_final_score", 0))
    graph_hits.sort(key=lambda x: -x.get("_final_score", 0))
    return (query_hits + graph_hits)[:limit]
