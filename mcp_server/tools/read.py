import json
import logging
from datetime import datetime, timezone
from typing import Annotated, Any, Literal

from pydantic import Field

from mcp_server.config import (
    DEFAULT_MIN_SIMILARITY,
    MEMORY_TYPE_HELP,
    SENTINEL_FORGOTTEN,
)
from mcp_server.db import get_conn
from mcp_server.envelope import STATUS_ERROR, STATUS_NOT_FOUND, STATUS_OK, fail, ok
from mcp_server.inference_client import embed
from mcp_server.projection import (
    PUBLIC_FIELDS,
    dedupe_near_duplicates,
    project_bootstrap,
    project_full,
    validate_fields,
)
from mcp_server.projection import project as project_row
from mcp_server.scopes import (
    _ag_str,
    _all_mem_tables,
    _count_entities,
    _is_global_scope,
    _memory_scopes,
    _norm_entity,
    _norm_project,
    _project_names,
    _retrieval_tables,
    _scope_key,
    _scope_label,
    _scope_of_table,
    _scopes_of_tables,
    _unknown_project_warning,
)
from mcp_server.scoring import _decay_norm
from mcp_server.search import (
    _dedup_rows_across_scopes,
    _extra_conditions,
    _search_memories,
    _search_memories_graph_boost,
)
from mcp_server.storage import (
    ACTIVE_ROW_SQL,
    _log_access,
    visibility_condition,
)
from mcp_server.tools.maintenance import _health_check

logger = logging.getLogger(__name__)

_EPOCH = datetime.min.replace(tzinfo=timezone.utc)


def _escape_like(value: str) -> str:
    """Escape the wildcards of a LIKE/ILIKE pattern."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _best_dropped_hint(best_dropped: dict[str, Any]) -> str:
    """Name the strongest vector hit that the similarity filter rejected.

    A count alone leaves the caller blind: the rejected hit may be exactly
    the memory the query was after (a natural-language question that missed
    the threshold by a hair). The hint names id, similarity and scope - and
    a content preview when it was fetched - so the caller can decide on an
    explicit retry with `min_similarity=null` without burning a discovery
    call first.
    """
    scope = best_dropped.get("scope") or "global"
    hint = (
        f"the strongest rejected vector hit was id '{best_dropped.get('id')}' "
        f"(similarity {best_dropped.get('similarity')}, scope {scope!r})"
    )
    preview = best_dropped.get("content_preview")
    if preview:
        hint += f": {preview!r}"
    return hint


def _visibility_condition(include_deleted: bool, include_superseded: bool,
                          include_expired: bool) -> str:
    """SQL filter for the orthogonal visibility flags.

    Active rows are always visible (unless expired); `include_superseded`
    adds history, `include_deleted` adds forgotten rows, and `include_expired`
    adds rows whose TTL has passed. A forgotten row stays visible through
    `include_deleted` even when it was superseded as well, exactly like the
    listing behaved before expired rows were separated out.
    """
    return visibility_condition(include_deleted, include_superseded,
                                include_expired)


def _is_expired(value: Any) -> bool:
    """True when a stored `expires_at` timestamp is in the past."""
    if not isinstance(value, datetime):
        return False
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value <= datetime.now(timezone.utc)


def _normalize_metadata_filter(value: Any) -> dict | None:
    """Accept a metadata filter as object or as JSON string.

    Some clients serialize object parameters as strings; a filter that is not
    a plain object would silently match nothing, so it is parsed here.
    """
    if value is None:
        return None
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            raise ValueError("metadata_filter must be an object, not a string")
        value = parsed
    if not isinstance(value, dict):
        raise ValueError("metadata_filter must be an object")
    return value


def recall(
    query: Annotated[
        str,
        Field(description="Search text. Non-empty; the query is embedded as a "
                          "query and matched against vector and full-text "
                          "indexes. Cannot be empty string.")],
    mem_type: Annotated[
        Literal["fact", "decision", "learning", "preference", "pattern", "incident",
                "runbook", "context", "observation", "convention", "command", "note"]
        | None,
        Field(description="Only return memories of this type. " + MEMORY_TYPE_HELP)] = None,
    entity: Annotated[
        str | None,
        Field(description="Only return memories of this entity. Required for "
                          "graph_boost if `graph_entity` is not given.")] = None,
    limit: Annotated[
        int,
        Field(description="Maximum number of results (default 10).")] = 10,
    offset: Annotated[
        int,
        Field(description="Skip this many ranked matches (default 0). Use it "
                          "with `limit` to page through a result set; the "
                          "response reports `has_more`.")] = 0,
    project: Annotated[
        str | None,
        Field(description="Project scope. Without it only global memories are "
                          "searched. '*' searches every scope at once.")] = None,
    hybrid: Annotated[
        bool,
        Field(description="True (default) fuses vector and full-text results "
                          "(RRF). False uses a single retriever.")] = True,
    include_decay: Annotated[
        bool,
        Field(description="Blend recency/access into the score (max 30% of it).")] = False,
    min_decay_score: Annotated[
        float | None,
        Field(description="Only keep memories with a normalized decay score of at "
                          "least this value (0.0-1.0). A fresh, important memory "
                          "scores ~0.9, an old one ~0.5.")] = None,
    include_superseded: Annotated[
        bool,
        Field(description="Also return superseded memories (history). Forgotten "
                          "memories stay hidden.")] = False,
    include_deleted: Annotated[
        bool,
        Field(description="Also return soft-forgotten memories (recognizable by "
                          "`forgotten: true`). Superseded history stays hidden "
                          "unless `include_superseded` is also set.")] = False,
    include_expired: Annotated[
        bool,
        Field(description="Also return memories whose TTL (`expires_at`) has "
                          "passed. They carry `expired: true` and are hidden "
                          "from every active read otherwise.")] = False,
    graph_boost: Annotated[
        bool,
        Field(description="Add memories of entities connected in the graph. "
                          "Requires `graph_entity` or `entity` to specify the "
                          "expansion point. Without an entity, graph_boost is "
                          "skipped and a warning is returned. Those hits have "
                          "no similarity and are marked with `_graph_boost`.")] = False,
    graph_entity: Annotated[
        str | None,
        Field(description="Entity to expand from when `graph_boost` is used. "
                          "With it the query search stays unfiltered, so the "
                          "result is 'query hits + graph neighbourhood'. "
                          "Without it `entity` is used for both the filter "
                          "and the expansion.")] = None,
    min_similarity: Annotated[
        float | None,
        Field(description="Drop vector hits below this cosine similarity "
                          "(0.0-1.0, default 0.40). Full-text hits are never "
                          "dropped by this filter: an exact keyword match "
                          "survives even when the embedding model scores it "
                          "low. Use null to disable the filter.")] = DEFAULT_MIN_SIMILARITY,
    fields: Annotated[
        list[str] | None,
        Field(description="Only return these columns, e.g. ['id', 'content']. "
                          "Valid names: id, content, mem_type, scope, entity, "
                          "importance, metadata, access_count, superseded_by, "
                          "created_at, updated_at, project, deleted_at, "
                          "_project, and the score annotations "
                          "_similarity, _distance, _text_score, _rrf_score, "
                          "_rrf_norm, _relevance_score, _scope_boost, "
                          "_decay_score, _decay_norm, _final_score, "
                          "_graph_boost, _graph_source, _graph_weight.")] = None,
    only_superseded: Annotated[
        bool,
        Field(description="Return only invalidated (superseded) memories.")] = False,
    include_global: Annotated[
        bool,
        Field(description="True (default) lets a project search also see global "
                          "memories; False restricts it to the project.")] = True,
    updated_after: Annotated[
        str | None,
        Field(description="Only memories changed at or after this ISO date, e.g. "
                          "'2026-10-01' or '2026-10-01T12:00:00Z'.")] = None,
    metadata_filter: Annotated[
        dict[str, Any] | None,
        Field(description="Only memories whose metadata contains these "
                          "key/value pairs. Scalar values compare tolerantly "
                          "(the string '8001' matches the number 8001); "
                          "objects and arrays use JSON containment.")] = None,
    include_relation_context: Annotated[
        bool,
        Field(description="True (default) includes the relation-context "
                          "sentences of graph edges (entity "
                          "'<from>_<relation>_<to>'); False hides them from "
                          "the ranking.")] = True,
) -> dict[str, Any]:
    """Search memories using hybrid vector + fulltext search (RRF fusion).

    Project searches also see global memories; global searches see only global ones.
    Set graph_boost=True to add the memories of entities connected in the graph.
    `graph_entity` names the entity to expand from and leaves the query search
    unfiltered ("query hits + graph neighbourhood"); without it `entity` is
    used for the expansion *and* filters the search, so unrelated hits are
    dropped. Graph hits have no similarity: their `_similarity` and
    `_relevance_score` are null and they carry `_graph_boost` instead.
    `min_similarity` defaults to 0.40 and only applies to the vector retriever;
    a full-text hit is kept even when its embedding is a poor match, so an
    exact identifier ("ZEBRAFISH_TOKEN") is found even if the embedding model
    handles that word badly. Pass `None` to disable the filter. The response
    reports `candidates` and `dropped_by_similarity`, and warns when the
    filter removed everything. When hits were dropped, it also names
    `best_dropped_by_similarity` (id, similarity, scope, content preview):
    the strongest rejected vector hit, so a natural-language question that
    missed the threshold by a hair is visible and one retry with
    `min_similarity=null` can recover it.
    `_relevance_score` is normalized to 0..1 and combines both signals: the
    cosine similarity of the vector hit and the rank of the full-text hit.
    A memory that both retrievers found scores higher than either signal
    alone. `_similarity` is the raw cosine similarity (0..1): a natural
    language question that matches scores ~0.6-0.95, a bare keyword that
    occurs verbatim in the memory only ~0.3-0.6.
    `_scope_boost` (0.05) is added to `_relevance_score` for a project row,
    so a project memory outranks an equally relevant global one.
    `_final_score` is the ranking score in 0..1: relevance blended with decay
    plus the graph boost. It is only present with `include_decay=True` or
    `graph_boost=True`; without them the ranking is `_relevance_score`.
    Set include_decay=True to blend in recency/frequency (max 30% of the score).
    min_decay_score (0..1) filters on the normalized decay score; a fresh,
    important memory scores ~0.9, an old one ~0.5.
    Set only_superseded=True to return only invalidated facts.
    Set fields=["id", "content"] to receive only the named columns.
    Set include_global=False to restrict a project search to project memories.
    `updated_after` and `metadata_filter` narrow the result set.
    include_relation_context=False hides the relation-context sentences of
    the graph edges from the ranking (they repeat what the edge already says).
    `offset` pages through the ranked results (with `limit`); the response
    reports `has_more` and `next_offset`.
    `project="*"` searches every scope at once. A project that does not
    exist is reported in `warnings`: the search then only covered the global
    scope.
    """
    logger.info("recall query='%s' entity=%s hybrid=%s include_decay=%s graph_boost=%s min_similarity=%s",
                query[:80], entity, hybrid, include_decay, graph_boost, min_similarity)
    # "*" means every scope. It is resolved before the project name is
    # normalized, otherwise it would become a project called "*".
    # The raw value tells explicit scope intent apart from the default: an
    # explicitly requested single scope must not produce a warning that
    # suggests searching the other scopes.
    raw_project = project
    all_projects = project == "*"
    try:
        project = None if all_projects else _norm_project(project)
        entity = _norm_entity(entity)
        graph_entity = _norm_entity(graph_entity)
    except ValueError as e:
        return fail(STATUS_ERROR, str(e))
    explicit_scope = bool(raw_project and str(raw_project).strip())
    warnings: list[str] = []
    if not query or not query.strip():
        return fail(STATUS_ERROR, "query must not be empty")
    if min_similarity is not None and not 0.0 <= min_similarity <= 1.0:
        return fail(STATUS_ERROR, "min_similarity must be between 0.0 and 1.0")
    fields_error = validate_fields(fields)
    if fields_error:
        return fail(STATUS_ERROR, fields_error)
    if min_decay_score is not None and not 0.0 <= min_decay_score <= 1.0:
        return fail(STATUS_ERROR, "min_decay_score must be between 0.0 and 1.0")
    if limit < 1:
        return fail(STATUS_ERROR, "limit must be >= 1")
    if offset < 0:
        return fail(STATUS_ERROR, "offset must be >= 0")
    if metadata_filter is not None and not isinstance(metadata_filter, dict):
        try:
            metadata_filter = _normalize_metadata_filter(metadata_filter)
        except ValueError as e:
            return fail(STATUS_ERROR, str(e))
    try:
        _extra_conditions(updated_after, metadata_filter)
    except ValueError as e:
        return fail(STATUS_ERROR, str(e))
    # The graph expansion needs a starting entity. `graph_entity` names it
    # explicitly, so the query search can stay unfiltered; `entity` is the
    # fallback and then also filters the search.
    boost_entity = graph_entity or entity
    if graph_boost and not boost_entity:
        warnings.append("graph_boost requires an entity (graph_entity or "
                        "entity); no graph boost was applied. Pass an entity "
                        "parameter to enable graph expansion.")
    if only_superseded:
        include_superseded = True
    query_emb = embed([query], mode="query")[0]
    conn = get_conn()
    try:
        cur = conn.cursor()
        # A scope that does not exist must not look like an empty scope: the
        # search silently falls back to the global table.
        if not all_projects:
            unknown = _unknown_project_warning(cur, project)
            if unknown:
                warnings.append(unknown)

        # Neue RRF-Funktion nutzen
        tables = (_all_mem_tables(cur) if all_projects
                  else _retrieval_tables(cur, project))
        if (project or all_projects) and not include_global:
            tables = [table for table in tables if table != "memories"]
        # The scopes that were really searched. An agent that finds nothing
        # must be able to see whether other scopes were covered - otherwise
        # "no result" is indistinguishable from "wrong scope".
        scopes_searched = _scopes_of_tables(cur, tables)
        other_scopes = [s for s in _scopes_of_tables(cur, _all_mem_tables(cur))
                        if s not in scopes_searched]

        stats: dict[str, Any] = {}
        # The page needs `offset + limit` candidates, plus one to know whether
        # a further page exists. Slicing happens after the optional filters so
        # `offset` counts real returned rows.
        fetch_limit = limit + offset + 1
        results = _search_memories(
            cur, query_emb, query if (hybrid or query_emb is None) else None,
            mem_type, entity, fetch_limit, project,
            include_decay, include_superseded=include_superseded,
            min_similarity=min_similarity,
            hybrid=hybrid, tables=tables,
            updated_after=updated_after, metadata_filter=metadata_filter,
            exclude_relation_context=not include_relation_context,
            stats=stats, include_expired=include_expired,
            include_deleted=include_deleted,
        )

        # Graph-enhanced retrieval when enabled
        if graph_boost and boost_entity:
            results = _search_memories_graph_boost(
                results, boost_entity, fetch_limit, project, include_decay,
                include_global=include_global,
                updated_after=updated_after, metadata_filter=metadata_filter,
                exclude_relation_context=not include_relation_context,
                all_projects=all_projects,
                include_expired=include_expired,
                include_deleted=include_deleted,
                warnings=warnings,
            )
            if results and all(r.get("_relevance_score") is None
                               for r in results if isinstance(r, dict)):
                warnings.append(
                    "no query match found: the results come from the graph "
                    "neighbourhood of the entity, not from the query"
                )

        # Filter on decay *before* the access counters are bumped: the filter
        # must see the stored access_count, not the one of this very query.
        dropped_by_decay = 0
        if min_decay_score is not None:
            kept = []
            for r in results:
                if not isinstance(r, dict) or "error" in r:
                    kept.append(r)
                    continue
                norm = r.get("_decay_norm")
                if norm is None:
                    norm = _decay_norm(
                        r.get("created_at"), r.get("updated_at"),
                        r.get("access_count") or 0, r.get("importance") or 1.0,
                    )
                    r["_decay_norm"] = norm
                # A fresh, important memory scores exactly 0.9 up to rounding,
                # so the comparison needs a small tolerance.
                if norm >= min_decay_score - 1e-6:
                    kept.append(r)
                else:
                    dropped_by_decay += 1
            results = kept

        dropped_by_superseded = 0
        if only_superseded:
            kept = []
            for r in results:
                if isinstance(r, dict) and r.get("superseded_by"):
                    kept.append(r)
                else:
                    dropped_by_superseded += 1
            results = kept

        has_more = len(results) > offset + limit
        results = results[offset:offset + limit]

        # Update access counters (decay frequency) only for the rows this
        # page actually returns: skipped matches were not read.
        hit_ids = [r.get("id") for r in results if isinstance(r, dict) and r.get("id")]
        _log_access(cur, hit_ids, project, tables=tables)
        for r in results:
            if isinstance(r, dict) and r.get("id") in hit_ids and "error" not in r:
                r["access_count"] = (r.get("access_count") or 0) + 1

        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.exception("recall failed")
        return fail(STATUS_ERROR, str(e))
    finally:
        conn.close()

    rows = []
    expired_count = 0
    for r in results:
        if not isinstance(r, dict):
            continue
        item = project_row(r, fields)
        if r.get("deleted_at"):
            item["forgotten"] = True
            if item.get("superseded_by") == SENTINEL_FORGOTTEN:
                item["superseded_by"] = None
        if _is_expired(r.get("expires_at")):
            expired_count += 1
            if fields is None or "expired" in fields:
                item["expired"] = True
        rows.append(item)
    if expired_count:
        warnings.append(
            f"{expired_count} result(s) have an expired TTL: they are not "
            f"active memories anymore")

    # An empty result set is ambiguous: the memory may simply not exist, or
    # a filter may have dropped it. Name *every* filter that removed
    # something - naming only the first one sent the caller down the wrong
    # path (a decay-filtered hit was reported as "below min_similarity").
    filtered_out = stats.get("dropped_by_similarity", 0)
    dropped_by_coverage = stats.get("dropped_by_coverage", 0)
    dropped_by_cross_scope = stats.get("dropped_by_cross_scope_duplicate", 0)
    candidates = stats.get("candidates", 0)
    best_dropped = stats.get("best_dropped_by_similarity")
    if not rows and offset > 0 and candidates:
        warnings.append(
            f"no results on this page: offset={offset} is beyond the "
            f"{candidates} ranked candidate(s); lower the offset"
        )
    elif not rows:
        reasons: list[str] = []
        if filtered_out:
            reasons.append(
                f"{filtered_out} vector hit(s) were below "
                f"min_similarity={min_similarity} (pass min_similarity=null "
                f"or a lower value to see them)")
            if best_dropped:
                reasons.append(_best_dropped_hint(best_dropped))
        if dropped_by_coverage:
            reasons.append(
                f"{dropped_by_coverage} full-text hit(s) covered too little of "
                f"the query (below {stats.get('min_coverage')} of the query "
                f"words) to be relevant")
        if dropped_by_decay:
            reasons.append(
                f"{dropped_by_decay} hit(s) were below "
                f"min_decay_score={min_decay_score}")
        if dropped_by_superseded:
            reasons.append(
                f"{dropped_by_superseded} hit(s) were not superseded "
                f"(only_superseded=true)")
        if dropped_by_cross_scope:
            reasons.append(
                f"{dropped_by_cross_scope} active hit(s) were identical copies "
                f"in another scope and were deduplicated")
        if metadata_filter:
            reasons.append(
                f"metadata_filter={metadata_filter!r} may have excluded the "
                f"matching rows")
        if updated_after:
            reasons.append(
                f"updated_after={updated_after!r} may have excluded the "
                f"matching rows")
        if reasons:
            warnings.append("no results: " + "; ".join(reasons))
        elif candidates:
            warnings.append(
                f"no results: {candidates} hit(s) matched but every one of "
                f"them was filtered out (mem_type, entity, the scope filter "
                f"or the page offset)"
            )
        else:
            warnings.append("no results: nothing in this scope matches the query")
        # Scope transparency: a caller who never named a project (and did not
        # switch globals off) must learn that other scopes exist. An
        # explicitly requested scope is no warning: the isolation was the
        # intent, and "use project='*'" would contradict it.
        if other_scopes and not explicit_scope and include_global:
            warnings.append(
                f"only the scope(s) {', '.join(scopes_searched)} were searched; "
                f"other scopes exist ({', '.join(other_scopes)}) - use "
                f"project='*' to search them as well"
            )
    if min_similarity is None and rows:
        weak = [r for r in rows
                if (r.get("_similarity") is not None
                    and r["_similarity"] < DEFAULT_MIN_SIMILARITY)]
        if weak:
            warnings.append(
                f"the min_similarity filter is disabled: {len(weak)} of "
                f"{len(rows)} result(s) have _similarity < "
                f"{DEFAULT_MIN_SIMILARITY} and may be only marginally "
                f"relevant to the query")
    payload: dict[str, Any] = {
        # `candidates` counts the hits that made it into the ranking.
        "candidates": candidates,
        "dropped_by_similarity": filtered_out,
        "offset": offset,
        "has_more": has_more,
        "next_offset": (offset + limit) if has_more else None,
        # The scopes that were actually searched: an agent that finds
        # nothing has to know whether other scopes were covered.
        "scopes_searched": scopes_searched,
    }
    # The strongest rejected vector hit: a count alone hides the natural
    # language question that barely missed the threshold. With a non-empty
    # result it is a data field; with none it is the reason list.
    if best_dropped:
        payload["best_dropped_by_similarity"] = dict(best_dropped)
    if dropped_by_coverage:
        payload["dropped_by_coverage"] = dropped_by_coverage
    # Filter counters are reported even when hits survive: an agent must be
    # able to see that the decay or superseded filter removed candidates
    # from a non-empty result instead of silently getting fewer hits.
    if dropped_by_decay:
        payload["dropped_by_decay"] = dropped_by_decay
    if dropped_by_superseded:
        payload["dropped_by_superseded"] = dropped_by_superseded
    if stats.get("fulltext_only"):
        payload["fulltext_only"] = stats["fulltext_only"]
    if dropped_by_cross_scope:
        # Silent dedup made a cross-scope copy look like a missing memory:
        # report the count so the caller can retry inside the other scope.
        payload["dropped_by_cross_scope_duplicate"] = dropped_by_cross_scope
        if rows:
            warnings.append(
                f"{dropped_by_cross_scope} active hit(s) were hidden as "
                f"identical copies in another scope; search that scope "
                f"directly to see it")
    return ok(STATUS_OK, count=len(rows), results=rows, warnings=warnings,
              **payload)


def list_memories(
    entity: Annotated[
        str | None,
        Field(description="Only list memories of this entity.")] = None,
    mem_type: Annotated[
        Literal["fact", "decision", "learning", "preference", "pattern", "incident",
                "runbook", "context", "observation", "convention", "command", "note"]
        | None,
        Field(description="Only list memories of this type. " + MEMORY_TYPE_HELP)] = None,
    limit: Annotated[
        int,
        Field(description="Maximum number of results (default 20).")] = 20,
    project: Annotated[
        str | None,
        Field(description="Project scope. Without it (or with '*') every "
                          "scope is listed; 'global' lists only the shared "
                          "scope.")] = None,
    include_superseded: Annotated[
        bool,
        Field(description="Also list superseded memories (history).")] = False,
    fields: Annotated[
        list[str] | None,
        Field(description="Only return these columns, e.g. ['id', 'content'].")] = None,
    include_global: Annotated[
        bool,
        Field(description="True (default) lists global memories as well; False "
                          "restricts a project browse to project memories.")] = True,
    updated_after: Annotated[
        str | None,
        Field(description="Only memories changed at or after this ISO date, e.g. "
                          "'2026-10-01'.")] = None,
    metadata_filter: Annotated[
        dict[str, Any] | None,
        Field(description="Only memories whose metadata contains these "
                          "key/value pairs. Scalar values compare tolerantly "
                          "(the string '8001' matches the number 8001); "
                          "objects and arrays use JSON containment.")] = None,
    order_by: Annotated[
        Literal["updated_at", "created_at", "access_count", "importance"],
        Field(description="Sort order (default: updated_at, newest first).")] = "updated_at",
    contains: Annotated[
        str | None,
        Field(description="Only memories whose content contains this text "
                          "(case-insensitive substring). Use it to look for a "
                          "literal string without the semantic ranking of "
                          "`recall`.")] = None,
    offset: Annotated[
        int,
        Field(description="Skip this many matches (default 0). Use it together "
                          "with `limit` to page through a large scope; the "
                          "response reports `total` and `has_more`.")] = 0,
    memory_id: Annotated[
        str | None,
        Field(description="Direct id lookup: returns the single memory as "
                          "`memory` (plus `results`) with its history "
                          "markers. Without `project` (or with '*') every "
                          "scope is searched; an id in several scopes is "
                          "refused. Browse filters are ignored for this "
                          "lookup.")] = None,
    include_deleted: Annotated[
        bool,
        Field(description="Also list soft-forgotten memories (recognizable by "
                          "a non-null `deleted_at`). Use it before "
                          "`update_memory(restore=true)`. Superseded history "
                          "stays hidden unless `include_superseded` is also "
                          "set.")] = False,
    include_expired: Annotated[
        bool,
        Field(description="Also list memories whose TTL (`expires_at`) has "
                          "passed. They carry `expired: true` and are hidden "
                          "from every active read otherwise. This is separate "
                          "from `include_deleted` (soft-forgotten rows).")] = False,
    include_relation_context: Annotated[
        bool,
        Field(description="True (default) also lists the relation-context "
                          "sentences of the graph edges; False hides them. "
                          "They are stored under the synthetic entity "
                          "'<from>_<relation>_<to>' and repeat what the edge "
                          "already says.")] = True,
    include_duplicates: Annotated[
        bool,
        Field(description="False (default) lists one active copy of identical "
                          "content even when it exists in several scopes. "
                          "True lists every copy, so a browse across scopes "
                          "shows each scope's row.")] = False,
) -> dict[str, Any]:
    """Browse stored memories without a search query.

    Newest first. Project scope also lists global memories, like `recall`.
    Without `project` (or with `project="*"`) every scope is listed;
    `project="global"` lists only the shared scope. Set `fields` to narrow
    the returned columns. Set
    `include_global=False` to restrict a project browse to project memories.
    `updated_after` and `metadata_filter` narrow the result set, `order_by`
    changes the sorting and `contains` filters on a literal substring of the
    content. `offset` pages through a scope that has more matches than
    `limit`. Pass `memory_id` for a direct id lookup: the answer then carries
    the single `memory` (and a one-item `results`) with the
    `superseded_by`/`forgotten`/`expired` markers, and a hidden row names the
    flag that reveals it. Browse filters are ignored for that lookup.
    `include_deleted=True` includes soft-forgotten memories (they carry
    `deleted_at`); superseded history stays hidden unless
    `include_superseded=True` is also passed — the two flags are orthogonal.
    Expired TTL rows are separate: they stay hidden until
    `include_expired=True` and then carry `expired: true`.
    `include_relation_context=False` hides the relation-context sentences of
    the graph edges. A project that does not exist is reported in `warnings`:
    the browse then only covered the global scope.
    Identical content that exists in several scopes is listed once; the answer
    says in `warnings` how many copies were hidden. Pass
    `include_duplicates=True` to see every copy.
    """
    logger.info("list_memories entity=%s mem_type=%s limit=%s offset=%s project=%s",
                entity, mem_type, limit, offset, project)
    raw_project = project
    all_projects = project == "*"
    try:
        project = None if all_projects else _norm_project(project)
        entity = _norm_entity(entity)
    except ValueError as e:
        return fail(STATUS_ERROR, str(e))
    global_only = not all_projects and _is_global_scope(raw_project)
    if limit < 1:
        return fail(STATUS_ERROR, "limit must be >= 1")
    if offset < 0:
        return fail(STATUS_ERROR, "offset must be >= 0")
    fields_error = validate_fields(fields)
    if fields_error:
        return fail(STATUS_ERROR, fields_error)
    if metadata_filter is not None:
        try:
            metadata_filter = _normalize_metadata_filter(metadata_filter)
        except ValueError as e:
            return fail(STATUS_ERROR, str(e))
    try:
        extra_conditions, extra_params = _extra_conditions(updated_after,
                                                            metadata_filter)
    except ValueError as e:
        return fail(STATUS_ERROR, str(e))
    order_column = {
        "updated_at": "updated_at",
        "created_at": "created_at",
        "access_count": "access_count",
        "importance": "importance",
    }.get(order_by)
    if not order_column:
        return fail(STATUS_ERROR,
                    "order_by must be one of updated_at, created_at, "
                    "access_count, importance")
    if memory_id:
        # Direct id lookup: one memory, its history state and the reason it is
        # hidden. The browse filters (entity, contains, ...) would contradict
        # "exactly this id", so they are reported as ignored instead of
        # silently narrowing or widening the answer.
        ignored = [
            name for name, value in (
                ("entity", entity), ("mem_type", mem_type),
                ("contains", contains), ("updated_after", updated_after),
                ("metadata_filter", metadata_filter),
            ) if value is not None
        ]
        if offset:
            ignored.append("offset")
        if order_by != "updated_at":
            ignored.append("order_by")
        result = _get_memory(
            memory_id=memory_id, project=raw_project,
            include_superseded=include_superseded,
            include_deleted=include_deleted,
            include_expired=include_expired, fields=fields,
            include_global=include_global)
        if result.get("ok"):
            result["count"] = 1
            result["results"] = [result["memory"]]
        if ignored:
            result.setdefault("warnings", []).append(
                "memory_id performs a direct id lookup: "
                f"{', '.join(ignored)} ignored")
        return result
    conn = get_conn()
    warnings: list[str] = []
    try:
        cur = conn.cursor()
        unknown = _unknown_project_warning(cur, project)
        if unknown:
            warnings.append(unknown)
        if all_projects:
            tables = _all_mem_tables(cur)
        elif project:
            tables = _retrieval_tables(cur, project)
        elif global_only:
            # The caller asked for the shared scope explicitly: only that one.
            tables = ["memories"]
        else:
            tables = _all_mem_tables(cur)
        if (project or all_projects) and not include_global:
            tables = [table for table in tables if table != "memories"]
        conditions: list[str] = [
            _visibility_condition(include_deleted, include_superseded,
                                  include_expired)]
        params: list[Any] = []
        if not include_relation_context:
            # The relation-context sentences live under a synthetic entity and
            # repeat what the edge already says.
            conditions.append(
                "COALESCE(metadata->>'relation_context', 'false') <> 'true'")
        if memory_id:
            conditions.append("id = %s")
            params.append(memory_id)
        if entity:
            conditions.append("entity = %s")
            params.append(entity)
        if mem_type:
            conditions.append("mem_type = %s")
            params.append(mem_type)
        if contains:
            conditions.append("content ILIKE %s")
            params.append(f"%{_escape_like(contains)}%")
        conditions.extend(extra_conditions)
        params.extend(extra_params)
        where = " AND ".join(conditions)
        # Two phases. First the light columns of every match: dedup and paging
        # must see the whole set, otherwise `total`, `has_more` and the page
        # disagree. Only then the full rows of this page are loaded, so a
        # large scope does not ship every embedding vector.
        index_rows: list[dict] = []
        for table in tables:
            project_names = None if table == "memories" else _project_names(cur)
            scope = ("global" if table == "memories"
                     else project_names.get(table[4:], table[4:]))
            cur.execute(
                f"SELECT id, content_hash, superseded_by, deleted_at, "
                f"{order_column} AS _order FROM {table} WHERE {where}",
                params,
            )
            for (row_id, c_hash, superseded_by,
                 deleted_at, order_value) in cur.fetchall():
                index_rows.append({
                    "id": row_id,
                    "content_hash": c_hash,
                    "superseded_by": superseded_by,
                    "deleted_at": deleted_at,
                    "_order": order_value,
                    "_project": scope,
                    "_table": table,
                })
        conn.commit()
    except Exception as e:
        conn.rollback()
        return fail(STATUS_ERROR, str(e))
    finally:
        conn.close()
    unique, hidden_duplicates = _dedup_rows_across_scopes(
        index_rows, include_duplicates)
    if hidden_duplicates:
        warnings.append(
            f"{hidden_duplicates} active memory/memories were hidden because "
            f"identical content exists in another scope; only one copy is "
            f"listed. Pass include_duplicates=true to list every copy")
    unique.sort(key=lambda r: (r.get("_order") is not None,
                                r.get("_order"),
                                str(r.get("id") or "")), reverse=True)
    total = len(unique)
    window_index = unique[offset:offset + limit]
    has_more = total > offset + limit
    # Full rows for the page only, then restore the index order.
    page: dict[tuple[str, str], dict] = {}
    if window_index:
        scope_by_key = {(r["_table"], r["id"]): r["_project"]
                        for r in window_index}
        by_table: dict[str, list[str]] = {}
        for r in window_index:
            by_table.setdefault(r["_table"], []).append(r["id"])
        conn = get_conn()
        try:
            cur = conn.cursor()
            for table, ids in by_table.items():
                cur.execute(f"SELECT * FROM {table} WHERE id = ANY(%s)", (ids,))
                columns = [d[0] for d in cur.description]
                for row in cur.fetchall():
                    item = dict(zip(columns, row))
                    item["_project"] = scope_by_key[(table, item["id"])]
                    page[(table, item["id"])] = item
            conn.commit()
        except Exception as e:
            conn.rollback()
            return fail(STATUS_ERROR, str(e))
        finally:
            conn.close()
    window = [page[(r["_table"], r["id"])]
              for r in window_index if (r["_table"], r["id"]) in page]
    if offset > 0 and total > 0 and not window:
        warnings.append(
            f"no results on this page: offset={offset} is beyond the "
            f"{total} matching memories; lower the offset")
    projection_fields = fields
    if include_deleted:
        # Without this the caller could not tell a forgotten row from an
        # active one.
        if projection_fields is None:
            projection_fields = list(PUBLIC_FIELDS) + ["deleted_at"]
        elif "deleted_at" not in projection_fields:
            projection_fields = projection_fields + ["deleted_at"]
    results = []
    expired_count = 0
    for row in window:
        item = project_row(row, projection_fields)
        # A machine-readable marker beats reading the '__forgotten__'
        # sentinel out of superseded_by.
        if row.get("deleted_at"):
            item["forgotten"] = True
            if item.get("superseded_by") == SENTINEL_FORGOTTEN:
                item["superseded_by"] = None
        elif fields and "forgotten" in fields:
            # Requested explicitly: report `false` instead of omitting the
            # key, so the shape stays stable for the caller.
            item["forgotten"] = False
        if _is_expired(row.get("expires_at")):
            expired_count += 1
            if projection_fields is None or (fields and "expired" in fields):
                item["expired"] = True
        elif fields and "expired" in fields:
            item["expired"] = False
        results.append(item)
    if expired_count:
        warnings.append(
            f"{expired_count} result(s) have an expired TTL: they are not "
            f"active memories anymore")
    return ok(STATUS_OK, count=len(results), total=total, offset=offset,
              has_more=has_more,
              next_offset=(offset + limit) if has_more else None,
              duplicates_hidden=hidden_duplicates,
              warnings=warnings, results=results)


def _get_memory(
    memory_id: Annotated[
        str,
        Field(description="Id of the memory to fetch.")],
    project: Annotated[
        str | None,
        Field(description="Scope to read. Without it the id is looked up in "
                          "every scope; 'global' reads only the shared "
                          "scope.")] = None,
    include_superseded: Annotated[
        bool,
        Field(description="Also return a superseded memory.")] = False,
    include_deleted: Annotated[
        bool,
        Field(description="Also return a soft-forgotten memory.")] = False,
    include_expired: Annotated[
        bool,
        Field(description="Also return a memory whose TTL has passed.")] = False,
    fields: Annotated[
        list[str] | None,
        Field(description="Only return these columns, e.g. ['id', 'content'].")] = None,
    include_global: bool = True,
) -> dict[str, Any]:
    """Fetch exactly one memory by id, including its history state.

    This is the singular path of `list_memories(memory_id=...)`: the answer
    names the scope and sets the `superseded_by`/`forgotten`/`expired`
    markers, so an agent can tell *why* a known id was missing from a search.
    Without `project` (or with `project="*"`) the id is looked up in every
    scope; an id that exists in several scopes is refused and the scopes are
    named.
    """
    logger.info("get_memory id=%s project=%s", memory_id, project)
    memory_id = (memory_id or "").strip()
    if not memory_id:
        return fail(STATUS_ERROR, "memory_id must not be empty")
    raw_project = project
    if raw_project is not None and str(raw_project).strip() == "*":
        raw_project = None
    global_only = _is_global_scope(raw_project)
    try:
        project = _norm_project(raw_project)
    except ValueError as e:
        return fail(STATUS_ERROR, str(e))
    fields_error = validate_fields(fields)
    if fields_error:
        return fail(STATUS_ERROR, fields_error)
    conn = get_conn()
    try:
        cur = conn.cursor()
        if global_only:
            tables = ["memories"]
        elif project:
            tables = _retrieval_tables(cur, project)
        else:
            scopes = _memory_scopes(cur, memory_id)
            if len(scopes) > 1:
                conn.rollback()
                return fail(
                    STATUS_ERROR,
                    f"ambiguous memory id {memory_id}: it exists in the "
                    f"scopes {scopes}; pass project=<name> to say which one "
                    f"to read", id=memory_id, scopes=scopes)
            tables = _all_mem_tables(cur)
        if not include_global:
            tables = [table for table in tables if table != "memories"]
        where = _visibility_condition(include_deleted, include_superseded,
                                      include_expired)
        found = None
        found_table = None
        for table in tables:
            cur.execute(f"SELECT * FROM {table} WHERE id = %s AND {where}",
                        (memory_id,))
            row = cur.fetchone()
            if row is None:
                continue
            columns = [d[0] for d in cur.description]
            found = dict(zip(columns, row))
            found_table = table
            break
        if found is None:
            # Explain the miss: the row may exist but be hidden by one of the
            # visibility flags. A bare "not found" would make an agent believe
            # the id never existed.
            hidden = None
            for table in tables:
                cur.execute(f"SELECT superseded_by, deleted_at, expires_at "
                            f"FROM {table} WHERE id = %s", (memory_id,))
                row = cur.fetchone()
                if row:
                    hidden = (table, row[0], row[1], row[2])
                    break
            conn.rollback()
            if hidden:
                table, superseded_by, deleted_at, expires_at = hidden
                reasons = []
                if deleted_at is not None:
                    reasons.append("soft-forgotten (pass include_deleted=true)")
                if superseded_by is not None and superseded_by != SENTINEL_FORGOTTEN:
                    reasons.append("superseded (pass include_superseded=true)")
                if _is_expired(expires_at):
                    reasons.append("expired (pass include_expired=true)")
                return fail(
                    STATUS_NOT_FOUND,
                    f"memory {memory_id} exists but is not visible: "
                    + ("; ".join(reasons) if reasons else "hidden by a filter"),
                    id=memory_id, scope=_scope_of_table(cur, table))
            return fail(STATUS_NOT_FOUND, f"NOT FOUND: {memory_id}",
                        id=memory_id)
        scope = _scope_of_table(cur, found_table)
        found["_project"] = scope
        projection_fields = fields
        if include_deleted:
            # Without this the caller could not tell a forgotten row from an
            # active one (same rule as the collection path).
            if projection_fields is None:
                projection_fields = list(PUBLIC_FIELDS) + ["deleted_at"]
            elif "deleted_at" not in projection_fields:
                projection_fields = projection_fields + ["deleted_at"]
        item = project_row(found, projection_fields)
        if found.get("deleted_at"):
            item["forgotten"] = True
            if item.get("superseded_by") == SENTINEL_FORGOTTEN:
                item["superseded_by"] = None
        expired = _is_expired(found.get("expires_at"))
        if expired and (fields is None or "expired" in fields):
            item["expired"] = True
        conn.commit()
    except Exception as e:
        conn.rollback()
        return fail(STATUS_ERROR, str(e), id=memory_id)
    finally:
        conn.close()
    return ok(STATUS_OK, id=memory_id, project=scope, memory=item)


def _relation_context_memories(cur, target: str, tables: list[str],
                               include_superseded: bool,
                               include_expired: bool = False) -> list[dict]:
    """Relation-context sentences attached to the edges of an entity.

    `relate(context=...)` stores the sentence under the synthetic entity
    `A_REL_B`, so a lookup of `A` or `B` never showed it. The sentence
    belongs to the edge and the edge belongs to the entity, so it is
    returned here - with its metadata (`relation_context`, `from`, `to`,
    `weight`) that tells the agent where it comes from.
    """
    rows: list[dict] = []
    names = _project_names(cur)
    for table in tables:
        conditions = [
            "entity IS NOT NULL",
            "metadata->>'relation_context' = 'true'",
            "(metadata->>'from' = %s OR metadata->>'to' = %s)",
        ]
        params: list[Any] = [target, target]
        conditions.append(visibility_condition(
            include_superseded=include_superseded,
            include_expired=include_expired,
        ))
        cur.execute(
            f"SELECT * FROM {table} WHERE {' AND '.join(conditions)}",
            params,
        )
        columns = [d[0] for d in cur.description]
        for row in cur.fetchall():
            item = dict(zip(columns, row))
            item.pop("embedding", None)
            item["_project"] = ("global" if table == "memories"
                                 else names.get(table[4:], table[4:]))
            rows.append(item)
    return rows


def _count_entity_memories(cur, target: str, tables: list[str],
                           include_superseded: bool,
                           include_expired: bool = False) -> int:
    """Number of memories that belong to an entity, across the given tables.

    The count is taken over the union of the tables, so a memory that exists
    in the project *and* in the global scope is counted once - exactly like
    the listing itself, which deduplicates by content.
    """
    if not tables:
        return 0
    where = ("entity = %s AND " + visibility_condition(
        include_superseded=include_superseded,
        include_expired=include_expired))
    parts = [
        f"SELECT COALESCE(content_hash, id) AS key FROM {table} WHERE {where}"
        for table in tables
    ]
    cur.execute(
        "SELECT COUNT(*) FROM (" + " UNION ".join(parts) + ") AS u",
        (target,) * len(parts),
    )
    return cur.fetchone()[0]


def get_entity(name: Annotated[
        str | None,
        Field(description="Entity to look up. Alias: `entity`.")] = None,
    project: Annotated[
        str | None,
        Field(description="Project scope. Without it only the global scope is "
                          "used.")] = None,
    include_superseded: Annotated[
        bool,
        Field(description="Also include superseded memories.")] = False,
    include_expired: Annotated[
        bool,
        Field(description="Also include memories whose TTL has passed. They "
                          "carry `expired: true`; without this flag an expired "
                          "row stays hidden even when "
                          "`include_superseded` is set.")] = False,
    entity: Annotated[
        str | None,
        Field(description="Entity to look up. Alias: `name`.")] = None,
    fields: Annotated[
        list[str] | None,
        Field(description="Only return these memory columns, e.g. "
                          "['id', 'content'].")] = None,
    include_global: Annotated[
        bool,
        Field(description="True (default) includes global memories; False "
                          "restricts a project lookup to project memories.")] = True,
    limit: Annotated[
        int,
        Field(description="Maximum number of memories to return (default 100). "
                          "The response reports `total` and `has_more`.")] = 100,
    offset: Annotated[
        int,
        Field(description="Skip this many memories (default 0). Use it with "
                          "`limit` to page through an entity with many "
                          "memories.")] = 0,
) -> dict[str, Any]:
    """Get everything known about an entity, including all linked memories.

    The entity can be passed as `entity` or `name`.
    Memory-only entities have no graph node; `details.type` is `memory-only`.
    A graph node's `details.type` is `graph` with the raw AGE label of its
    scope in `details.label`.
    Set fields=["id", "content"] to receive only the named memory columns.
    Set include_global=False to restrict a project lookup to project memories.
    The relation-context sentences of the entity's edges are included, even
    though they are stored under the synthetic entity `<from>_<rel>_<to>`;
    `total` counts them too.
    The memories are paged with `limit`/`offset`; `total` counts them all, so
    a truncated answer is visible instead of looking complete.
    `project="*"` is rejected: the answer describes exactly one entity.
    """
    if entity and name and entity.strip() != name.strip():
        return fail(
            STATUS_ERROR,
            f"entity and name were both given but differ ({entity!r} vs "
            f"{name!r}); pass only one of the aliases")
    target = entity or name
    logger.info("get_entity name=%s", target)
    if project == "*":
        return fail(
            STATUS_ERROR,
            "project='*' is not supported by get_entity: the answer describes "
            "exactly one entity. Omit `project` for the global scope, pass a "
            "project name, or use list_entities to browse every scope",
        )
    try:
        project = _norm_project(project)
        target = _norm_entity(target)
    except ValueError as e:
        return fail(STATUS_ERROR, str(e))
    if not target:
        return fail(STATUS_ERROR, "entity is required")
    if limit < 1:
        return fail(STATUS_ERROR, "limit must be >= 1")
    if offset < 0:
        return fail(STATUS_ERROR, "offset must be >= 0")
    fields_error = validate_fields(fields)
    if fields_error:
        return fail(STATUS_ERROR, fields_error)
    conn = get_conn()
    memories = []
    entity_data = None
    warnings: list[str] = []
    total = 0
    relation_context_count = 0
    try:
        cur = conn.cursor()
        tables = _retrieval_tables(cur, project) if project else None
        if project and not include_global:
            tables = [table for table in tables if table != "memories"]
        used_tables = tables or ["memories"]
        memories = _search_memories(cur, None, None, None, target,
                                    limit + offset + 1, project,
                                    include_superseded=include_superseded,
                                    include_expired=include_expired,
                                    tables=tables)
        total = _count_entity_memories(cur, target, used_tables,
                                       include_superseded, include_expired)
        # The context sentence of an edge belongs to both of its endpoints.
        known_ids = {m.get("id") for m in memories}
        for row in _relation_context_memories(
            cur, target, used_tables, include_superseded, include_expired
        ):
            if row.get("id") in known_ids:
                continue
            known_ids.add(row.get("id"))
            memories.append(row)
            total += 1
            relation_context_count += 1

        scope = _scope_key(project)
        cur.execute(
            "SELECT id, name, type, project FROM entities WHERE name = %s AND project = %s",
            (target, scope),
        )
        row = cur.fetchone()
        if not row:
            label = _scope_label(project)
            name_json = json.dumps(target)
            cur.execute("SET search_path TO ag_catalog, public")
            cur.execute(
                f"SELECT * FROM cypher('memory_graph', $$ "
                f"MATCH (n:{label} {{name: {name_json}}}) RETURN n $$) AS (n agtype)"
            )
            if cur.fetchone():
                row = (None, target, label, scope)
        if row:
            entity_data = {"id": row[0], "name": row[1], "type": "graph",
                           "label": row[2], "project": row[3],
                           "connections": []}
            # also query AGE for graph neighbors (same scope label)
            label = _scope_label(project)
            cur.execute("SET search_path TO ag_catalog, public")
            name_json = json.dumps(target)
            cur.execute(
                f"SELECT * FROM cypher('memory_graph', $$ "
                f"MATCH (a:{label} {{name: {name_json}}})-[r]-(b) "
                f"RETURN b.name, type(r), coalesce(r.weight, 1.0), startNode(r).name $$) AS "
                f"(neighbor agtype, rel agtype, w agtype, src agtype)"
            )
            entity_data["connections"] = [
                {"entity": _ag_str(r[0]), "relation": _ag_str(r[1]),
                 "weight": float(str(r[2])),
                 "direction": "outgoing" if _ag_str(r[3]) == target else "incoming"}
                for r in cur.fetchall()
            ]
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.exception("get_entity failed")
        return fail(STATUS_ERROR, str(e), entity=target)
    finally:
        conn.close()
    if entity_data is None and memories:
        entity_data = {
            "id": None,
            "name": target,
            "type": "memory-only",
            "project": _scope_key(project),
            "connections": [],
        }
    if entity_data is None and not memories:
        return fail(STATUS_NOT_FOUND, f"entity not found: {target}", entity=target)
    # The rows come from several tables: newest update first makes the
    # detail view deterministic page-over-page.
    memories.sort(key=lambda m: m.get("updated_at") or "", reverse=True)
    has_more = total > offset + limit
    window = memories[offset:offset + limit]
    if offset > 0 and total > 0 and not window:
        warnings.append(
            f"no results on this page: offset={offset} is beyond the "
            f"{total} memories of {target!r}; lower the offset")
    if has_more:
        warnings.append(
            f"showing {len(window)} of {total} memories of {target!r}; use "
            f"offset={offset + limit} to page through the rest"
        )
    projected: list[dict] = []
    expired_count = 0
    for m in window:
        item = project_row(m, fields)
        if _is_expired(m.get("expires_at")):
            expired_count += 1
            if fields is None or "expired" in fields:
                item["expired"] = True
        projected.append(item)
    if expired_count:
        warnings.append(
            f"{expired_count} memory/memories of {target!r} have an expired "
            f"TTL: they are not active anymore")
    return ok(STATUS_OK, entity=target, details=entity_data,
              memories=projected,
              total=total, relation_context_count=relation_context_count,
              count=len(window), offset=offset,
              has_more=has_more,
              next_offset=(offset + limit) if has_more else None,
              warnings=warnings)



def list_entities(
    entity_type: Annotated[
        str | None,
        Field(description="Only list entities of this kind. 'graph' selects "
                          "every graph node, 'memory-only' only the entities "
                          "without a graph node; a raw AGE label (e.g. "
                          "'Project_loci_test_c602412f') selects exactly that "
                          "scope's nodes. Without it both kinds are listed.")] = None,
    limit: Annotated[
        int,
        Field(description="Maximum number of results (default 50).")] = 50,
    project: Annotated[
        str | None,
        Field(description="Project scope. Without it (or with '*') every "
                          "scope is listed; 'global' lists only the shared "
                          "scope.")] = None,
    include_global: Annotated[
        bool,
        Field(description="True (default) lists global entities as well; False "
                          "restricts a project browse to project entities.")] = True,
    offset: Annotated[
        int,
        Field(description="Skip this many matches (default 0). Use it together "
                          "with `limit` to page through a long entity list; the "
                          "response reports `has_more`.")] = 0,
) -> dict[str, Any]:
    """Browse entities.

    Returns graph entities (created via `relate`) plus entity names that only
    exist on memories. With `project`, that project and global scope are
    listed by default; set `include_global=False` for strict project
    isolation. Without `project` (or with `project="*"`) every scope is
    listed; `project="global"` lists only the shared scope. Each row carries
    its `project`; `graph_node`
    is false for memory-only entities. The `type` column says which kind an
    entity is: 'graph' (with the raw AGE label of its scope in `label`) or
    'memory-only'. Synthetic relation-context entities
    (`<from>_<relation>_<to>`) are not listed. The list is
    sorted by name; `offset` pages through it.
    `count` is the page size and `total` the full match count: an entity that
    has a graph node is still reported as such even when `entity_type` hides
    that node.
    `entity_type` selects one kind of entity: 'graph', 'memory-only', or a
    raw AGE label for one scope's nodes. Without it both kinds are listed.
    A project that does not exist is reported in `warnings`: the browse then
    only covered the global scope.
    """
    logger.info("list_entities entity_type=%s limit=%s offset=%s project=%s",
                entity_type, limit, offset, project)
    raw_project = project
    all_projects = project == "*"
    try:
        project = None if all_projects else _norm_project(project)
    except ValueError as e:
        return fail(STATUS_ERROR, str(e))
    global_only = not all_projects and _is_global_scope(raw_project)
    if limit < 1:
        return fail(STATUS_ERROR, "limit must be >= 1")
    if offset < 0:
        return fail(STATUS_ERROR, "offset must be >= 0")
    conn = get_conn()
    warnings: list[str] = []
    try:
        cur = conn.cursor()
        unknown = _unknown_project_warning(cur, project)
        if unknown:
            warnings.append(unknown)
        if project:
            tables = _retrieval_tables(cur, project)
        elif global_only:
            tables = ["memories"]
        else:
            tables = _all_mem_tables(cur)
        if (project or all_projects) and not include_global:
            tables = [table for table in tables if table != "memories"]
        names = _project_names(cur)
        scope_of = {t: (names.get(t[len("mem_"):], t[len("mem_"):])
                        if t != "memories" else "global") for t in tables}

        if project and include_global:
            scope_conditions = ["(project = %s OR project = 'global')"]
            scope_params: list[Any] = [project]
        elif project:
            scope_conditions = ["project = %s"]
            scope_params = [project]
        elif global_only:
            scope_conditions = ["project = 'global'"]
            scope_params = []
        elif all_projects and not include_global:
            scope_conditions = ["project <> 'global'"]
            scope_params = []
        else:
            scope_conditions = ["TRUE"]
            scope_params = []


        # Every graph node of the selected scopes, independent of the type
        # filter: a node that the filter hides must not come back as a
        # "memory-only" entity, and it must not be counted twice.
        cur.execute(
            f"SELECT name, project, type FROM entities "
            f"WHERE {' AND '.join(scope_conditions)}",
            scope_params,
        )
        node_rows = cur.fetchall()
        graph_nodes = {(r[0], r[1]) for r in node_rows}
        graph_labels = {r[2] for r in node_rows if r[2]}

        # A raw AGE label is accepted for backwards compatibility, but a typo
        # used to answer an empty page silently. Name the valid labels instead.
        if (entity_type and entity_type not in ("graph", "memory-only")
                and entity_type not in graph_labels):
            conn.rollback()
            if graph_labels:
                valid = ", ".join(sorted(graph_labels))
                return fail(
                    STATUS_ERROR,
                    f"unknown entity_type {entity_type!r}: use 'graph', "
                    f"'memory-only' or one of the graph labels in this scope: "
                    f"{valid}")
            return fail(
                STATUS_ERROR,
                f"unknown entity_type {entity_type!r}: the selected scope has "
                f"no graph labels; use 'graph' for every graph node or "
                f"'memory-only' for entities without one")

        conditions = list(scope_conditions)
        params = list(scope_params)
        # The type filter selects one kind of entity: "graph" selects every
        # graph node, "memory-only" only the entities without one, and a raw
        # AGE label (e.g. "Project_loci_test_x1") selects exactly that scope's
        # nodes - the old, label-based filter stays accepted.
        if entity_type and entity_type not in ("graph", "memory-only"):
            conditions.append("type = %s")
            params.append(entity_type)
        where = " AND ".join(conditions)
        # No LIMIT here: `total` must count every match, and the memory-only
        # entities below are loaded in full anyway. The window is cut after
        # the sort.
        if entity_type == "memory-only":
            # The graph rows above are cut, so `total` must not count them:
            # the memory-only counts below are the whole total then.
            total = 0
        else:
            cur.execute(f"SELECT COUNT(*) FROM entities WHERE {where}", params)
            total = cur.fetchone()[0]
        cur.execute(
            f"SELECT id, name, type, project FROM entities "
            f"WHERE {where} ORDER BY name",
            params,
        )
        rows = [{"id": r[0], "name": r[1], "type": "graph", "label": r[2],
                 "project": r[3], "graph_node": True} for r in cur.fetchall()]
        if entity_type == "memory-only":
            # A raw label filter restricted the graph rows above; the
            # "memory-only" kind has no SQL label, so the graph rows are cut
            # here and only the memory-only entities below stay listed.
            rows = []

        seen = {(r["name"], r["project"]) for r in rows}
        # The type filter selects one kind of entity: a graph node carries
        # the label of its scope, a memory-only entity carries the type
        # "memory-only". Asking for one kind must not return the other.
        want_memory_only = entity_type in (None, "memory-only")
        for table in tables if want_memory_only else []:
            scope = scope_of[table]
            # Memory-only entities: a name that occurs in the memories of this
            # scope but has no graph node there. Synthetic relation-context
            # entities (e.g. "A_KNOWS_B") are excluded: they exist only to make
            # the edge context searchable, not as browsable knowledge. An
            # entity that also has a real memory stays visible.
            context_flag = "COALESCE(metadata->>'relation_context', 'false') <> 'true'"
            cur.execute(
                f"SELECT COUNT(*) FROM (SELECT entity FROM {table} "
                f"WHERE entity IS NOT NULL AND " + ACTIVE_ROW_SQL + " "
                f"GROUP BY entity HAVING bool_or({context_flag})) e "
                f"WHERE NOT EXISTS (SELECT 1 FROM entities en "
                f"WHERE en.name = e.entity AND en.project = %s)",
                (scope,),
            )
            total += cur.fetchone()[0]
            cur.execute(
                f"SELECT entity FROM {table} "
                f"WHERE entity IS NOT NULL AND " + ACTIVE_ROW_SQL + " "
                f"GROUP BY entity HAVING bool_or({context_flag})"
            )
            for (ent,) in cur.fetchall():
                if (ent, scope) in seen:
                    continue
                # A name that has a graph node in this scope is not a
                # memory-only entity - even when the type filter hid that
                # node from the listing above.
                if (ent, scope) in graph_nodes:
                    continue
                seen.add((ent, scope))
                # Memory-only entity: same row shape as a graph node, so an
                # agent never has to guess which keys exist.
                rows.append({"id": None, "name": ent, "type": "memory-only",
                             "label": None, "project": scope,
                             "graph_node": False})
        conn.commit()
    except Exception as e:
        conn.rollback()
        return fail(STATUS_ERROR, str(e))
    finally:
        conn.close()
    # A stable order is what makes `offset` usable.
    rows.sort(key=lambda r: (str(r.get("name") or ""), str(r.get("project") or "")))
    window = rows[offset:offset + limit]
    has_more = total > offset + limit
    if offset > 0 and total > 0 and not window:
        warnings.append(
            f"no results on this page: offset={offset} is beyond the "
            f"{total} matching entities; lower the offset")
    return ok(STATUS_OK, count=len(window), total=total, offset=offset,
              has_more=has_more,
              next_offset=(offset + limit) if has_more else None,
              warnings=warnings, results=window)


def init_context(project: Annotated[
        str | None,
        Field(description="Project to bootstrap for. '*' bootstraps all "
                          "projects; without a project only global memories "
                          "are summarized.")] = None,
    detail: Annotated[
        Literal["slim", "full"],
        Field(description="'slim' (default) returns only what a session needs "
                          "and removes near-duplicates; 'full' returns every "
                          "column (never internal ones, never the search "
                          "score annotations).")] = "slim",
    include_global: Annotated[
        bool,
        Field(description="True (default) includes global memories; False "
                          "excludes them.")] = True,
    limit: Annotated[
        int,
        Field(description="How many memories to return (default 10). The list "
                          "always contains the most recently changed "
                          "memories of the scope.")] = 10,
    include_empty_projects: Annotated[
        bool,
        Field(description="True (default) lists every known scope in "
                          "`project_stats`, including scopes without active "
                          "memories; False only scopes with active "
                          "memories.")] = True,
) -> dict[str, Any]:
    """Call at session start. Returns recent + important bootstrap info + health.

    Pass `project` to bootstrap for one project; global memories are included
    by default. Set `include_global=False` to exclude them. Without a project,
    only global memories are summarized; pass `project="*"` for all projects.
    `detail="slim"` (default) returns only the fields a session needs and
    removes near-duplicates; `detail="full"` returns every column except the
    internal ones and the search score annotations. Every memory carries the
    `project` it belongs to, and the list is sorted by the last update.

    The memories are the most recently changed ones of the scope - a freshly
    stored fact always shows up, even if it was never read before.
    `projects` lists every scope the answer covers, including the shared
    one ("global"). `active_memories` counts the active rows across the
    covered tables (a cross-scope copy can be counted in both).
    `superseded_memories` and `forgotten_memories` are independent counts:
    a row can be BOTH superseded and forgotten, so the numbers can overlap
    and do not have to add up to `total_memories - active_memories`.
    `project_stats` lists every covered scope with its own sizes (memories,
    total, superseded, forgotten, entities, relations) and replaces the
    separate project listing; `include_empty_projects=False` drops scopes
    without active memories.
    `graph_entities_count` counts the graph nodes of the covered scopes.
    """
    raw_project = project
    all_projects = project == "*"
    explicit_scope = bool(raw_project and str(raw_project).strip())
    try:
        project = None if all_projects else _norm_project(project)
    except ValueError as e:
        return fail(STATUS_ERROR, str(e))
    if detail not in ("slim", "full"):
        return fail(STATUS_ERROR, "detail must be 'slim' or 'full'")
    if limit < 1:
        return fail(STATUS_ERROR, "limit must be >= 1")
    conn = get_conn()
    memories = []
    total = 0
    active = 0
    superseded = 0
    forgotten = 0
    projects: list[str] = []
    warnings: list[str] = []
    try:
        cur = conn.cursor()
        unknown = _unknown_project_warning(cur, project)
        if unknown:
            warnings.append(unknown)
        tables = (_all_mem_tables(cur) if all_projects else
                  _retrieval_tables(cur, project) if project else ["memories"])
        if (project or all_projects) and not include_global:
            tables = [table for table in tables if table != "memories"]
        # recent_first: the bootstrap shows what changed last, not what was
        # read most often.
        memories = _search_memories(cur, None, None, None, None, limit, project,
                                    include_decay=True, tables=tables,
                                    recent_first=True)
        scope_names = _scopes_of_tables(cur, tables)
        # Every scope the bootstrap covers is listed - including the shared
        # one. Without it the caller cannot see that global memories were
        # part of the answer.
        projects = list(scope_names)
        if not all_projects and not explicit_scope:
            # The default bootstrap only shows the shared scope. A session
            # start that reports "0 memories" while a dozen project facts
            # exist must not look like an empty database.
            other_scopes = [s for s in _scopes_of_tables(cur, _all_mem_tables(cur))
                            if s not in scope_names]
            if other_scopes:
                warnings.append(
                    f"only the global scope was summarized; other scopes "
                    f"exist ({', '.join(other_scopes)}) - pass project='*' to "
                    f"bootstrap all of them")
        project_stats: list[dict[str, Any]] = []
        for table, scope in zip(tables, scope_names):
            cur.execute(
                f"SELECT COUNT(*), "
                f"COUNT(*) FILTER (WHERE superseded_by IS NULL AND deleted_at IS NULL), "
                # Superseded is a history marker orthogonal to the soft
                # delete: a row can be BOTH superseded and forgotten. Counting
                # `deleted_at IS NULL` here made a forgotten-but-once-
                # superseded fact vanish from the superseded counter, and
                # total - active - forgotten <= superseded broke its promise.
                f"COUNT(*) FILTER (WHERE superseded_by IS NOT NULL), "
                f"COUNT(*) FILTER (WHERE deleted_at IS NOT NULL) "
                f"FROM {table}"
            )
            table_total, table_active, table_superseded, table_forgotten = \
                cur.fetchone()
            total += table_total
            active += table_active
            superseded += table_superseded
            forgotten += table_forgotten
            cur.execute(
                "SELECT COUNT(*) FROM entities WHERE project = %s", (scope,))
            entities = cur.fetchone()[0]
            cur.execute(
                "SELECT COUNT(*) FROM relations WHERE source_id IN "
                "(SELECT id FROM entities WHERE project = %s) OR target_id IN "
                "(SELECT id FROM entities WHERE project = %s)",
                (scope, scope),
            )
            relations = cur.fetchone()[0]
            if include_empty_projects or table_active:
                project_stats.append({
                    "project": scope,
                    "scope": "global" if scope == "global" else "project",
                    "memories": table_active,
                    "total_memories": table_total,
                    "superseded_memories": table_superseded,
                    "forgotten_memories": table_forgotten,
                    "entities": entities,
                    "relations": relations,
                })
        project_stats.sort(
            key=lambda r: (r["scope"] != "global", str(r["project"])))
        conn.commit()
    except Exception as e:
        conn.rollback()
        return fail(STATUS_ERROR, str(e))
    finally:
        conn.close()
    if detail == "slim":
        memories = dedupe_near_duplicates(memories)
    # The key promises recency: sort the selected memories by their last
    # update instead of by score.
    memories.sort(key=lambda m: m.get("updated_at") or _EPOCH, reverse=True)
    memories = memories[:limit]
    if detail == "slim":
        memories = [project_bootstrap(m) for m in memories]
    else:
        memories = [project_full(m) for m in memories]
    return ok(
        STATUS_OK,
        total_memories=total,
        active_memories=active,
        superseded_memories=superseded,
        forgotten_memories=forgotten,
        projects=sorted(projects),
        project_stats=project_stats,
        count=len(memories),
        recent_memories=memories,
        graph_entities_count=_count_entities(scope_names),
        inference_healthy=_health_check(),
        warnings=warnings,
    )
