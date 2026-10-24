import json
import logging
from datetime import datetime, timezone
from typing import Annotated, Any, Literal

from pydantic import Field

from mcp_server.browse import (
    BrowseRequest,
    browse,
    dedupe_rows,
    merge_unique_rows,
    paginate,
    project_rows,
    tables_and_scopes,
)
from mcp_server.browse import (
    order_column as browse_order_column,
)
from mcp_server.config import (
    CONFIDENCE_HIGH,
    CONFIDENCE_MEDIUM,
    DEFAULT_MIN_SIMILARITY,
    MEMORY_TYPE_HELP,
    MEMORY_TYPES,
    SENTINEL_FORGOTTEN,
)
from mcp_server.duplicates import (
    dedupe_similar_rows as dedupe_near_duplicates,
)
from mcp_server.envelope import STATUS_ERROR, STATUS_NOT_FOUND, STATUS_OK, fail, ok
from mcp_server.graph_store import GraphStore
from mcp_server.inference_gateway import embed
from mcp_server.lifecycle import (
    counts as lifecycle_counts,
)
from mcp_server.lifecycle import (
    is_expired as _is_expired,
)
from mcp_server.lifecycle import (
    visibility_sql as _visibility_condition,
)
from mcp_server.projection import (
    PUBLIC_FIELDS,
    project_bootstrap,
    project_full,
    validate_fields,
)
from mcp_server.projection import project as project_row
from mcp_server.scope import (
    MODE_BOOTSTRAP,
    MODE_BROWSE,
    MODE_LOOKUP,
    MODE_PROJECT,
    resolve_scope,
)
from mcp_server.scope_catalog import (
    _all_mem_tables,
    _count_entities,
    _norm_entity,
    _project_names,
    _scope_key,
)
from mcp_server.scoring import _decay_norm
from mcp_server.search import (
    SearchRequest,
    _extra_conditions,
    _search_memories,
    _search_memories_graph_boost,
)
from mcp_server.storage import (
    _log_access,
)
from mcp_server.store import MemoryStore
from mcp_server.store_maintenance import entity_count, relation_count
from mcp_server.store_rows import (
    count_graph_where,
    entity_present,
    entity_row_count,
    graph_entities,
    graph_nodes as graph_node_rows,
    graph_projects,
    memory_only_count,
    memory_only_names,
    relation_context_rows,
)
from mcp_server.tools.maintenance import _health_check

logger = logging.getLogger(__name__)

_EPOCH = datetime.min.replace(tzinfo=timezone.utc)

# The token-cheap recall view: enough to judge a hit and call the next tool
# (id/content/scope/entity/type/rank), without the metadata blob or history
# bookkeeping. An explicit `fields` list always wins over `detail`.
SLIM_RECALL_FIELDS = (
    "id", "content", "mem_type", "entity", "project", "importance",
    "updated_at", "superseded_by", "forgotten", "expired",
    "_similarity", "_relevance_score", "_rerank_score", "_final_score",
    "_graph_boost", "_graph_source", "_graph_relation_context",
)


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
                          "least this value (0.0-1.0). A fresh memory with "
                          "importance 1.0 scores 0.9, importance 0.5 scores "
                          "0.45; a 90-day-old one ~0.5 of that.")] = None,
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
                          "(0.0-1.0, default 0.30, just above the embedding "
                          "model's 0.288 noise floor). Full-text hits are never "
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
                          "_rrf_norm, _relevance_score, _rerank_score, "
                          "_scope_boost, _decay_score, _decay_norm, "
                          "_final_score, _graph_boost, _graph_source, "
                          "_graph_weight, _graph_relation_context, "
                          "_exact_match.")] = None,
    detail: Annotated[
        Literal["full", "slim"],
        Field(description="'full' (default) returns every public column and "
                          "score annotation. 'slim' returns only the fields a "
                          "caller needs to judge and reuse a hit (id, content, "
                          "entity, mem_type, project, importance, timestamps "
                          "and the ranking scores), which saves tokens for "
                          "large result sets. An explicit `fields` list wins.")] = "full",
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
    Graph-only hits fill the slots left by query hits; they never push a
    query match out of the page.
    `min_similarity` defaults to 0.30, just above the embedding model's
    measured 0.288 noise floor, and only applies to the vector retriever;
    a full-text hit is kept even when its embedding is a poor match, so an
    exact identifier ("ZEBRAFISH_TOKEN") is found even if the embedding model
    handles that word badly. Pass `None` to disable the filter. The response
    reports `candidates` and `dropped_by_similarity`, and warns when the
    filter removed everything. When hits were dropped, it also names
    `best_dropped_by_similarity` (id, similarity, scope, content preview):
    the strongest rejected vector hit, so a natural-language question that
    missed the threshold by a hair is visible and one retry with
    `min_similarity=null` can recover it.
    When the first stage is ambiguous (best relevance below 0.85 and margin
    to the runner-up below 0.10), the top candidates are reranked by a local
    cross-encoder; `reranked` reports how many were rescored. Reranked rows
    carry `_rerank_score` (a softmax over that candidate block) and an
    updated `_relevance_score`; `_similarity` still reports the vector score.
    A reranker failure is reported in `rerank_error` and falls back to the
    first-stage ranking.
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
    min_decay_score (0..1) filters on the normalized decay score; a fresh
    memory with importance 1.0 scores 0.9 (importance 0.5 scores 0.45), and
    a 90-day-old one is roughly half of that.
    Set only_superseded=True to return only invalidated facts.
    Set fields=["id", "content"] to receive only the named columns.
    Set include_global=False to restrict a project search to project memories.
    `updated_after` and `metadata_filter` narrow the result set.
    include_relation_context=False hides the relation-context sentences of
    the graph edges from the ranking (they repeat what the edge already says).
    `offset` pages through the ranked results (with `limit`); the response
    reports `has_more` and `next_offset`.
    The answer carries `confidence` (`high` >= 0.60, `medium` >= 0.45,
    `low` below that, `none` for no result, `graph` for graph-only hits) and
    `best_relevance`. A low-confidence answer warns instead of looking like a
    confident miss; the warning names `min_similarity=null` when the floor
    may have removed the intended fact.
    `project="*"` searches every scope at once. A project that does not
    exist is reported in `warnings`: the search then only covered the global
    scope.
    """
    logger.info("recall query='%s' entity=%s hybrid=%s include_decay=%s graph_boost=%s min_similarity=%s",
                query[:80], entity, hybrid, include_decay, graph_boost, min_similarity)
    # "*" means every scope. The raw value tells explicit scope intent apart
    # from the default: an explicitly requested single scope must not produce
    # a warning that suggests searching the other scopes.
    try:
        scope = resolve_scope(project, include_global)
        project = scope.project
        entity = _norm_entity(entity)
        graph_entity = _norm_entity(graph_entity)
    except ValueError as e:
        return fail(STATUS_ERROR, str(e))
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
    db = MemoryStore()
    try:
        cur = db.cur
        # A scope that does not exist must not look like an empty scope: the
        # search silently falls back to the global table.
        unknown = scope.unknown_warning(cur)
        if unknown:
            warnings.append(unknown)

        # Neue RRF-Funktion nutzen
        tables, scopes_searched = tables_and_scopes(cur, scope, MODE_PROJECT)
        # The scopes that were really searched. An agent that finds nothing
        # must be able to see whether other scopes were covered - otherwise
        # "no result" is indistinguishable from "wrong scope".
        other_scopes = scope.other_scopes(cur, tables)

        stats: dict[str, Any] = {}
        # The page needs `offset + limit` candidates, plus one to know whether
        # a further page exists. Slicing happens after the optional filters so
        # `offset` counts real returned rows.
        fetch_limit = limit + offset + 1
        results = _search_memories(cur, SearchRequest(
            query_embedding=query_emb,
            query_text=query if (hybrid or query_emb is None) else None,
            mem_type=mem_type,
            entity=entity,
            limit=fetch_limit,
            project=project,
            include_decay=include_decay,
            include_superseded=include_superseded,
            min_similarity=min_similarity,
            hybrid=hybrid,
            tables=tables,
            updated_after=updated_after,
            metadata_filter=metadata_filter,
            exclude_relation_context=not include_relation_context,
            stats=stats,
            include_expired=include_expired,
            include_deleted=include_deleted,
        ))

        # Graph-enhanced retrieval when enabled
        if graph_boost and boost_entity:
            results = _search_memories_graph_boost(
                results, boost_entity, fetch_limit, project, include_decay,
                include_global=include_global,
                updated_after=updated_after, metadata_filter=metadata_filter,
                exclude_relation_context=not include_relation_context,
                all_projects=scope.all_projects,
                include_expired=include_expired,
                include_deleted=include_deleted,
                warnings=warnings,
                cur=cur,
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

        paging = paginate(results, limit, offset)
        results = paging.window

        # Update access counters (decay frequency) only for the rows this
        # page actually returns: skipped matches were not read.
        hit_ids = [r.get("id") for r in results if isinstance(r, dict) and r.get("id")]
        _log_access(cur, hit_ids, project, tables=tables)
        for r in results:
            if isinstance(r, dict) and r.get("id") in hit_ids and "error" not in r:
                r["access_count"] = (r.get("access_count") or 0) + 1

        db.commit()
    except Exception as e:
        db.rollback()
        logger.exception("recall failed")
        return fail(STATUS_ERROR, str(e))
    finally:
        db.close()

    rows = []
    projection_fields = fields
    if projection_fields is None and detail == "slim":
        projection_fields = list(SLIM_RECALL_FIELDS)
    projectable = [r for r in results if isinstance(r, dict)]
    rows, expired_count = project_rows(
        projectable, fields, projection_fields=projection_fields,
        all_markers=fields is None)
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
        if other_scopes and not scope.explicit and include_global:
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
    # A hit is not automatically an answer. The best relevance of the page
    # tells the caller whether the query was actually answered or whether the
    # rows are merely the closest noise above the similarity floor.
    relevance_scores = [
        r["_relevance_score"] for r in results
        if isinstance(r, dict) and r.get("_relevance_score") is not None
    ]
    best_relevance = (round(max(relevance_scores), 4)
                      if relevance_scores else None)
    if not rows:
        confidence = "none"
    elif best_relevance is None:
        confidence = "graph"
    elif best_relevance >= CONFIDENCE_HIGH:
        confidence = "high"
    elif best_relevance >= CONFIDENCE_MEDIUM:
        confidence = "medium"
    else:
        confidence = "low"
    if confidence == "low" and not (include_superseded or include_deleted
                                    or include_expired or only_superseded):
        warnings.append(
            f"low confidence: the best match has _relevance_score="
            f"{best_relevance} (below {CONFIDENCE_MEDIUM}); the results may "
            f"not answer the query. Rephrase the query or pass "
            f"min_similarity=null to inspect weaker hits")
    payload: dict[str, Any] = {
        # `candidates` counts the hits that made it into the ranking.
        "candidates": candidates,
        "confidence": confidence,
        "best_relevance": best_relevance,
        "dropped_by_similarity": filtered_out,
        "offset": offset,
        "has_more": paging.has_more,
        "next_offset": paging.next_offset,
        # The scopes that were actually searched: an agent that finds
        # nothing has to know whether other scopes were covered.
        "scopes_searched": scopes_searched,
        "detail": detail,
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
        fulltext_only = stats["fulltext_only"]
        payload["fulltext_only"] = fulltext_only
        if min_similarity is not None:
            # `_similarity` is null for these rows, so the documented filter
            # did not judge them. Saying so prevents the wrong conclusion
            # ("the hit has similarity above min_similarity").
            warnings.append(
                f"{fulltext_only} result(s) matched only the full-text index: "
                f"min_similarity={min_similarity} does not apply to them "
                f"(_similarity is null)")
    hidden_exact = stats.get("hidden_exact_match")
    if hidden_exact:
        payload["hidden_exact_match"] = dict(hidden_exact)
        flags = hidden_exact.get("flags") or []
        flag_hint = ("; pass " + ", ".join(f"{flag}=true" for flag in flags)
                     if flags else "")
        warnings.append(
            f"{hidden_exact['count']} hidden memory/memories contain this "
            f"exact text but are not active (expired, superseded or "
            f"soft-forgotten){flag_hint} to see them")
    if stats.get("reranked"):
        payload["reranked"] = stats["reranked"]
    if stats.get("rerank_error"):
        payload["rerank_error"] = stats["rerank_error"]
        warnings.append(
            f"reranking skipped: {stats['rerank_error']}; the first-stage "
            f"ranking is unchanged")
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


def recall_batch(
    queries: Annotated[
        list[str],
        Field(description="Search texts, 1-25 non-empty strings. Each query is "
                          "ranked independently (its own limit/offset), so one "
                          "call can cover several intents.")],
    project: Annotated[
        str | None,
        Field(description="Project scope shared by every query. Without it "
                          "only global memories are searched; '*' searches "
                          "every scope.")] = None,
    include_global: Annotated[
        bool,
        Field(description="True (default) lets a project search also see "
                          "global memories.")] = True,
    limit: Annotated[
        int,
        Field(description="Maximum results per query (default 10).")] = 10,
    offset: Annotated[
        int,
        Field(description="Skip this many ranked matches per query.")] = 0,
    hybrid: Annotated[
        bool,
        Field(description="True (default) fuses vector and full-text results.")] = True,
    min_similarity: Annotated[
        float | None,
        Field(description="Vector similarity floor per query (default 0.30); "
                          "None disables it.")] = DEFAULT_MIN_SIMILARITY,
    include_decay: Annotated[
        bool,
        Field(description="Blend recency/access into the score.")] = False,
    include_superseded: Annotated[
        bool,
        Field(description="Also return superseded memories.")] = False,
    include_deleted: Annotated[
        bool,
        Field(description="Also return soft-forgotten memories.")] = False,
    include_expired: Annotated[
        bool,
        Field(description="Also return memories whose TTL has passed.")] = False,
    include_relation_context: Annotated[
        bool,
        Field(description="True (default) includes edge sentences; False hides "
                          "them.")] = True,
    graph_boost: Annotated[
        bool,
        Field(description="Add graph-neighbour memories. Without an entity "
                          "the boost is skipped with a warning.")] = False,
    entity: Annotated[
        str | None,
        Field(description="Only return memories of this entity (also the "
                          "graph_boost expansion point).")] = None,
    mem_type: Annotated[
        str | None,
        Field(description="Only return memories of this type. " + MEMORY_TYPE_HELP)] = None,
    detail: Annotated[
        Literal["full", "slim"],
        Field(description="'full' (default) returns every public column and "
                          "score annotation; 'slim' saves tokens.")] = "full",
    fields: Annotated[
        list[str] | None,
        Field(description="Only return these columns per hit, e.g. "
                          "['id', 'content'].")] = None,
) -> dict[str, Any]:
    """Run several independent `recall` queries in one call.

    A multi-intent turn (\"how do we deploy and where are the logs?\") often
    needs two searches; this tool returns one ranked list per query instead of
    forcing the agent to concatenate them (which dilutes the vector score).
    The shared options apply to every query; each entry reports its own
    `count`, `warnings`, filter counters and `scopes_searched`.
    An empty query is reported in its own entry (`ok: false`, `status:
    error`) and in the top-level `warnings`; the other queries still run, so
    one bad intent does not discard the whole batch.
    """
    logger.info("recall_batch queries=%s", len(queries or []))
    if not isinstance(queries, list) or not queries:
        return fail(STATUS_ERROR, "queries must be a non-empty list of strings")
    if len(queries) > 25:
        return fail(STATUS_ERROR,
                    "queries accepts at most 25 entries per call")
    cleaned = [str(q).strip() if isinstance(q, str) else "" for q in queries]
    if mem_type is not None and mem_type not in MEMORY_TYPES:
        return fail(STATUS_ERROR,
                    f"Invalid mem_type. Choose from: "
                    f"{', '.join(sorted(MEMORY_TYPES))}")
    results: list[dict[str, Any]] = []
    warnings: list[str] = []
    for index, query in enumerate(cleaned):
        # One bad query must not cost the other intents of the batch: it is
        # reported in its own entry (and named in the top-level warnings), the
        # remaining queries still run.
        if not query:
            entry = {
                "index": index, "query": query, "ok": False,
                "status": STATUS_ERROR,
                "error": "query must not be empty",
                "warnings": ["empty query skipped; pass non-empty text"],
                "count": 0, "results": [],
            }
            warnings.append(
                f"query {index} is empty and was skipped; every query needs "
                f"text")
            results.append(entry)
            continue
        call = recall(
            query, mem_type=mem_type, entity=entity, limit=limit,
            offset=offset, project=project, hybrid=hybrid,
            include_decay=include_decay, include_superseded=include_superseded,
            include_deleted=include_deleted, include_expired=include_expired,
            graph_boost=graph_boost, include_global=include_global,
            min_similarity=min_similarity, fields=fields, detail=detail,
            include_relation_context=include_relation_context)
        entry = {"index": index, "query": query, **call}
        if not call.get("ok"):
            warnings.append(
                f"query {index} ({query!r}) failed: {call.get('error')}")
        results.append(entry)
    return ok(STATUS_OK, count=len(results), results=results,
              warnings=warnings)


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
    memory_ids: Annotated[
        list[str] | None,
        Field(description="Bulk id lookup: returns every requested memory in "
                          "`results`, names the ids that are missing or "
                          "ambiguous, and lists rows that exist but are "
                          "hidden (superseded/forgotten/expired) in `hidden` "
                          "with the flag that reveals them. Mutually "
                          "exclusive with `memory_id`; browse filters are "
                          "ignored.")] = None,
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
    try:
        scope = resolve_scope(project, include_global)
        project = scope.project
        entity = _norm_entity(entity)
    except ValueError as e:
        return fail(STATUS_ERROR, str(e))
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
        _extra_conditions(updated_after, metadata_filter)
    except ValueError as e:
        return fail(STATUS_ERROR, str(e))
    order_value = browse_order_column(order_by)
    if not order_value:
        return fail(STATUS_ERROR,
                    "order_by must be one of updated_at, created_at, "
                    "access_count, importance")
    if memory_ids is not None:
        if memory_id:
            return fail(STATUS_ERROR,
                        "pass either memory_id or memory_ids, not both")
        if not isinstance(memory_ids, list) or not memory_ids or any(
                not isinstance(mid, str) or not mid.strip()
                for mid in memory_ids):
            return fail(STATUS_ERROR,
                        "memory_ids must be a non-empty list of memory ids")
        wanted = list(dict.fromkeys(mid.strip() for mid in memory_ids))
        if len(wanted) > 200:
            return fail(STATUS_ERROR,
                        "memory_ids accepts at most 200 ids per call")
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
        results: list[dict] = []
        missing: list[str] = []
        hidden: list[dict] = []
        errors: list[dict] = []
        for mid in wanted:
            found = _get_memory(
                memory_id=mid, project=raw_project,
                include_superseded=include_superseded,
                include_deleted=include_deleted,
                include_expired=include_expired, fields=fields,
                include_global=include_global)
            if found.get("ok"):
                results.append(found["memory"])
            elif found.get("scopes"):
                errors.append({"id": mid, "status": found.get("status"),
                               "error": found.get("error"),
                               "scopes": found["scopes"]})
            elif found.get("hidden"):
                # A row that exists but is superseded/forgotten/expired is not
                # "missing": merging the two sent the caller looking for a
                # typo instead of a visibility flag.
                hidden.append({
                    "id": mid,
                    "scope": found.get("scope"),
                    "reasons": found.get("hidden_reasons") or [],
                    "reveal_flags": found.get("reveal_flags") or [],
                })
            else:
                missing.append(mid)
        warnings = []
        if ignored:
            warnings.append("memory_ids performs a bulk id lookup: "
                            f"{', '.join(ignored)} ignored")
        if missing:
            warnings.append(
                f"{len(missing)} id(s) not found: "
                + ", ".join(missing[:10])
                + (" ..." if len(missing) > 10 else ""))
        if hidden:
            warnings.append(
                f"{len(hidden)} id(s) exist but are hidden (superseded, "
                f"soft-forgotten or expired): "
                + ", ".join(
                    f"{item['id']} ({', '.join(item['reasons'])}; pass "
                    f"{' and '.join(item['reveal_flags'])}=true)"
                    for item in hidden[:10])
                + (" ..." if len(hidden) > 10 else ""))
        if errors:
            warnings.append(
                f"{len(errors)} id(s) are ambiguous across scopes; pass "
                f"project=<name> to resolve them")
        return ok(STATUS_OK, count=len(results), results=results,
                  missing=missing, hidden=hidden, errors=errors,
                  warnings=warnings)

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
    # Visibility, cross-scope dedupe, ordering and the page counters are the
    # same pipeline as every other browse-shaped read: browse owns them once.
    db = MemoryStore()
    try:
        page = browse(db, BrowseRequest(
            project=raw_project,
            include_global=include_global,
            include_deleted=include_deleted,
            include_expired=include_expired,
            include_superseded=include_superseded,
            include_duplicates=include_duplicates,
            limit=limit,
            offset=offset,
            order_by=order_by,
            entity=entity,
            mem_type=mem_type,
            contains=contains,
            updated_after=updated_after,
            metadata_filter=metadata_filter,
            include_relation_context=include_relation_context,
            fields=fields,
        ))
        db.commit()
    except Exception as e:
        db.rollback()
        return fail(STATUS_ERROR, str(e))
    finally:
        db.close()
    return ok(STATUS_OK, count=len(page.results), total=page.total,
              offset=page.offset,
              has_more=page.has_more,
              next_offset=page.next_offset,
              duplicates_hidden=page.duplicates_hidden,
              scopes_searched=page.scopes_searched,
              total_by_scope=page.total_by_scope,
              listed_by_scope=page.listed_by_scope,
              warnings=page.warnings, results=page.results)


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
    try:
        scope = resolve_scope(raw_project, include_global)
        project = scope.project
    except ValueError as e:
        return fail(STATUS_ERROR, str(e))
    fields_error = validate_fields(fields)
    if fields_error:
        return fail(STATUS_ERROR, fields_error)
    db = MemoryStore()
    try:
        cur = db.cur
        if scope.global_only:
            tables = ["memories"]
        elif project:
            tables = scope.tables(cur, MODE_LOOKUP)
        else:
            scopes = db.scopes_of(memory_id)
            if len(scopes) > 1:
                db.rollback()
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
        found, found_table = db.find(
            memory_id, tables=tables, where=where)
        if found is None:
            # Explain the miss: the row may exist but be hidden by one of the
            # visibility flags. A bare "not found" would make an agent believe
            # the id never existed.
            hidden, hidden_table = db.locate(memory_id, tables=tables)
            db.rollback()
            if hidden:
                superseded_by = hidden.get("superseded_by")
                deleted_at = hidden.get("deleted_at")
                expires_at = hidden.get("expires_at")
                reasons = []
                flags = []
                if deleted_at is not None:
                    reasons.append("soft-forgotten")
                    flags.append("include_deleted")
                if superseded_by is not None and superseded_by != SENTINEL_FORGOTTEN:
                    reasons.append("superseded")
                    flags.append("include_superseded")
                if _is_expired(expires_at):
                    reasons.append("expired")
                    flags.append("include_expired")
                return fail(
                    STATUS_NOT_FOUND,
                    f"memory {memory_id} exists but is not visible: "
                    + ("; ".join(
                        f"{reason} (pass {flag}=true)"
                        for reason, flag in zip(reasons, flags))
                       if reasons else "hidden by a filter"),
                    id=memory_id, scope=db.scope_of_table(hidden_table),
                    hidden=True, hidden_reasons=reasons or ["hidden"],
                    reveal_flags=flags)
            return fail(STATUS_NOT_FOUND, f"NOT FOUND: {memory_id}",
                        id=memory_id)
        scope = db.scope_of_table(found_table)
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
        db.commit()
    except Exception as e:
        db.rollback()
        return fail(STATUS_ERROR, str(e), id=memory_id)
    finally:
        db.close()
    return ok(STATUS_OK, id=memory_id, project=scope, memory=item)


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
    include_relation_context: Annotated[
        bool,
        Field(description="True (default) also lists the relation-context "
                          "sentences of the entity's graph edges; False "
                          "hides them (the memories of the entity itself are "
                          "unaffected).")] = True,
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
    `total` counts them too. Pass `include_relation_context=false` to hide
    them (the entity's own memories stay).
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
        scope = resolve_scope(project, include_global)
        project = scope.project
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
    db = MemoryStore()
    memories = []
    entity_data = None
    warnings: list[str] = []
    total = 0
    relation_context_count = 0
    scopes_searched: list[str] = []
    memories_by_scope: dict[str, int] = {}
    scope_presence: dict[str, str] = {}
    try:
        cur = db.cur
        tables, scopes_searched = (tables_and_scopes(cur, scope, MODE_PROJECT)
                                   if project else (None, ["global"]))
        used_tables = tables or ["memories"]
        # An entity name can exist in several scopes. The caller asked for one
        # lookup, so the answer has to say where else the name occurs instead
        # of silently merging a global memory-only entity with this project's
        # graph node.
        for table, scope_name in zip(used_tables, scopes_searched):
            if entity_present(db, table, target):
                scope_presence[scope_name] = "memory"
        for entity_project in graph_projects(db, target):
            scope_presence[entity_project or "global"] = "graph"
        total, hidden_duplicates = entity_row_count(
            db, target, used_tables, include_superseded, include_expired)
        # Fetch enough rows that the cross-scope copies the dedupe removes
        # cannot leave the page short. Adding just the hidden count keeps a
        # big entity from loading every row for a ten-item page.
        memories = _search_memories(cur, SearchRequest(
            query_embedding=None,
            query_text=None,
            entity=target,
            limit=offset + limit + 1 + hidden_duplicates,
            project=project,
            include_superseded=include_superseded,
            include_expired=include_expired,
            tables=tables,
        ))
        # An identical active fact in project and global is one row: the page
        # must apply the same rule as the count, otherwise `total`, `count`
        # and the last page disagree (a 12-row page could claim total=11).
        memories, _hidden_duplicates = dedupe_rows(memories)
        # The context sentence of an edge belongs to both of its endpoints.
        # Its content is deduplicated against the entity rows and against the
        # other scopes' context sentences for the same reason.
        if include_relation_context:
            context_rows = relation_context_rows(
                db, target, used_tables, include_superseded, include_expired)
        else:
            context_rows = []
        memories, added_context = merge_unique_rows(memories, context_rows)
        total += added_context
        relation_context_count += added_context

        details = GraphStore(cur=cur).entity_details(project, target)
        if details:
            row = details["row"]
            entity_data = {"id": row[0], "name": row[1], "type": "graph",
                           "label": row[2], "project": row[3],
                           "connections": details["connections"]}
        db.commit()
    except Exception as e:
        db.rollback()
        logger.exception("get_entity failed")
        return fail(STATUS_ERROR, str(e), entity=target)
    finally:
        db.close()
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
    if len(scope_presence) > 1:
        entity_data["scopes_with_entity"] = scope_presence
        detail_scope = entity_data.get("project") or "global"
        warnings.append(
            f"entity {target!r} exists in several scopes "
            f"({', '.join(sorted(scope_presence))}); `details` describes the "
            f"{detail_scope!r} occurrence and the memories come from "
            f"{', '.join(scopes_searched)}. Pass project=<name> for one scope")
    for memory in memories:
        scope_name = memory.get("_project") or "global"
        memories_by_scope[scope_name] = (
            memories_by_scope.get(scope_name, 0) + 1)
    # The rows come from several tables: newest update first makes the
    # detail view deterministic page-over-page.
    memories.sort(key=lambda m: m.get("updated_at") or "", reverse=True)
    paging = paginate(memories, limit, offset, total=total)
    window = paging.window
    if offset > 0 and paging.total > 0 and not window:
        warnings.append(
            f"no results on this page: offset={offset} is beyond the "
            f"{paging.total} memories of {target!r}; lower the offset")
    if paging.has_more:
        warnings.append(
            f"showing {len(window)} of {paging.total} memories of {target!r}; "
            f"use offset={offset + limit} to page through the rest"
        )
    projected, expired_count = project_rows(window, fields,
                                            mark_forgotten=False)
    if expired_count:
        warnings.append(
            f"{expired_count} memory/memories of {target!r} have an expired "
            f"TTL: they are not active anymore")
    return ok(STATUS_OK, entity=target, details=entity_data,
              memories=projected,
              total=paging.total, relation_context_count=relation_context_count,
              count=len(window), offset=offset,
              has_more=paging.has_more,
              next_offset=paging.next_offset,
              scopes_searched=scopes_searched,
              memories_by_scope=memories_by_scope,
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
    try:
        scope = resolve_scope(project, include_global)
        project = scope.project
    except ValueError as e:
        return fail(STATUS_ERROR, str(e))
    if limit < 1:
        return fail(STATUS_ERROR, "limit must be >= 1")
    if offset < 0:
        return fail(STATUS_ERROR, "offset must be >= 0")
    db = MemoryStore()
    warnings: list[str] = []
    try:
        cur = db.cur
        unknown = scope.unknown_warning(cur)
        if unknown:
            warnings.append(unknown)
        tables, _ = tables_and_scopes(cur, scope, MODE_BROWSE)
        names = _project_names(cur)
        scope_of = {t: (names.get(t[len("mem_"):], t[len("mem_"):])
                        if t != "memories" else "global") for t in tables}

        scope_condition, scope_params = scope.graph_filter()


        # Every graph node of the selected scopes, independent of the type
        # filter: a node that the filter hides must not come back as a
        # "memory-only" entity, and it must not be counted twice.
        node_rows = graph_node_rows(db, scope_condition, scope_params)
        graph_nodes = {(r[0], r[1]) for r in node_rows}
        graph_labels = {r[2] for r in node_rows if r[2]}

        # A raw AGE label is accepted for backwards compatibility, but a typo
        # used to answer an empty page silently. Name the valid labels instead.
        if (entity_type and entity_type not in ("graph", "memory-only")
                and entity_type not in graph_labels):
            db.rollback()
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

        conditions = [scope_condition]
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
            total = count_graph_where(db, where, params)
        rows = [{"id": r[0], "name": r[1], "type": "graph", "label": r[2],
                 "project": r[3], "graph_node": True}
                for r in graph_entities(db, where, params)]
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
            scope_name = scope_of[table]
            # Memory-only entities: a name that occurs in the memories of this
            # scope but has no graph node there. Synthetic relation-context
            # entities (e.g. "A_KNOWS_B") are excluded: they exist only to make
            # the edge context searchable, not as browsable knowledge. An
            # entity that also has a real memory stays visible.
            total += memory_only_count(db, table, scope_name)
            for ent in memory_only_names(db, table):
                if (ent, scope_name) in seen:
                    continue
                # A name that has a graph node in this scope is not a
                # memory-only entity - even when the type filter hid that
                # node from the listing above.
                if (ent, scope_name) in graph_nodes:
                    continue
                seen.add((ent, scope_name))
                # Memory-only entity: same row shape as a graph node, so an
                # agent never has to guess which keys exist.
                rows.append({"id": None, "name": ent, "type": "memory-only",
                             "label": None, "project": scope_name,
                             "graph_node": False})
        db.commit()
    except Exception as e:
        db.rollback()
        return fail(STATUS_ERROR, str(e))
    finally:
        db.close()
    # A stable order is what makes `offset` usable.
    rows.sort(key=lambda r: (str(r.get("name") or ""), str(r.get("project") or "")))
    paging = paginate(rows, limit, offset, total=total)
    window = paging.window
    if offset > 0 and paging.total > 0 and not window:
        warnings.append(
            f"no results on this page: offset={offset} is beyond the "
            f"{paging.total} matching entities; lower the offset")
    return ok(STATUS_OK, count=len(window), total=paging.total, offset=offset,
              has_more=paging.has_more,
              next_offset=paging.next_offset,
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
    try:
        scope = resolve_scope(project, include_global)
        project = scope.project
    except ValueError as e:
        return fail(STATUS_ERROR, str(e))
    if detail not in ("slim", "full"):
        return fail(STATUS_ERROR, "detail must be 'slim' or 'full'")
    if limit < 1:
        return fail(STATUS_ERROR, "limit must be >= 1")
    db = MemoryStore()
    memories = []
    total = 0
    active = 0
    superseded = 0
    forgotten = 0
    projects: list[str] = []
    warnings: list[str] = []
    try:
        cur = db.cur
        unknown = scope.unknown_warning(cur)
        if unknown:
            warnings.append(unknown)
        tables, scope_names = tables_and_scopes(
            cur, scope, MODE_BOOTSTRAP)
        # recent_first: the bootstrap shows what changed last, not what was
        # read most often.
        memories = _search_memories(cur, SearchRequest(
            query_embedding=None,
            query_text=None,
            limit=limit,
            project=project,
            include_decay=True,
            tables=tables,
            recent_first=True,
        ))
        # Every scope the bootstrap covers is listed - including the shared
        # one. Without it the caller cannot see that global memories were
        # part of the answer.
        projects = list(scope_names)
        if not scope.all_projects and not scope.explicit:
            # The default bootstrap only shows the shared scope. A session
            # start that reports "0 memories" while a dozen project facts
            # exist must not look like an empty database.
            other_scopes = scope.other_scopes(cur, tables)
            if other_scopes:
                warnings.append(
                    f"only the global scope was summarized; other scopes "
                    f"exist ({', '.join(other_scopes)}) - pass project='*' to "
                    f"bootstrap all of them")
        project_stats: list[dict[str, Any]] = []
        for table, scope_name in zip(tables, scope_names):
            table_counts = lifecycle_counts(cur, table)
            table_total = table_counts["total"]
            table_active = table_counts["active"]
            table_superseded = table_counts["superseded"]
            table_forgotten = table_counts["forgotten"]
            total += table_total
            active += table_active
            superseded += table_superseded
            forgotten += table_forgotten
            entities = entity_count(db, scope_name)
            relations = relation_count(db, scope_name)
            if include_empty_projects or table_active:
                project_stats.append({
                    "project": scope_name,
                    "scope": "global" if scope_name == "global" else "project",
                    "memories": table_active,
                    "total_memories": table_total,
                    "superseded_memories": table_superseded,
                    "forgotten_memories": table_forgotten,
                    "entities": entities,
                    "relations": relations,
                })
        project_stats.sort(
            key=lambda r: (r["scope"] != "global", str(r["project"])))
        db.commit()
    except Exception as e:
        db.rollback()
        return fail(STATUS_ERROR, str(e))
    finally:
        db.close()
    if detail == "slim":
        memories = dedupe_near_duplicates(memories)
    # The key promises recency: sort the selected memories by their last
    # update instead of by score.
    memories.sort(key=lambda m: m.get("updated_at") or _EPOCH, reverse=True)
    memories = paginate(memories, limit, 0).window
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
