import json
import logging
import math
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
from mcp_server.envelope import (
    STATUS_ERROR,
    STATUS_NOT_FOUND,
    STATUS_OK,
    STATUS_PARTIAL,
    fail,
    ok,
)
from mcp_server.graph_store import GraphStore
from mcp_server.inference_gateway import embed
from mcp_server.lifecycle import (
    counts as lifecycle_counts,
)
from mcp_server.lifecycle import (
    is_expired as _is_expired,
)
from mcp_server.lifecycle import (
    is_visible,
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
    _mem_table,
    _norm_entity,
    _project_names,
    _scope_key,
    _scope_of_table,
)
from mcp_server.scoring import _decay_norm
from mcp_server.search import (
    SearchRequest,
    _extra_conditions,
    _search_memories,
    _search_memories_graph_boost,
    fetch_embeddings,
    fetch_pinned_rows,
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

# A hit found only by full-text search (no vector score) is never filtered by
# `min_similarity`. A stopword-only OR match can then reach the caller with a
# relevance around 0.24 and read like an answer. Exact literal hits are exempt:
# they carry `_exact_match` and a 0.85/0.95 relevance floor of their own.
_FTS_ONLY_MIN_RELEVANCE = 0.25

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
    min_relevance: Annotated[
        float | None,
        Field(description="Drop query hits below this combined "
                          "`_relevance_score` (0.0-1.0), the blend of vector "
                          "similarity and full-text rank. Graph-boost-only "
                          "hits carry no relevance score and are never "
                          "dropped by this filter. Null (default) keeps every "
                          "hit that passed the similarity floor; FTS-only "
                          "hits below an implicit 0.25 noise floor are still "
                          "dropped.")] = None,
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
                          "_exact_match, _pinned.")] = None,
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
    include_pinned: Annotated[
        bool,
        Field(description="Backfill the page with the scope's pinned memories. "
                          "A pinned fact satisfies the structural filters "
                          "(entity, tags, provenance, project, as_of, the "
                          "visibility flags) but is never dropped by a "
                          "relevance or decay floor; a missing pin displaces "
                          "the lowest-ranked non-pinned hit. Appended pins "
                          "carry `_pinned: true` and are listed after the "
                          "ranked hits. The answer adds `pinned_included`.")] = False,
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
    as_of: Annotated[
        str | None,
        Field(description="Only memories that were valid at this ISO 8601 "
                          "instant: valid_from <= as_of and (valid_to is null "
                          "or valid_to > as_of).")] = None,
    tags: Annotated[
        list[str] | None,
        Field(description="Only memories carrying every one of these tags "
                          "(AND). Mutually exclusive with `tags_any`.")] = None,
    tags_any: Annotated[
        list[str] | None,
        Field(description="Only memories carrying at least one of these tags "
                          "(OR). Mutually exclusive with `tags`.")] = None,
    source: Annotated[
        str | None,
        Field(description="Only memories whose provenance `source` equals this "
                          "(case-sensitive).")] = None,
    agent_id: Annotated[
        str | None,
        Field(description="Only memories written by this agent id.")] = None,
    min_confidence: Annotated[
        float | None,
        Field(description="Only memories with a confidence at or above this "
                          "value (0.0-1.0). Memories without a confidence are "
                          "excluded.")] = None,
    include_relation_context: Annotated[
        bool,
        Field(description="True (default) includes the relation-context "
                          "sentences of graph edges (entity "
                          "'<from>_<relation>_<to>'); False hides them from "
                          "the ranking.")] = True,
) -> dict[str, Any]:
    """Hybrid vector + full-text search (RRF fusion).

    Project searches also see global memories; global searches see only global
    ones; `project="*"` searches every scope. `include_global=False` restricts a
    project search to project rows.

    `min_similarity` (default 0.30) filters vector hits only: a full-text hit
    survives a poor embedding, so an exact identifier is still found. Pass None
    to disable it. An empty result names every filter that removed something,
    including `best_dropped_by_similarity` (the strongest rejected vector hit);
    retry with `min_similarity=None` when one nearly matched. `min_relevance`
    filters on `_relevance_score`.

    Ambiguous first-stage results are reranked by a local cross-encoder
    (`reranked`); rows then carry `_rerank_score`. A failure sets `rerank_error`
    and keeps the first-stage order. `_similarity` is raw cosine (0..1),
    `_relevance_score` the normalized vector+FTS blend.
    `min_decay_score` filters on the decay score; include_decay=True blends
    recency/frequency into `_final_score` (at most 30%).

    `graph_boost` adds memories of connected entities. Without `graph_entity`,
    `entity` both filters the search and names the expansion point; with it the
    query stays unfiltered. Graph hits have no similarity and carry
    `_graph_boost`; they only fill slots left by query hits.

    Flags: `only_superseded`, `include_superseded`, `include_deleted`,
    `include_expired`, `include_relation_context` (False hides edge sentences),
    `fields`, `detail="slim"`. `include_pinned=True` backfills the page with
    the scope's pinned facts (structural filters still apply, relevance/decay
    floors do not); they carry `_pinned` and the answer adds `pinned_included`.
    `offset` pages; the answer reports `has_more`,
    `next_offset`, `confidence` (high >= 0.60, medium >= 0.45, low, none, graph)
    and `best_relevance`; a low-confidence answer warns. An unknown project is
    named in `warnings` and only the global scope is searched.
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
    if min_relevance is not None and not 0.0 <= min_relevance <= 1.0:
        return fail(STATUS_ERROR, "min_relevance must be between 0.0 and 1.0")
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
        _extra_conditions(updated_after, metadata_filter, as_of=as_of,
                          tags=tags, tags_any=tags_any, source=source,
                          agent_id=agent_id, min_confidence=min_confidence)
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
    pinned_included = 0
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
            include_entity_relation_context=(
                bool(entity) and include_relation_context),
            stats=stats,
            include_expired=include_expired,
            include_deleted=include_deleted,
            as_of=as_of,
            tags=tags,
            tags_any=tags_any,
            source=source,
            agent_id=agent_id,
            min_confidence=min_confidence,
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
                as_of=as_of,
                tags=tags,
                tags_any=tags_any,
                source=source,
                agent_id=agent_id,
                min_confidence=min_confidence,
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

        dropped_by_relevance = 0
        kept = []
        for r in results:
            if not isinstance(r, dict) or "error" in r:
                kept.append(r)
                continue
            relevance = r.get("_relevance_score")
            if relevance is None:
                # Graph-boost-only hits carry no relevance score: the
                # filter cannot judge them and must not drop them.
                kept.append(r)
                continue
            threshold = min_relevance
            if (threshold is None and r.get("_similarity") is None
                    and not r.get("_exact_match")):
                # FTS-only hit: apply the implicit noise floor even when the
                # caller did not ask for a relevance filter.
                threshold = _FTS_ONLY_MIN_RELEVANCE
            if threshold is None or relevance >= threshold - 1e-6:
                kept.append(r)
            else:
                dropped_by_relevance += 1
        results = kept

        # `only_superseded` keeps the invalidated rows and drops the ACTIVE
        # ones, so the dropped count is "not superseded" - naming it
        # `dropped_by_superseded` reported the opposite of what happened.
        dropped_by_superseded = 0
        dropped_by_not_superseded = 0
        if only_superseded:
            kept = []
            for r in results:
                if isinstance(r, dict) and r.get("superseded_by"):
                    kept.append(r)
                else:
                    dropped_by_not_superseded += 1
            results = kept

        paging = paginate(results, limit, offset)
        results = paging.window

        # A pinned row that ranked naturally is still a pin: mark it so the
        # caller (and `compile_context`) can order and report it. Marking
        # before the backfill also keeps the backfill from re-adding it.
        if not only_superseded:
            for r in results:
                if isinstance(r, dict) and r.get("pinned"):
                    r["_pinned"] = True

        pinned_included = 0
        if include_pinned and not only_superseded:
            # A pin is an always-include fact: it is fetched with the
            # structural filters only and merged into the page, so no
            # relevance or decay floor can drop it.
            existing = {
                (r.get("_project") or "global", r.get("id"))
                for r in results if isinstance(r, dict)
            }
            seen_hashes = {
                r.get("content_hash") for r in results
                if isinstance(r, dict) and r.get("content_hash")
            }
            candidates: list[dict] = []
            for pin in fetch_pinned_rows(
                    cur, tables,
                    mem_type=mem_type, entity=entity,
                    include_entity_relation_context=(
                        bool(entity) and include_relation_context),
                    exclude_relation_context=not include_relation_context,
                    updated_after=updated_after,
                    metadata_filter=metadata_filter, as_of=as_of,
                    tags=tags, tags_any=tags_any, source=source,
                    agent_id=agent_id, min_confidence=min_confidence,
                    include_deleted=include_deleted,
                    include_superseded=include_superseded,
                    include_expired=include_expired, limit=limit):
                key = (pin.get("_project") or "global", pin.get("id"))
                if key in existing:
                    continue
                pin_hash = pin.get("content_hash")
                if pin_hash and pin_hash in seen_hashes:
                    continue
                if pin_hash:
                    seen_hashes.add(pin_hash)
                existing.add(key)
                candidates.append(pin)
            if candidates:
                page = list(results)
                for pin in candidates:
                    if len(page) < limit:
                        page.append(pin)
                        continue
                    # The page is full: a missing pin displaces the
                    # lowest-ranked non-pinned hit.
                    evict = next(
                        (i for i in range(len(page) - 1, -1, -1)
                         if not page[i].get("_pinned")), None)
                    if evict is None:
                        break
                    page[evict] = pin
                # Appended pins sit after the ranked hits (stable order).
                page.sort(key=lambda r: 1 if r.get("_pinned") else 0)
                results = page
                pinned_included = sum(
                    1 for r in results
                    if isinstance(r, dict) and r.get("_pinned"))

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
        all_markers=fields is None, include_deleted=include_deleted)
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
    dropped_by_expired = stats.get("dropped_by_expired", 0)
    dropped_by_validity = stats.get("dropped_by_validity", 0)
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
        if dropped_by_expired:
            reasons.append(
                f"{dropped_by_expired} memory/memories were hidden because "
                f"their TTL expired (pass include_expired=true to see them)")
        if dropped_by_validity:
            reasons.append(
                f"{dropped_by_validity} memory/memories were not valid at "
                f"as_of={as_of!r} (their validity window did not contain that "
                f"time; omit as_of or pass a timestamp inside the window)")
        if dropped_by_decay:
            reasons.append(
                f"{dropped_by_decay} hit(s) were below "
                f"min_decay_score={min_decay_score}")
        if dropped_by_relevance:
            reasons.append(
                f"{dropped_by_relevance} hit(s) were below "
                f"min_relevance={min_relevance}")
        if dropped_by_not_superseded:
            reasons.append(
                f"{dropped_by_not_superseded} hit(s) were not superseded "
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
                                    or include_expired or only_superseded
                                    or include_pinned):
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
    if include_pinned:
        payload["pinned_included"] = pinned_included
    # The strongest rejected vector hit: a count alone hides the natural
    # language question that barely missed the threshold. With a non-empty
    # result it is a data field; with none it is the reason list.
    if best_dropped:
        payload["best_dropped_by_similarity"] = dict(best_dropped)
    if dropped_by_coverage:
        payload["dropped_by_coverage"] = dropped_by_coverage
    if dropped_by_expired:
        payload["dropped_by_expired"] = dropped_by_expired
    if dropped_by_validity:
        payload["dropped_by_validity"] = dropped_by_validity
    # Filter counters are reported even when hits survive: an agent must be
    # able to see that the decay or superseded filter removed candidates
    # from a non-empty result instead of silently getting fewer hits.
    if dropped_by_decay:
        payload["dropped_by_decay"] = dropped_by_decay
    if dropped_by_relevance:
        payload["dropped_by_relevance"] = dropped_by_relevance
    if only_superseded:
        # `only_superseded` drops active rows, not superseded ones: the honest
        # counter is `dropped_by_not_superseded`, and `dropped_by_superseded`
        # stays present (0) so a caller reading the key is not misled by its
        # absence.
        payload["dropped_by_superseded"] = 0
        if dropped_by_not_superseded:
            payload["dropped_by_not_superseded"] = dropped_by_not_superseded
    elif dropped_by_superseded:
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
    min_relevance: Annotated[
        float | None,
        Field(description="Drop hits below this combined `_relevance_score` "
                          "(0.0-1.0), the blend of vector similarity and "
                          "full-text rank. Graph-boost-only hits are never "
                          "dropped by this filter. Null (default) applies no "
                          "explicit floor.")] = None,
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
    as_of: Annotated[
        str | None,
        Field(description="Only memories valid at this ISO 8601 instant "
                          "(applies to every query).")] = None,
    tags: Annotated[
        list[str] | None,
        Field(description="Only memories carrying every one of these tags "
                          "(AND). Mutually exclusive with `tags_any`.")] = None,
    tags_any: Annotated[
        list[str] | None,
        Field(description="Only memories carrying at least one of these tags "
                          "(OR). Mutually exclusive with `tags`.")] = None,
    source: Annotated[
        str | None,
        Field(description="Only memories whose provenance `source` equals "
                          "this.")] = None,
    agent_id: Annotated[
        str | None,
        Field(description="Only memories written by this agent id.")] = None,
    min_confidence: Annotated[
        float | None,
        Field(description="Only memories with a confidence at or above this "
                          "value (0.0-1.0).")] = None,
) -> dict[str, Any]:
    """Run up to 25 independent `recall` queries in one call.

    One ranked list per query instead of concatenating intents (which dilutes
    the scores). Shared options apply to every query; each entry reports its own
    `count`, `warnings`, filter counters and `scopes_searched`.
    A failed query (including an empty one) is named in the top-level
    `warnings` and answered in its own entry; the rest still run.
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
            min_similarity=min_similarity, min_relevance=min_relevance,
            fields=fields, detail=detail,
            include_relation_context=include_relation_context,
            as_of=as_of, tags=tags, tags_any=tags_any, source=source,
            agent_id=agent_id, min_confidence=min_confidence)
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
        Literal["updated_at", "created_at", "access_count", "importance",
                "pinned"],
        Field(description="Sort order (default: updated_at, newest first). "
                          "'pinned' lists pinned memories first, then the rest "
                          "newest first.")] = "updated_at",
    as_of: Annotated[
        str | None,
        Field(description="Only memories that were valid at this ISO 8601 "
                          "instant: valid_from <= as_of and (valid_to is null "
                          "or valid_to > as_of).")] = None,
    tags: Annotated[
        list[str] | None,
        Field(description="Only memories carrying every one of these tags "
                          "(AND). Mutually exclusive with `tags_any`.")] = None,
    tags_any: Annotated[
        list[str] | None,
        Field(description="Only memories carrying at least one of these tags "
                          "(OR). Mutually exclusive with `tags`.")] = None,
    source: Annotated[
        str | None,
        Field(description="Only memories whose provenance `source` equals "
                          "this.")] = None,
    agent_id: Annotated[
        str | None,
        Field(description="Only memories written by this agent id.")] = None,
    min_confidence: Annotated[
        float | None,
        Field(description="Only memories with a confidence at or above this "
                          "value (0.0-1.0). Memories without a confidence are "
                          "excluded.")] = None,
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
    """Browse stored memories without a search query, newest first.

    Without `project` (or with `"*"`) every scope is listed; `project="global"`
    lists only the shared table; a project browse also shows global rows unless
    `include_global=False`. `updated_after`, `metadata_filter`, `contains`
    (literal, case-insensitive) and `mem_type` narrow the set; `order_by`
    changes the sort; `limit`/`offset` page and the answer reports
    `total`/`has_more`/`next_offset`.

    `memory_id` is a direct lookup: the answer carries the single `memory` (and
    a one-item `results`) with history markers; a hidden row names the flag that
    reveals it and browse filters are ignored. `memory_ids` is the bulk form
    (max 200; missing ids are reported, `hidden` lists rows behind a flag).
    `include_deleted` shows soft-forgotten rows, `include_superseded` history,
    `include_expired` passed TTLs; the flags are orthogonal.
    `include_relation_context=False` hides edge sentences.

    Identical active content in several scopes is listed once; a warning names
    the hidden copies. `include_duplicates=True` shows every copy. An unknown
    project warns that only the global scope was browsed.
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
        _extra_conditions(updated_after, metadata_filter, as_of=as_of,
                          tags=tags, tags_any=tags_any, source=source,
                          agent_id=agent_id, min_confidence=min_confidence)
    except ValueError as e:
        return fail(STATUS_ERROR, str(e))
    order_value = browse_order_column(order_by)
    if not order_value:
        return fail(STATUS_ERROR,
                    "order_by must be one of updated_at, created_at, "
                    "access_count, importance, pinned")
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
                ("metadata_filter", metadata_filter), ("as_of", as_of),
                ("tags", tags), ("tags_any", tags_any), ("source", source),
                ("agent_id", agent_id), ("min_confidence", min_confidence),
            ) if value is not None
        ]
        if offset:
            ignored.append("offset")
        if order_by != "updated_at":
            ignored.append("order_by")
        results, missing, hidden, errors = _bulk_memory_lookup(
            wanted, raw_project=raw_project,
            include_global=include_global,
            include_superseded=include_superseded,
            include_deleted=include_deleted,
            include_expired=include_expired, fields=fields)
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
                ("metadata_filter", metadata_filter), ("as_of", as_of),
                ("tags", tags), ("tags_any", tags_any), ("source", source),
                ("agent_id", agent_id), ("min_confidence", min_confidence),
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
            as_of=as_of,
            tags=tags,
            tags_any=tags_any,
            source=source,
            agent_id=agent_id,
            min_confidence=min_confidence,
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


def _bulk_memory_lookup(
    wanted: list[str],
    *,
    raw_project: str | None,
    include_global: bool,
    include_superseded: bool,
    include_deleted: bool,
    include_expired: bool,
    fields: list[str] | None,
) -> tuple[list[dict], list[str], list[dict], list[dict]]:
    """Fetch many ids with one query per scope table (no N+1).

    The result keeps the per-id buckets of the singular lookup: found rows,
    missing ids, hidden rows with their reveal flags, and ids that exist in
    several scopes.
    """
    scope = resolve_scope(raw_project, include_global)
    db = MemoryStore()
    try:
        cur = db.cur
        if scope.global_only:
            tables = ["memories"]
        elif scope.project:
            tables = scope.tables(cur, MODE_LOOKUP)
        else:
            tables = _all_mem_tables(cur)
        if not include_global:
            tables = [table for table in tables if table != "memories"]

        matches: dict[str, list[tuple[str, dict]]] = {}
        for table in tables:
            for row in db.all(
                    f"SELECT * FROM {table} WHERE id = ANY(%s)", (wanted,)):
                matches.setdefault(row["id"], []).append((table, row))

        all_scopes = raw_project is None or str(raw_project).strip() == "*"
        results: list[dict] = []
        missing: list[str] = []
        hidden: list[dict] = []
        errors: list[dict] = []
        for mid in wanted:
            found = matches.get(mid)
            if not found:
                missing.append(mid)
                continue
            if all_scopes:
                scopes = []
                for table, _row in found:
                    name = db.scope_of_table(table)
                    if name not in scopes:
                        scopes.append(name)
                if len(scopes) > 1:
                    errors.append({
                        "id": mid, "status": STATUS_ERROR,
                        "error": (f"ambiguous memory id {mid}: it exists in "
                                  f"the scopes {scopes}; pass project=<name> "
                                  f"to say which one to read"),
                        "scopes": scopes,
                    })
                    continue
            table, row = found[0]
            if not is_visible(row, include_deleted, include_superseded,
                              include_expired):
                superseded_by = row.get("superseded_by")
                deleted_at = row.get("deleted_at")
                reasons = []
                flags = []
                if deleted_at is not None:
                    reasons.append("soft-forgotten")
                    flags.append("include_deleted")
                if (superseded_by is not None
                        and superseded_by != SENTINEL_FORGOTTEN):
                    reasons.append("superseded")
                    flags.append("include_superseded")
                if _is_expired(row.get("expires_at")):
                    reasons.append("expired")
                    flags.append("include_expired")
                hidden.append({
                    "id": mid,
                    "scope": db.scope_of_table(table),
                    "reasons": reasons or ["hidden"],
                    "reveal_flags": flags,
                })
                continue
            row["_project"] = db.scope_of_table(table)
            projection_fields = fields
            if include_deleted:
                if projection_fields is None:
                    projection_fields = list(PUBLIC_FIELDS) + ["deleted_at"]
                elif "deleted_at" not in projection_fields:
                    projection_fields = projection_fields + ["deleted_at"]
            item = project_row(row, projection_fields)
            if row.get("deleted_at"):
                item["forgotten"] = True
                if item.get("superseded_by") == SENTINEL_FORGOTTEN:
                    item["superseded_by"] = None
            if _is_expired(row.get("expires_at")) and (
                    fields is None or "expired" in fields):
                item["expired"] = True
            results.append(item)
        db.commit()
        return results, missing, hidden, errors
    finally:
        db.rollback()
        db.close()


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
                    f"to read", scopes=scopes)
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
                    scope=db.scope_of_table(hidden_table),
                    hidden=True, hidden_reasons=reasons or ["hidden"],
                    reveal_flags=flags)
            return fail(STATUS_NOT_FOUND, f"NOT FOUND: {memory_id}")
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
        return fail(STATUS_ERROR, str(e))
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
    as_of: Annotated[
        str | None,
        Field(description="Only memories that were valid at this ISO 8601 "
                          "instant.")] = None,
    tags: Annotated[
        list[str] | None,
        Field(description="Only memories carrying every one of these tags "
                          "(AND). Mutually exclusive with `tags_any`.")] = None,
    tags_any: Annotated[
        list[str] | None,
        Field(description="Only memories carrying at least one of these tags "
                          "(OR). Mutually exclusive with `tags`.")] = None,
    source: Annotated[
        str | None,
        Field(description="Only memories whose provenance `source` equals "
                          "this.")] = None,
    agent_id: Annotated[
        str | None,
        Field(description="Only memories written by this agent id.")] = None,
    min_confidence: Annotated[
        float | None,
        Field(description="Only memories with a confidence at or above this "
                          "value (0.0-1.0).")] = None,
) -> dict[str, Any]:
    """Everything known about one entity: details, linked memories, connections.

    `entity` (alias `name`) is required; `project="*"` is rejected, an unknown
    project warns. `details.type` is 'graph' (raw AGE label in `details.label`)
    or 'memory-only'. `include_global=False` restricts a project lookup to
    project memories. `fields` narrows the returned columns.
    The entity's edge context sentences are included under `total`; set
    `include_relation_context=false` to hide them. `include_superseded` and
    `include_expired` reveal hidden rows. `limit`/`offset` page but `total`
    counts every match. If the name exists in several scopes,
    `scopes_with_entity` names each one and a warning says which scope was
    described; a miss names the other scopes, so repeat with `project=<name>`.
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
    try:
        entity_feature_conditions, entity_feature_params = _extra_conditions(
            None, None, as_of=as_of, tags=tags, tags_any=tags_any,
            source=source, agent_id=agent_id,
            min_confidence=min_confidence)
    except ValueError as e:
        return fail(STATUS_ERROR, str(e))
    db = MemoryStore()
    memories = []
    entity_data = None
    warnings: list[str] = []
    total = 0
    relation_context_count = 0
    scopes_searched: list[str] = []
    memories_by_scope: dict[str, int] = {}
    scope_presence: dict[str, str] = {}
    found_elsewhere: dict[str, str] = {}
    try:
        cur = db.cur
        tables, scopes_searched = (tables_and_scopes(cur, scope, MODE_PROJECT)
                                   if project else (None, ["global"]))
        # A scope that does not exist must not look like an empty scope:
        # the lookup silently falls back to the global table.
        unknown_scope = scope.unknown_warning(cur)
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
            db, target, used_tables, include_superseded, include_expired,
            extra_conditions=entity_feature_conditions,
            extra_params=entity_feature_params)
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
            as_of=as_of,
            tags=tags,
            tags_any=tags_any,
            source=source,
            agent_id=agent_id,
            min_confidence=min_confidence,
        ))
        # An identical active fact in project and global is one row: the page
        # must apply the same rule as the count, otherwise `total`, `count`
        # and the last page disagree (a 12-row page could claim total=11).
        memories, deduped_hidden = dedupe_rows(memories)
        # `entity_row_count` measures the copies the dedupe rule hides across
        # the whole entity, while `_search_memories` may already have
        # collapsed them for the page; report the larger of the two so the
        # warning never claims one scope's rows are present when they are not.
        duplicates_hidden = max(hidden_duplicates, deduped_hidden)
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
        if entity_data is None and not memories:
            # "Not found" in the searched scopes does not mean the name is
            # unknown: the same entity tag can live in another scope the
            # caller did not search (a project table or its graph). Name
            # those scopes, so a plain miss is never mistaken for a typo.
            # Best-effort: a hint failure must degrade to a plain not-found,
            # not turn the whole lookup into an error.
            try:
                for entity_table in _all_mem_tables(cur):
                    table_scope = _scope_of_table(cur, entity_table)
                    if table_scope in scopes_searched:
                        continue
                    if entity_present(db, entity_table, target):
                        found_elsewhere[table_scope] = "memory"
                for entity_project in graph_projects(db, target):
                    scope_name = entity_project or "global"
                    if scope_name in scopes_searched:
                        continue
                    found_elsewhere[scope_name] = "graph"
            except Exception as e:
                logger.warning("get_entity scope hint failed: %s", e)
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
        # The fallback warning matters before the miss is explained: the
        # caller must know the named scope never existed.
        miss_warnings = [unknown_scope] if unknown_scope else []
        if found_elsewhere:
            return fail(
                STATUS_NOT_FOUND,
                f"entity not found: {target} (in the searched scopes: "
                f"{', '.join(scopes_searched)}; it exists in the scope(s) "
                f"{', '.join(sorted(found_elsewhere))} - repeat the call "
                f"with project=<name> for one of them)",
                entity=target,
                scopes_with_entity=found_elsewhere,
                warnings=miss_warnings + [
                    f"entity {target!r} exists in the scope(s) "
                    f"{', '.join(sorted(found_elsewhere))} - repeat the call "
                    f"with project=<name> for one of them"
                ],
            )
        return fail(STATUS_NOT_FOUND, f"entity not found: {target}",
                    entity=target, warnings=miss_warnings)
    if len(scope_presence) > 1:
        entity_data["scopes_with_entity"] = scope_presence
        detail_scope = entity_data.get("project") or "global"
        duplicate_note = (
            f" Identical content in several scopes is listed once "
            f"({duplicates_hidden} hidden copy/copies); pass project=<name> to "
            f"see that scope's copy."
            if duplicates_hidden else "")
        warnings.append(
            f"entity {target!r} exists in several scopes "
            f"({', '.join(sorted(scope_presence))}); `details` describes the "
            f"{detail_scope!r} occurrence and the memories come from "
            f"{', '.join(scopes_searched)}.{duplicate_note}")
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
    if unknown_scope:
        warnings.append(unknown_scope)
    return ok(STATUS_OK, entity=target, details=entity_data,
              memories=projected,
              total=paging.total, relation_context_count=relation_context_count,
              count=len(window), offset=offset,
              has_more=paging.has_more,
              next_offset=paging.next_offset,
              scopes_searched=scopes_searched,
              memories_by_scope=memories_by_scope,
              duplicates_hidden=duplicates_hidden,
              warnings=warnings)


def get_entities(
    entities: Annotated[
        list[str],
        Field(description="Entity names to look up (1-25, duplicates are "
                          "removed). A blank name is an argument error.")],
    project: Annotated[
        str | None,
        Field(description="Project scope. Without it only the global scope is "
                          "used.")] = None,
    include_global: Annotated[
        bool,
        Field(description="True (default) includes global memories; False "
                          "restricts a project lookup to project memories.")] = True,
    include_superseded: Annotated[
        bool,
        Field(description="Also include superseded memories.")] = False,
    include_expired: Annotated[
        bool,
        Field(description="Also include memories whose TTL has passed. They "
                          "carry `expired: true`.")] = False,
    include_relation_context: Annotated[
        bool,
        Field(description="True (default) also lists the relation-context "
                          "sentences of each entity's graph edges.")] = True,
    fields: Annotated[
        list[str] | None,
        Field(description="Only return these memory columns, e.g. "
                          "['id', 'content'].")] = None,
    limit_per_entity: Annotated[
        int,
        Field(description="Maximum number of memories per entity (default 50). "
                          "Each entry reports its own `total` and `has_more`.")] = 50,
    offset_per_entity: Annotated[
        int,
        Field(description="Skip this many memories per entity (default 0).")] = 0,
    as_of: Annotated[
        str | None,
        Field(description="Only memories that were valid at this ISO 8601 "
                          "instant (applies to every entity).")] = None,
    tags: Annotated[
        list[str] | None,
        Field(description="Only memories carrying every one of these tags "
                          "(AND). Mutually exclusive with `tags_any`.")] = None,
    tags_any: Annotated[
        list[str] | None,
        Field(description="Only memories carrying at least one of these tags "
                          "(OR). Mutually exclusive with `tags`.")] = None,
    source: Annotated[
        str | None,
        Field(description="Only memories whose provenance `source` equals "
                          "this.")] = None,
    agent_id: Annotated[
        str | None,
        Field(description="Only memories written by this agent id.")] = None,
    min_confidence: Annotated[
        float | None,
        Field(description="Only memories with a confidence at or above this "
                          "value (0.0-1.0).")] = None,
) -> dict[str, Any]:
    """Look up several entities in one call.

    Each entry answers exactly like `get_entity` (same details, memories,
    cross-scope dedupe and paging), so an agent that already knows the names
    does not need one round trip per entity. An unknown name is a
    `not_found` entry and never aborts the call; the envelope reports
    `count`, `succeeded` and `failed`.

    `project="*"` is rejected: every entry describes exactly one entity.
    Duplicate names are removed, and names are trimmed like in `get_entity`.
    """
    logger.info("get_entities count=%s project=%s", len(entities or []), project)
    if project == "*":
        return fail(
            STATUS_ERROR,
            "project='*' is not supported by get_entities: each entry "
            "describes exactly one entity. Omit `project` for the global "
            "scope or pass one project name",
        )
    if not entities:
        return fail(STATUS_ERROR,
                    "entities is required (1-25 entity names)")
    if len(entities) > 25:
        return fail(
            STATUS_ERROR,
            f"entities accepts at most 25 names, got {len(entities)}")
    fields_error = validate_fields(fields)
    if fields_error:
        return fail(STATUS_ERROR, fields_error)
    if limit_per_entity < 1:
        return fail(STATUS_ERROR, "limit_per_entity must be >= 1")
    if offset_per_entity < 0:
        return fail(STATUS_ERROR, "offset_per_entity must be >= 0")
    names: list[str] = []
    seen: set[str] = set()
    for raw in entities:
        if not isinstance(raw, str):
            return fail(STATUS_ERROR, "entity names must be strings")
        name = raw.strip()
        if not name:
            return fail(STATUS_ERROR, "entity names must not be blank")
        if name in seen:
            continue
        seen.add(name)
        names.append(name)
    results: list[dict[str, Any]] = []
    for name in names:
        payload = get_entity(
            name,
            project=project,
            include_global=include_global,
            include_superseded=include_superseded,
            include_expired=include_expired,
            include_relation_context=include_relation_context,
            fields=fields,
            limit=limit_per_entity,
            offset=offset_per_entity,
            as_of=as_of,
            tags=tags,
            tags_any=tags_any,
            source=source,
            agent_id=agent_id,
            min_confidence=min_confidence,
        )
        if payload.get("ok"):
            entry: dict[str, Any] = {
                "ok": True,
                "status": payload.get("status"),
                "entity": payload.get("entity", name),
                "details": payload.get("details"),
                "memories": payload.get("memories", []),
                "total": payload.get("total", 0),
                "has_more": payload.get("has_more", False),
                "duplicates_hidden": payload.get("duplicates_hidden", 0),
                "scopes_searched": payload.get("scopes_searched"),
                "memories_by_scope": payload.get("memories_by_scope"),
                "warnings": list(payload.get("warnings") or []),
            }
            if payload.get("relation_context_count"):
                entry["relation_context_count"] = payload[
                    "relation_context_count"]
            details = payload.get("details") or {}
            if details.get("scopes_with_entity"):
                entry["scopes_with_entity"] = details["scopes_with_entity"]
        else:
            entry = {
                "ok": False,
                "status": payload.get("status"),
                "entity": name,
                "error": payload.get("error"),
                "warnings": list(payload.get("warnings") or []),
            }
            if payload.get("scopes_with_entity"):
                entry["scopes_with_entity"] = payload["scopes_with_entity"]
        results.append(entry)
    failed_entries = [entry for entry in results if not entry["ok"]]
    succeeded = len(results) - len(failed_entries)
    if not succeeded:
        return fail(
            STATUS_NOT_FOUND,
            f"none of the {len(results)} entities were found",
            count=len(results),
            succeeded=0,
            failed=len(failed_entries),
            results=results,
        )
    warnings: list[str] = []
    if failed_entries:
        warnings.append(
            f"{len(failed_entries)} entity/entities not found: "
            + ", ".join(str(entry["entity"]) for entry in failed_entries)
            + "; the other entries are complete")
    status = STATUS_PARTIAL if failed_entries else STATUS_OK
    return ok(status, count=len(results), succeeded=succeeded,
              failed=len(failed_entries), results=results, warnings=warnings)


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
    """Browse entities, sorted by name.

    Lists graph nodes (created via `relate`) plus names that only exist on
    memories. With `project` the project and global scope are listed unless
    `include_global=False`; without it (or with `"*"`) every scope is listed;
    `"global"` lists only the shared table. Each row names its `project`, and
    `type` is 'graph' (raw AGE label in `label`) or 'memory-only'.

    `entity_type` selects 'graph', 'memory-only' or a raw AGE label; an unknown
    value is refused and names the labels present. `limit`/`offset` page;
    `count` is the page size, `total` the full match count. Synthetic
    relation-context entities are hidden. An unknown project warns that only
    the global scope was browsed.
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
    """Session bootstrap: the most recently changed memories, counts and health.

    Pass `project` for one project (global rows included unless
    `include_global=False`); without it only the global scope is summarized and
    a warning names the other scopes. `project="*"` covers every scope.
    The list holds the most recently changed memories, sorted by update time,
    not a decay ranking. `detail="slim"` (default) returns the session fields
    and removes near-duplicates; "full" returns every public column.

    `projects` names the covered scopes (including "global"),
    `project_stats` carries their sizes (memories, total, superseded, forgotten,
    entities, relations), and `include_empty_projects=False` drops scopes
    without active memories. `active_memories`, `total_memories`,
    `superseded_memories`, `forgotten_memories` and `graph_entities_count`
    describe the covered scopes; a row can be both superseded and forgotten.
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


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two vectors; 0.0 when either is degenerate."""
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def _diversify_context(
    hits: list[dict],
    *,
    diversity: float,
    max_per_entity: int | None,
) -> tuple[list[dict], int, int]:
    """Apply the entity cap and MMR diversity to the ranked hits.

    Pins are an always-include promise: they are never dropped by the entity
    cap nor by MMR (they still count toward the cap). Returns
    (ordered_hits, dropped_by_entity_cap, dropped_by_diversity); a near
    duplicate (identical content hash or cosine >= 0.95) of an already kept
    row is collapsed and counted as diversity.
    """
    dropped_cap = 0
    entity_counts: dict[str, int] = {}
    kept: list[dict] = []
    for hit in hits:
        if not isinstance(hit, dict):
            continue
        is_pin = bool(hit.get("_pinned"))
        entity = hit.get("entity")
        if (max_per_entity is not None and entity is not None and not is_pin
                and entity_counts.get(entity, 0) >= max_per_entity):
            dropped_cap += 1
            continue
        if entity is not None:
            entity_counts[entity] = entity_counts.get(entity, 0) + 1
        kept.append(hit)

    # `diversity=0` must be a strict no-op: the hash collapse below is part of
    # the diversity feature, so it only runs when diversity is on.
    if diversity <= 0:
        return kept, dropped_cap, 0

    dropped_div = 0
    collapsed: list[dict] = []
    seen_hashes: set[str] = set()
    for hit in kept:
        content_hash = hit.get("content_hash")
        if (content_hash and content_hash in seen_hashes
                and not hit.get("_pinned")):
            dropped_div += 1
            continue
        if content_hash:
            seen_hashes.add(content_hash)
        collapsed.append(hit)

    pins = [h for h in collapsed if h.get("_pinned")]
    rest = [h for h in collapsed if not h.get("_pinned")]
    vectors: dict[str, list[float]] = {}
    ids = [h.get("id") for h in rest if h.get("id")]
    if ids:
        tables: list[str] = []
        for hit in rest:
            table = _mem_table(hit.get("_project") or hit.get("project"))
            if table not in tables:
                tables.append(table)
        store = MemoryStore()
        try:
            vectors = fetch_embeddings(store.cur, tables, ids)
        except Exception:
            vectors = {}
        finally:
            store.close()

    ordered: list[dict] = []
    chosen: list[list[float]] = []
    remaining = list(rest)
    while remaining:
        best_idx = 0
        best_score: float | None = None
        for i, row in enumerate(remaining):
            relevance = float(row.get("_relevance_score") or 0.0)
            vec = vectors.get(str(row.get("id")))
            penalty = 0.0
            if vec is not None and chosen:
                penalty = max(_cosine(vec, cv) for cv in chosen)
            score = (1.0 - diversity) * relevance - diversity * penalty
            if best_score is None or score > best_score:
                best_score = score
                best_idx = i
        hit = remaining.pop(best_idx)
        vec = vectors.get(str(hit.get("id")))
        if vec is not None and any(_cosine(vec, cv) >= 0.95 for cv in chosen):
            dropped_div += 1
            continue
        ordered.append(hit)
        if vec is not None:
            chosen.append(vec)
    return pins + ordered, dropped_cap, dropped_div


def compile_context(
    query: Annotated[
        str,
        Field(description="Search text used to select and rank the memories "
                          "that go into the context block. Non-empty.")],
    project: Annotated[
        str | None,
        Field(description="Project scope. Without it only global memories are "
                          "searched; '*' searches every scope.")] = None,
    include_global: Annotated[
        bool,
        Field(description="True (default) lets a project search also see "
                          "global memories.")] = True,
    max_tokens: Annotated[
        int,
        Field(description="Token budget of the context block (256-32000, "
                          "default 4000). The block never exceeds it.")] = 4000,
    limit: Annotated[
        int,
        Field(description="Maximum number of ranked memories to consider "
                          "(default 50).")] = 50,
    entity: Annotated[
        str | None,
        Field(description="Only consider memories of this entity.")] = None,
    metadata_filter: Annotated[
        dict[str, Any] | None,
        Field(description="Only memories whose metadata contains these "
                          "key/value pairs (same comparison as `recall`).")] = None,
    updated_after: Annotated[
        str | None,
        Field(description="Only memories changed at or after this ISO date, "
                          "e.g. '2026-10-01'.")] = None,
    hybrid: Annotated[
        bool,
        Field(description="True (default) fuses vector and full-text "
                          "retrieval.")] = True,
    include_decay: Annotated[
        bool,
        Field(description="Blend recency/frequency into the ranking (default "
                          "true).")] = True,
    graph_boost: Annotated[
        bool,
        Field(description="Add memories of graph-connected entities. Needs "
                          "`entity` or `graph_entity`.")] = False,
    graph_entity: Annotated[
        str | None,
        Field(description="Entity to expand the graph neighbourhood from "
                          "without restricting the query.")] = None,
    as_of: Annotated[
        str | None,
        Field(description="Only memories that were valid at this ISO 8601 "
                          "instant.")] = None,
    tags: Annotated[
        list[str] | None,
        Field(description="Only memories carrying every one of these tags "
                          "(AND). Mutually exclusive with `tags_any`.")] = None,
    tags_any: Annotated[
        list[str] | None,
        Field(description="Only memories carrying at least one of these tags "
                          "(OR). Mutually exclusive with `tags`.")] = None,
    source: Annotated[
        str | None,
        Field(description="Only memories whose provenance `source` equals "
                          "this.")] = None,
    agent_id: Annotated[
        str | None,
        Field(description="Only memories written by this agent id.")] = None,
    min_confidence: Annotated[
        float | None,
        Field(description="Only memories with a confidence at or above this "
                          "value (0.0-1.0).")] = None,
    include_pinned: Annotated[
        bool,
        Field(description="True (default) puts the scope's pinned memories at "
                          "the top of the context (they satisfy the structural "
                          "filters but no relevance or decay floor). Items "
                          "carry `pinned: true` and the answer adds "
                          "`pinned_included`.")] = True,
    min_relevance: Annotated[
        float | None,
        Field(description="Drop matches below this combined `_relevance_score` "
                          "(0.0-1.0) before packing. Pins are never dropped. "
                          "Null (default) keeps every hit the search returned.")] = None,
    min_similarity: Annotated[
        float | None,
        Field(description="Vector similarity floor passed to the internal "
                          "search (default 0.30, just above the embedding "
                          "noise floor). Null disables it.")] = 0.30,
    diversity: Annotated[
        float,
        Field(description="MMR diversity weight (0.0-1.0, default 0.3: the "
                          "block prefers results unlike the ones already "
                          "chosen, and near-duplicates (identical content hash "
                          "or cosine >= 0.95) are collapsed). 0.0 is a strict "
                          "no-op that reproduces the plain ranking.")] = 0.3,
    max_per_entity: Annotated[
        int | None,
        Field(description="Cap on how many memories one entity may contribute "
                          "(default None = no cap). Pins count toward the cap "
                          "but are never evicted by it.")] = None,
) -> dict[str, Any]:
    """Assemble a token-bounded context block from ranked memories.

    Runs the same hybrid search as `recall` (same scope, filter and dedupe
    semantics), then packs the hits in ranking order into one text block that
    never exceeds `max_tokens`. With `include_pinned` (default) the pinned
    facts go first. With `diversity` (default 0.3) near-duplicates collapse and
    the block prefers hits unlike the ones already chosen; `max_per_entity`
    caps one entity's contribution. The estimate is conservative (about four
    characters per token plus a fixed per-item overhead), so callers get a
    context they can paste into a prompt without blowing its budget.
    """
    try:
        if not isinstance(query, str) or not query.strip():
            return fail(STATUS_ERROR, "query must be a non-empty string")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
            return fail(STATUS_ERROR, "max_tokens must be an integer")
        if not 256 <= max_tokens <= 32000:
            return fail(STATUS_ERROR,
                        "max_tokens must be between 256 and 32000")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            return fail(STATUS_ERROR, "limit must be a positive integer")
        for name, value in (("min_relevance", min_relevance),
                            ("min_similarity", min_similarity)):
            if value is not None and (isinstance(value, bool)
                                      or not isinstance(value, (int, float))
                                      or not 0.0 <= value <= 1.0):
                return fail(STATUS_ERROR,
                            f"{name} must be between 0.0 and 1.0")
        if (isinstance(diversity, bool)
                or not isinstance(diversity, (int, float))
                or not 0.0 <= diversity <= 1.0):
            return fail(STATUS_ERROR, "diversity must be between 0.0 and 1.0")
        if max_per_entity is not None and (
                isinstance(max_per_entity, bool)
                or not isinstance(max_per_entity, int)
                or max_per_entity < 1):
            return fail(STATUS_ERROR,
                        "max_per_entity must be a positive integer")
        search = recall(
            query=query,
            project=project,
            include_global=include_global,
            limit=min(limit, 200),
            offset=0,
            hybrid=hybrid,
            include_decay=include_decay,
            graph_boost=graph_boost,
            graph_entity=graph_entity,
            entity=entity,
            metadata_filter=metadata_filter,
            updated_after=updated_after,
            as_of=as_of,
            tags=tags,
            tags_any=tags_any,
            source=source,
            agent_id=agent_id,
            min_confidence=min_confidence,
            include_pinned=include_pinned,
            min_relevance=min_relevance,
            min_similarity=min_similarity,
            detail="full",
        )
        if not search.get("ok"):
            return search
    except Exception as e:
        return fail(STATUS_ERROR, str(e))

    hits = search.get("results") or []
    if include_pinned:
        # Pins are an always-include promise: they lead the context block so a
        # token cut can never drop them behind a ranked hit.
        hits = ([h for h in hits if isinstance(h, dict) and h.get("_pinned")]
                + [h for h in hits
                   if not (isinstance(h, dict) and h.get("_pinned"))])

    dropped_by_entity_cap = 0
    dropped_by_diversity = 0
    if diversity > 0 or max_per_entity is not None or include_pinned:
        hits, dropped_by_entity_cap, dropped_by_diversity = _diversify_context(
            hits, diversity=diversity, max_per_entity=max_per_entity)
    # A fixed overhead covers the header line, separators and the tokenizer's
    # per-message cost; the content estimate stays deliberately pessimistic so
    # the block never overruns max_tokens.
    item_overhead = 8
    used = 0
    truncated = False
    items: list[dict[str, Any]] = []
    blocks: list[str] = []
    for index, hit in enumerate(hits, start=1):
        content = str(hit.get("content") or "")
        header = (
            f"[{index}] entity={hit.get('entity') or '-'} "
            f"project={hit.get('project') or 'global'} "
            f"type={hit.get('mem_type') or 'fact'} id={hit.get('id')}"
        )
        cost = item_overhead + (len(header) + 3) // 4
        text_cost = (len(content) + 3) // 4
        if used + cost + text_cost > max_tokens:
            remaining = max_tokens - used - cost
            if remaining >= 16 and not blocks:
                # Keep at least a slice of the best hit instead of an empty
                # context when a single long memory exceeds the whole budget.
                keep = max(remaining, 16) * 4
                content = content[:keep].rstrip()
                text_cost = (len(content) + 3) // 4
                truncated = True
            else:
                truncated = True
                break
        used += cost + text_cost
        blocks.append(f"{header}\n{content}")
        items.append({
            "id": hit.get("id"),
            "content": content,
            "entity": hit.get("entity"),
            "mem_type": hit.get("mem_type"),
            "project": hit.get("project"),
            "pinned": bool(hit.get("_pinned")),
            "_relevance_score": hit.get("_relevance_score"),
            "_final_score": hit.get("_final_score"),
        })
        if used >= max_tokens:
            truncated = index < len(hits)
            break

    warnings = list(search.get("warnings") or [])
    if not blocks:
        warnings.append(
            f"no memories matched the query in scope "
            f"{search.get('scopes_searched') or 'global'}")
    elif truncated:
        warnings.append(
            f"context truncated at {max_tokens} tokens; "
            f"{len(hits) - len(items)} memory/memories did not fit")
    pinned_included = sum(1 for item in items if item.get("pinned"))
    dropped_pins = (sum(1 for h in hits
                        if isinstance(h, dict) and h.get("_pinned"))
                    - pinned_included)
    if dropped_pins > 0:
        truncated = True
        warnings.append(
            f"{dropped_pins} pinned memory/memories did not fit the "
            f"{max_tokens}-token budget; raise max_tokens to include them")
    if any(hit.get("_pinned") for hit in hits) and not include_pinned:
        warnings.append(
            "pinned memories exist in the scope but include_pinned=false "
            "was requested; they are not forced into the context")
    context = "\n\n".join(blocks)
    return ok(
        STATUS_OK,
        context=context,
        count=len(items),
        items=items,
        used_tokens=used,
        max_tokens=max_tokens,
        truncated=truncated,
        pinned_included=pinned_included,
        dropped_by_relevance=search.get("dropped_by_relevance"),
        dropped_by_similarity=search.get("dropped_by_similarity"),
        dropped_by_diversity=dropped_by_diversity,
        dropped_by_entity_cap=dropped_by_entity_cap,
        diversity=diversity,
        candidates=search.get("candidates"),
        scopes_searched=search.get("scopes_searched"),
        warnings=warnings,
    )
