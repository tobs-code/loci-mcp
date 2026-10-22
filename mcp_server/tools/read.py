import json
import logging
from datetime import datetime, timezone
from typing import Annotated, Any, Literal

from pydantic import Field

from mcp_server.db import get_conn
from mcp_server.config import DEFAULT_MIN_SIMILARITY, MEMORY_TYPE_HELP
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
    _norm_project,
    _project_names,
    _retrieval_tables,
    _scope_key,
    _scope_label,
)
from mcp_server.scoring import _decay_norm
from mcp_server.search import (
    _extra_conditions,
    _search_memories,
    _search_memories_graph_boost,
)
from mcp_server.storage import _log_access
from mcp_server.tools.maintenance import _health_check

logger = logging.getLogger(__name__)

_EPOCH = datetime.min.replace(tzinfo=timezone.utc)


def _escape_like(value: str) -> str:
    """Escape the wildcards of a LIKE/ILIKE pattern."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


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
                          "indexes.")],
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
                          "searched.")] = None,
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
    graph_boost: Annotated[
        bool,
        Field(description="Add memories of entities connected in the graph. "
                          "Needs `graph_entity` (or `entity` as fallback). "
                          "Those hits have no similarity and are marked with "
                          "`_graph_boost`.")] = False,
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
                          "(0.0-1.0, default 0.35). Full-text hits are never "
                          "dropped by this filter: an exact keyword match "
                          "survives even when the embedding model scores it "
                          "low. Use null to disable the filter.")] = DEFAULT_MIN_SIMILARITY,
    fields: Annotated[
        list[str] | None,
        Field(description="Only return these columns, e.g. ['id', 'content']. "
                          "Valid names: id, content, mem_type, scope, entity, "
                          "importance, metadata, access_count, superseded_by, "
                          "created_at, updated_at, project.")] = None,
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
                          "key/value pairs, e.g. {'ticket': 'QA-1'}.")] = None,
) -> dict[str, Any]:
    """Search memories using hybrid vector + fulltext search (RRF fusion).

    Project searches also see global memories; global searches see only global ones.
    Set graph_boost=True to add the memories of entities connected in the graph.
    `graph_entity` names the entity to expand from and leaves the query search
    unfiltered ("query hits + graph neighbourhood"); without it `entity` is
    used for the expansion *and* filters the search, so unrelated hits are
    dropped. Graph hits have no similarity: their `_similarity` and
    `_relevance_score` are null and they carry `_graph_boost` instead.
    `min_similarity` defaults to 0.35 and only applies to the vector retriever;
    a full-text hit is kept even when its embedding is a poor match, so an
    exact identifier ("ZEBRAFISH_TOKEN") is found even if the embedding model
    handles that word badly. Pass `None` to disable the filter. The response
    reports `candidates` and `dropped_by_similarity`, and warns when the
    filter removed everything.
    `_relevance_score` is normalized to 0..1 and combines both signals: the
    cosine similarity of the vector hit and the rank of the full-text hit.
    A memory that both retrievers found scores higher than either signal
    alone. `_similarity` is the raw cosine similarity (0..1): a natural
    language question that matches scores ~0.6-0.95, a bare keyword that
    occurs verbatim in the memory only ~0.3-0.6.
    `_final_score` is the ranking score in 0..1: relevance blended with decay
    plus the graph boost.
    Set include_decay=True to blend in recency/frequency (max 30% of the score).
    min_decay_score (0..1) filters on the normalized decay score; a fresh,
    important memory scores ~0.9, an old one ~0.5.
    Set only_superseded=True to return only invalidated facts.
    Set fields=["id", "content"] to receive only the named columns.
    Set include_global=False to restrict a project search to project memories.
    `updated_after` and `metadata_filter` narrow the result set.
    `offset` pages through the ranked results (with `limit`); the response
    reports `has_more` and `next_offset`.
    """
    logger.info("recall query='%s' entity=%s hybrid=%s include_decay=%s graph_boost=%s min_similarity=%s",
                query[:80], entity, hybrid, include_decay, graph_boost, min_similarity)
    project = _norm_project(project)
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
                        "entity); no graph boost was applied")
    if only_superseded:
        include_superseded = True
    query_emb = embed([query], mode="query")[0]
    conn = get_conn()
    try:
        cur = conn.cursor()
        
        # Neue RRF-Funktion nutzen
        tables = _retrieval_tables(cur, project)
        if project and not include_global:
            tables = [table for table in tables if table != "memories"]
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
            stats=stats,
        )

        # Graph-enhanced retrieval when enabled
        if graph_boost and boost_entity:
            results = _search_memories_graph_boost(
                results, boost_entity, fetch_limit, project, include_decay,
                include_global=include_global,
                updated_after=updated_after, metadata_filter=metadata_filter,
            )
            if results and all(r.get("_relevance_score") is None
                               for r in results if isinstance(r, dict)):
                warnings.append(
                    "no query match found: the results come from the graph "
                    "neighbourhood of the entity, not from the query"
                )

        # Filter on decay *before* the access counters are bumped: the filter
        # must see the stored access_count, not the one of this very query.
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
            results = kept

        if only_superseded:
            results = [r for r in results
                       if isinstance(r, dict) and r.get("superseded_by")]

        has_more = len(results) > offset + limit
        results = results[offset:offset + limit]

        # Update access counters (decay frequency) only for the rows this
        # page actually returns: skipped matches were not read.
        hit_ids = [r.get("id") for r in results if isinstance(r, dict) and r.get("id")]
        _log_access(cur, hit_ids, project)
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

    rows = [project_row(r, fields) for r in results if isinstance(r, dict)]

    # An empty result set is ambiguous: the memory may simply not exist, or
    # a filter may have dropped it. Say which one it was.
    filtered_out = stats.get("dropped_by_similarity", 0)
    candidates = stats.get("candidates", 0)
    if not rows and offset > 0 and candidates:
        warnings.append(
            f"no results on this page: offset={offset} is beyond the "
            f"{candidates} ranked candidate(s); lower the offset"
        )
    elif not rows and filtered_out:
        warnings.append(
            f"no results: {filtered_out} candidate(s) were below "
            f"min_similarity={min_similarity}; pass min_similarity=null or a "
            f"lower value to see them"
        )
    elif not rows and candidates:
        warnings.append(
            f"no results: {candidates} candidate(s) matched but every one of "
            f"them was filtered out (min_decay_score, only_superseded or the "
            f"scope filter)"
        )
    elif not rows:
        warnings.append("no results: nothing in this scope matches the query")
    if min_similarity is None and rows:
        weak = [r for r in rows
                if (r.get("_similarity") is not None
                    and r["_similarity"] < 0.35)]
        if weak:
            warnings.append(
                f"the min_similarity filter is disabled: {len(weak)} of "
                f"{len(rows)} result(s) have _similarity < 0.35 and may be "
                f"only marginally relevant to the query")
    payload: dict[str, Any] = {
        "candidates": candidates,
        "dropped_by_similarity": filtered_out,
        "offset": offset,
        "has_more": has_more,
        "next_offset": (offset + limit) if has_more else None,
    }
    if stats.get("fulltext_only"):
        payload["fulltext_only"] = stats["fulltext_only"]
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
        Field(description="Project scope. Without it every scope is listed.")] = None,
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
                          "key/value pairs.")] = None,
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
        Field(description="Only list the memory with this id. Use it as a "
                          "direct id lookup across the selected scope(s).")] = None,
    include_deleted: Annotated[
        bool,
        Field(description="Also list soft-forgotten memories (recognizable by "
                          "a non-null `deleted_at`). Use it before "
                          "`restore_memory`.")] = False,
) -> dict[str, Any]:
    """Browse stored memories without a search query.

    Newest first. Project scope also lists global memories, like `recall`.
    Set `fields` to narrow the returned columns. Set `include_global=False`
    to restrict a project browse to project memories. `updated_after` and
    `metadata_filter` narrow the result set, `order_by` changes the sorting
    and `contains` filters on a literal substring of the content.
    `offset` pages through a scope that has more matches than `limit`.
    Pass `memory_id` for a direct id lookup and `include_deleted=True` to
    include soft-forgotten memories (they carry `deleted_at`).
    """
    logger.info("list_memories entity=%s mem_type=%s limit=%s offset=%s project=%s",
                entity, mem_type, limit, offset, project)
    project = _norm_project(project)
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
    conn = get_conn()
    rows: list[dict] = []
    try:
        cur = conn.cursor()
        tables = (_retrieval_tables(cur, project) if project
                  else _all_mem_tables(cur))
        if project and not include_global:
            tables = [table for table in tables if table != "memories"]
        conditions: list[str] = []
        params: list[Any] = []
        if include_deleted:
            # Everything, including superseded history and forgotten rows.
            conditions.append("TRUE")
        elif include_superseded:
            conditions.append("deleted_at IS NULL")
        else:
            conditions.append("superseded_by IS NULL AND deleted_at IS NULL")
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
        # Oversample: the same content can live in several scopes and the
        # cross-scope dedup below removes the copies again. Without the
        # reserve a page could come back short.
        fetch = (offset + limit) * 2
        total = 0
        for table in tables:
            cur.execute(f"SELECT COUNT(*) FROM {table} WHERE {where}", params)
            total += cur.fetchone()[0]
            project_names = None if table == "memories" else _project_names(cur)
            cur.execute(
                f"SELECT * FROM {table} WHERE {where} "
                f"ORDER BY {order_column} DESC, id LIMIT %s",
                params + [fetch],
            )
            columns = [d[0] for d in cur.description]
            table_rows = cur.fetchall()
            for row in table_rows:
                item = dict(zip(columns, row))
                if table == "memories":
                    item["_project"] = "global"
                else:
                    item["_project"] = project_names.get(table[4:], table[4:])
                rows.append(item)
        conn.commit()
    except Exception as e:
        conn.rollback()
        return fail(STATUS_ERROR, str(e))
    finally:
        conn.close()
    # The rows come from several tables: sort globally before trimming.
    unique = list(rows)
    unique.sort(key=lambda r: (r.get(order_column) is not None,
                                r.get(order_column)), reverse=True)
    # Cross-scope dedup: the same content can live in a project and in the
    # global scope. The project copy wins, like it does in `recall`.
    by_hash: dict[str, dict] = {}
    without_hash: list[dict] = []
    for row in unique:
        c_hash = row.get("content_hash") or ""
        if not c_hash:
            without_hash.append(row)
            continue
        current = by_hash.get(c_hash)
        if current is None:
            by_hash[c_hash] = row
            continue
        if (current.get("_project") == "global"
                and row.get("_project") != "global"):
            by_hash[c_hash] = row
    unique = list(by_hash.values()) + without_hash
    unique.sort(key=lambda r: (r.get(order_column) is not None,
                                r.get(order_column)), reverse=True)
    window = unique[offset:offset + limit]
    has_more = len(unique) > offset + limit
    projection_fields = fields
    if include_deleted and projection_fields is None:
        # Without this the caller could not tell a forgotten row from an
        # active one.
        projection_fields = list(PUBLIC_FIELDS) + ["deleted_at"]
    return ok(STATUS_OK, count=len(window), total=total, offset=offset,
              has_more=has_more,
              next_offset=(offset + limit) if has_more else None,
              results=[project_row(r, projection_fields) for r in window])


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
) -> dict[str, Any]:
    """Get everything known about an entity, including all linked memories.

    The entity can be passed as `entity` or `name`.
    Memory-only entities have no graph node; `details.type` is `memory-only`.
    Set fields=["id", "content"] to receive only the named memory columns.
    Set include_global=False to restrict a project lookup to project memories.
    """
    target = entity or name
    logger.info("get_entity name=%s", target)
    project = _norm_project(project)
    if not target:
        return fail(STATUS_ERROR, "entity is required")
    fields_error = validate_fields(fields)
    if fields_error:
        return fail(STATUS_ERROR, fields_error)
    conn = get_conn()
    memories = []
    entity_data = None
    try:
        cur = conn.cursor()
        tables = _retrieval_tables(cur, project) if project else None
        if project and not include_global:
            tables = [table for table in tables if table != "memories"]
        memories = _search_memories(cur, None, None, None, target, 100, project,
                                    include_superseded=include_superseded,
                                    tables=tables)
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
            entity_data = {"id": row[0], "name": row[1], "type": row[2],
                           "project": row[3], "connections": []}
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
    return ok(STATUS_OK, entity=target, details=entity_data,
              memories=[project_row(m, fields) for m in memories])


def list_entities(entity_type: Annotated[
        str | None,
        Field(description="Only list entities of this graph type (e.g. the "
                          "project label).")] = None,
    limit: Annotated[
        int,
        Field(description="Maximum number of results (default 50).")] = 50,
    project: Annotated[
        str | None,
        Field(description="Project scope. Without it every scope is listed.")] = None,
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
    isolation. Without `project` every scope is listed. Each row carries its
    `project`; `graph_node` is false for memory-only entities. Synthetic
    relation-context entities (`A_KNOWS_B`) are not listed. The list is
    sorted by name; `offset` pages through it.
    """
    logger.info("list_entities entity_type=%s limit=%s offset=%s project=%s",
                entity_type, limit, offset, project)
    project = _norm_project(project)
    if limit < 1:
        return fail(STATUS_ERROR, "limit must be >= 1")
    if offset < 0:
        return fail(STATUS_ERROR, "offset must be >= 0")
    conn = get_conn()
    try:
        cur = conn.cursor()
        tables = (_retrieval_tables(cur, project) if project
                  else _all_mem_tables(cur))
        if project and not include_global:
            tables = [table for table in tables if table != "memories"]
        names = _project_names(cur)
        scope_of = {t: (names.get(t[len("mem_"):], t[len("mem_"):])
                        if t != "memories" else "global") for t in tables}

        if project and include_global:
            conditions = ["(project = %s OR project = 'global')"]
            params: list[Any] = [project]
        elif project:
            conditions = ["project = %s"]
            params = [project]
        else:
            conditions = ["TRUE"]
            params = []
        if entity_type:
            conditions.append("type = %s")
            params.append(entity_type)
        where = " AND ".join(conditions)
        cur.execute(f"SELECT COUNT(*) FROM entities WHERE {where}", params)
        total = cur.fetchone()[0]
        cur.execute(
            f"SELECT id, name, type, project FROM entities "
            f"WHERE {where} ORDER BY name LIMIT %s",
            params + [limit * 2 + offset],
        )
        rows = [{"id": r[0], "name": r[1], "type": r[2], "project": r[3],
                 "graph_node": True} for r in cur.fetchall()]

        seen = {(r["name"], r["project"]) for r in rows}
        for table in tables:
            scope = scope_of[table]
            # Memory-only entities: a name that occurs in the memories of this
            # scope but has no graph node there. Synthetic relation-context
            # entities (e.g. "A_KNOWS_B") are excluded: they exist only to make
            # the edge context searchable, not as browsable knowledge. An
            # entity that also has a real memory stays visible.
            context_flag = "COALESCE(metadata->>'relation_context', 'false') <> 'true'"
            cur.execute(
                f"SELECT COUNT(*) FROM (SELECT entity FROM {table} "
                f"WHERE entity IS NOT NULL AND superseded_by IS NULL "
                f"AND deleted_at IS NULL "
                f"GROUP BY entity HAVING bool_or({context_flag})) e "
                f"WHERE NOT EXISTS (SELECT 1 FROM entities en "
                f"WHERE en.name = e.entity AND en.project = %s)",
                (scope,),
            )
            total += cur.fetchone()[0]
            cur.execute(
                f"SELECT entity FROM {table} "
                f"WHERE entity IS NOT NULL AND superseded_by IS NULL AND deleted_at IS NULL "
                f"GROUP BY entity HAVING bool_or({context_flag})"
            )
            for (ent,) in cur.fetchall():
                if (ent, scope) in seen:
                    continue
                seen.add((ent, scope))
                # Memory-only entity: same row shape as a graph node, so an
                # agent never has to guess which keys exist.
                rows.append({"id": None, "name": ent, "type": "memory-only",
                             "project": scope, "graph_node": False})
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
    return ok(STATUS_OK, count=len(window), total=total, offset=offset,
              has_more=has_more,
              next_offset=(offset + limit) if has_more else None,
              results=window)


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
    """
    all_projects = project == "*"
    project = None if all_projects else _norm_project(project)
    if detail not in ("slim", "full"):
        return fail(STATUS_ERROR, "detail must be 'slim' or 'full'")
    if limit < 1:
        return fail(STATUS_ERROR, "limit must be >= 1")
    conn = get_conn()
    memories = []
    total = 0
    active = 0
    projects: list[str] = []
    try:
        cur = conn.cursor()
        tables = (_all_mem_tables(cur) if all_projects else
                  _retrieval_tables(cur, project) if project else ["memories"])
        if (project or all_projects) and not include_global:
            tables = [table for table in tables if table != "memories"]
        # recent_first: the bootstrap shows what changed last, not what was
        # read most often.
        memories = _search_memories(cur, None, None, None, None, limit, project,
                                    include_decay=True, tables=tables,
                                    recent_first=True)
        names = _project_names(cur)
        projects = [names.get(t[len("mem_"):], t[len("mem_"):])
                    for t in tables if t != "memories"]
        if all_projects and include_global and "memories" in tables:
            # The bootstrap covers every scope, so the shared one is a project
            # entry too - otherwise the caller cannot see that it was included.
            projects.insert(0, "global")
        for table in tables:
            cur.execute(
                f"SELECT COUNT(*), "
                f"COUNT(*) FILTER (WHERE superseded_by IS NULL AND deleted_at IS NULL) "
                f"FROM {table}"
            )
            table_total, table_active = cur.fetchone()
            total += table_total
            active += table_active
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
        projects=sorted(projects),
        count=len(memories),
        recent_memories=memories,
        graph_entities_count=_count_entities(),
        inference_healthy=_health_check(),
    )
