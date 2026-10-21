import json
import logging
from datetime import datetime, timezone
from typing import Annotated, Any, Literal

from pydantic import Field

from mcp_server.db import get_conn
from mcp_server.config import DEFAULT_MIN_SIMILARITY
from mcp_server.envelope import STATUS_ERROR, STATUS_NOT_FOUND, STATUS_OK, fail, ok
from mcp_server.inference_client import embed
from mcp_server.projection import (
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
        Field(description="Only return memories of this type.")] = None,
    entity: Annotated[
        str | None,
        Field(description="Only return memories of this entity. Required for "
                          "graph_boost.")] = None,
    limit: Annotated[
        int,
        Field(description="Maximum number of results (default 10).")] = 10,
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
        Field(description="Add memories of entities connected to `entity` in the "
                          "graph. Requires `entity`; those hits have no "
                          "similarity and are marked with `_graph_boost`.")] = False,
    min_similarity: Annotated[
        float | None,
        Field(description="Drop vector hits below this cosine similarity "
                          "(0.0-1.0, default 0.4). Use null to disable the "
                          "filter.")] = DEFAULT_MIN_SIMILARITY,
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
    Set graph_boost=True to include related entities in search via graph traversal
    (requires `entity`, otherwise a warning is returned). Graph hits have no
    similarity: their `_similarity` and `_relevance_score` are null and they
    carry `_graph_boost` instead.
    `min_similarity` defaults to 0.4; lower it to broaden results, or pass
    `None` to disable the similarity filter. `_relevance_score` is normalized
    to 0..1 across retrieval modes: RRF for hybrid, cosine similarity for
    vector-only, and normalized text rank for the FTS fallback.
    `_final_score` is the ranking score: relevance plus at most 30% decay and
    the graph boost.
    Set include_decay=True to blend in recency/frequency (max 30% of the score).
    min_decay_score (0..1) filters on the normalized decay score; a fresh,
    important memory scores ~0.9, an old one ~0.5.
    Set only_superseded=True to return only invalidated facts.
    Set fields=["id", "content"] to receive only the named columns.
    Set include_global=False to restrict a project search to project memories.
    `updated_after` and `metadata_filter` narrow the result set.
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
    if metadata_filter is not None and not isinstance(metadata_filter, dict):
        try:
            metadata_filter = _normalize_metadata_filter(metadata_filter)
        except ValueError as e:
            return fail(STATUS_ERROR, str(e))
    try:
        _extra_conditions(updated_after, metadata_filter)
    except ValueError as e:
        return fail(STATUS_ERROR, str(e))
    if graph_boost and not entity:
        warnings.append("graph_boost requires an entity; no graph boost was applied")
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
        results = _search_memories(
            cur, query_emb, query if (hybrid or query_emb is None) else None,
            mem_type, entity, limit, project,
            include_decay, include_superseded=include_superseded,
            min_similarity=min_similarity,
            hybrid=hybrid, tables=tables,
            updated_after=updated_after, metadata_filter=metadata_filter,
        )

        # Graph-enhanced retrieval when enabled
        if graph_boost and entity:
            results = _search_memories_graph_boost(
                results, entity, limit, project, include_decay,
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

        # Update access counters (decay frequency)
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

    if only_superseded:
        results = [r for r in results
                   if isinstance(r, dict) and r.get("superseded_by")]

    results = results[:limit]
    rows = [project_row(r, fields) for r in results if isinstance(r, dict)]
    return ok(STATUS_OK, count=len(rows), results=rows, warnings=warnings)


def list_memories(
    entity: Annotated[
        str | None,
        Field(description="Only list memories of this entity.")] = None,
    mem_type: Annotated[
        Literal["fact", "decision", "learning", "preference", "pattern", "incident",
                "runbook", "context", "observation", "convention", "command", "note"]
        | None,
        Field(description="Only list memories of this type.")] = None,
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
) -> dict[str, Any]:
    """Browse stored memories without a search query.

    Newest first. Project scope also lists global memories, like `recall`.
    Set `fields` to narrow the returned columns. Set `include_global=False`
    to restrict a project browse to project memories. `updated_after` and
    `metadata_filter` narrow the result set, `order_by` changes the sorting.
    """
    logger.info("list_memories entity=%s mem_type=%s limit=%s project=%s",
                entity, mem_type, limit, project)
    project = _norm_project(project)
    if limit < 1:
        return fail(STATUS_ERROR, "limit must be >= 1")
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
        if include_superseded:
            conditions.append("deleted_at IS NULL")
        else:
            conditions.append("superseded_by IS NULL AND deleted_at IS NULL")
        if entity:
            conditions.append("entity = %s")
            params.append(entity)
        if mem_type:
            conditions.append("mem_type = %s")
            params.append(mem_type)
        conditions.extend(extra_conditions)
        params.extend(extra_params)
        where = " AND ".join(conditions)
        for table in tables:
            project_names = None if table == "memories" else _project_names(cur)
            cur.execute(
                f"SELECT * FROM {table} WHERE {where} "
                f"ORDER BY {order_column} DESC, id LIMIT %s",
                params + [limit],
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
    seen_hashes: set[str] = set()
    unique: list[dict] = []
    for row in rows:
        c_hash = row.get("content_hash") or ""
        if c_hash and c_hash in seen_hashes:
            continue
        if c_hash:
            seen_hashes.add(c_hash)
        unique.append(row)
    # The rows come from several tables: sort globally before trimming.
    unique.sort(key=lambda r: (r.get(order_column) is not None,
                                r.get(order_column)), reverse=True)
    unique = unique[:limit]
    return ok(STATUS_OK, count=len(unique),
              results=[project_row(r, fields) for r in unique])


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
) -> dict[str, Any]:
    """Browse entities.

    Returns graph entities (created via `relate`) plus entity names that only
    exist on memories. With `project`, that project and global scope are
    listed by default; set `include_global=False` for strict project
    isolation. Without `project` every scope is listed. Each row carries its
    `project`; `graph_node` is false for memory-only entities.
    """
    logger.info("list_entities entity_type=%s limit=%s project=%s",
                entity_type, limit, project)
    project = _norm_project(project)
    if limit < 1:
        return fail(STATUS_ERROR, "limit must be >= 1")
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
        cur.execute(
            f"SELECT id, name, type, project FROM entities "
            f"WHERE {' AND '.join(conditions)} ORDER BY name LIMIT %s",
            params + [limit],
        )
        rows = [{"id": r[0], "name": r[1], "type": r[2], "project": r[3],
                 "graph_node": True} for r in cur.fetchall()]

        seen = {(r["name"], r["project"]) for r in rows}
        for table in tables:
            scope = scope_of[table]
            cur.execute(
                f"SELECT DISTINCT entity FROM {table} "
                f"WHERE entity IS NOT NULL AND superseded_by IS NULL AND deleted_at IS NULL"
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
    rows = rows[:limit]
    return ok(STATUS_OK, count=len(rows), results=rows)


def init_context(project: Annotated[
        str | None,
        Field(description="Project to bootstrap for. '*' bootstraps all "
                          "projects; without a project only global memories "
                          "are summarized.")] = None,
    detail: Annotated[
        Literal["slim", "full"],
        Field(description="'slim' (default) returns only what a session needs "
                          "and removes near-duplicates; 'full' returns every "
                          "column (never internal ones).")] = "slim",
    include_global: Annotated[
        bool,
        Field(description="True (default) includes global memories; False "
                          "excludes them.")] = True,
) -> dict[str, Any]:
    """Call at session start. Returns recent + important bootstrap info + health.

    Pass `project` to bootstrap for one project; global memories are included
    by default. Set `include_global=False` to exclude them. Without a project,
    only global memories are summarized; pass `project="*"` for all projects.
    `detail="slim"` (default) returns only the fields a session needs and
    removes near-duplicates; `detail="full"` returns every column except the
    internal ones. Every memory carries the `project` it belongs to, and the
    list is sorted by the last update.
    """
    all_projects = project == "*"
    project = None if all_projects else _norm_project(project)
    if detail not in ("slim", "full"):
        return fail(STATUS_ERROR, "detail must be 'slim' or 'full'")
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
        memories = _search_memories(cur, None, None, None, None, 10, project,
                                    include_decay=True, tables=tables)
        names = _project_names(cur)
        projects = [names.get(t[len("mem_"):], t[len("mem_"):])
                    for t in tables if t != "memories"]
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
