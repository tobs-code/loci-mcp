import json
import logging
from typing import Annotated, Any, Literal, NotRequired, TypedDict

from pydantic import Field

from mcp_server.config import SENTINEL_FORGOTTEN
from mcp_server.db import get_conn
from mcp_server.envelope import (
    STATUS_DELETED,
    STATUS_ERROR,
    STATUS_NOT_FOUND,
    STATUS_OK,
    fail,
    ok,
)
from mcp_server.inference_client import embed
from mcp_server.scopes import (
    _ag_str,
    _ensure_project_table,
    _mem_table,
    _norm_entity,
    _norm_project,
    _retrieval_tables,
    _safe_ident,
    _scope_key,
    _scope_label,
    _table_exists,
)
from mcp_server.storage import _store_memory, _supersede

logger = logging.getLogger(__name__)


class RelationEdge(TypedDict):
    from_entity: Annotated[str, Field(description="Source entity.")]
    to_entity: Annotated[str, Field(description="Target entity.")]
    relation_type: Annotated[str, Field(description="Relation label.")]
    weight: NotRequired[Annotated[float, Field(
        description="Edge strength in 0.0-1.0 (default 1.0).")]]
    context: NotRequired[Annotated[str | None, Field(
        description="Sentence describing the relation; stored as a memory.")]]
    project: NotRequired[Annotated[str | None, Field(
        description="Project scope for this edge; overrides the tool-level "
                    "project.")]]


def _forget_relation_context(
    cur,
    project: str | None,
    from_entity: str,
    to_entity: str,
    relation_type: str,
) -> int:
    """Hide the active context memory attached to a deleted edge."""
    table = _mem_table(project)
    if not _table_exists(cur, table):
        return 0
    cur.execute(
        f"UPDATE {table} SET superseded_by = %s, deleted_at = NOW() "
        "WHERE deleted_at IS NULL "
        "AND metadata->>'relation_context' = 'true' "
        "AND metadata->>'from' = %s "
        "AND metadata->>'to' = %s "
        "AND metadata->>'relation_type' = %s",
        (SENTINEL_FORGOTTEN, from_entity, to_entity, relation_type),
    )
    return cur.rowcount

def _ensure_graph_entity(cur, name: str, project: str | None,
                         ent_type: str | None = None) -> bool:
    """Create the AGE node and SQL entity row for one scope (idempotent).

    Returns True when the node did not exist before, so an import can report
    how many entities it really created instead of how many it processed.
    """
    label = _scope_label(project)
    scope = _scope_key(project)
    ent_json = json.dumps(name)
    scope_json = json.dumps(scope)
    cur.execute(
        f"SELECT * FROM cypher('memory_graph', $$ "
        f"MATCH (n:{label} {{name: {ent_json}}}) RETURN n $$) AS (n agtype)"
    )
    created = cur.fetchone() is None
    if created:
        cur.execute(
            f"SELECT * FROM cypher('memory_graph', $$ "
            f"CREATE (n:{label} {{name: {ent_json}, project: {scope_json}}}) RETURN n $$) AS (n agtype)"
        )
    cur.execute(
        "INSERT INTO entities (name, type, project) VALUES (%s, %s, %s) "
        "ON CONFLICT (name, project) DO UPDATE SET type = EXCLUDED.type",
        (name, ent_type or label, scope),
    )
    return created


def _validate_weight(value: Any) -> float:
    """Edge weights are strengths in [0, 1].

    They feed directly into the retrieval boost (0.3 * weight), so an
    unbounded weight would let a single edge dominate every score.
    """
    try:
        weight = float(value)
    except (TypeError, ValueError):
        raise ValueError("weight must be a number between 0.0 and 1.0")
    if not 0.0 <= weight <= 1.0:
        raise ValueError("weight must be between 0.0 and 1.0")
    return weight


def _entity_not_found_hint(cur, entity_name: str, project: str | None) -> str:
    """Explain why an entity was not found in the searched graph.

    Agents cannot see the scope of the graph they searched, so the message
    names the scope that was used and points to the likely cause.
    """
    scope = _scope_key(project)
    cur.execute("SELECT DISTINCT project FROM entities WHERE name = %s",
                (entity_name,))
    other_scopes = [r[0] for r in cur.fetchall()]
    memory_only = False
    for table in _retrieval_tables(cur, project):
        if not _table_exists(cur, table):
            continue
        cur.execute(
            f"SELECT 1 FROM {table} WHERE entity = %s "
            "AND superseded_by IS NULL AND deleted_at IS NULL LIMIT 1",
            (entity_name,))
        if cur.fetchone():
            memory_only = True
            break
    if memory_only:
        return (f"entity not found in the {scope} graph: {entity_name} "
                f"(it exists as a memory-only entity; create a relation with "
                f"`relate` to add it to the graph)")
    if other_scopes:
        return (f"entity not found in the {scope} graph: {entity_name} "
                f"(it exists in: {', '.join(sorted(other_scopes))} - pass "
                f"project=<name> to search that graph)")
    if project is None:
        return (f"entity not found in the {scope} graph: {entity_name} "
                f"(without project the global graph is searched; pass "
                f"project=<name> if the entity lives in a project)")
    return (f"entity not found in the {scope} graph: {entity_name} "
            f"(the entity may live in another project; pass that project)")


def _sync_context_weight(cur, project: str | None, from_entity: str,
                         to_entity: str, relation_type: str,
                         weight: float) -> None:
    """Keep the weight in a relation context memory in sync with its edge.

    `relate(context=...)` stores the edge weight in the metadata of the
    context sentence. When the edge is re-weighted later, the sentence must
    follow - otherwise it reports a strength the graph no longer has.
    """
    table = _mem_table(project)
    if not _table_exists(cur, table):
        return
    cur.execute(
        f"UPDATE {table} SET metadata = jsonb_set(metadata, '{{weight}}', %s::jsonb) "
        "WHERE deleted_at IS NULL AND superseded_by IS NULL "
        "AND metadata->>'relation_context' = 'true' "
        "AND metadata->>'from' = %s "
        "AND metadata->>'to' = %s "
        "AND metadata->>'relation_type' = %s",
        (json.dumps(weight), from_entity, to_entity, relation_type),
    )


def _relate_edges(cur, edges: list[dict], project: str | None) -> list[dict]:
    """Create/update directed AGE edges and the SQL mirror rows.

    Caller must set search_path to ag_catalog, public and handle the
    transaction. Uses MERGE, so it is idempotent and keeps both directions
    of the same relation type independent.
    """
    label = _scope_label(project)
    prepared: list[dict] = []
    for edge in edges:
        frm = _norm_entity(edge.get("from_entity"))
        to = _norm_entity(edge.get("to_entity"))
        rtype = edge.get("relation_type")
        if not frm or not to:
            raise ValueError("each edge needs 'from_entity' and 'to_entity'")
        _safe_ident(rtype or "", "relation_type")
        prepared.append({
            "from_entity": frm,
            "to_entity": to,
            "relation_type": rtype,
            "weight": _validate_weight(edge.get("weight", 1.0)),
            "context": edge.get("context"),
        })

    entities: list[str] = []
    seen: set[str] = set()
    for e in prepared:
        for ent in (e["from_entity"], e["to_entity"]):
            if ent not in seen:
                seen.add(ent)
                entities.append(ent)

    scope = _scope_key(project)
    for ent in entities:
        _ensure_graph_entity(cur, ent, project)

    results: list[dict] = []
    for e in prepared:
        from_json = json.dumps(e["from_entity"])
        to_json = json.dumps(e["to_entity"])
        w_json = json.dumps(e["weight"])
        rtype = e["relation_type"]

        cur.execute(
            f"SELECT * FROM cypher('memory_graph', $$ "
            f"MATCH (a:{label} {{name: {from_json}}})-[r:{rtype}]->(b:{label} {{name: {to_json}}}) "
            f"RETURN r.weight $$) AS (w agtype)"
        )
        existed = cur.fetchone() is not None

        cur.execute(
            f"SELECT * FROM cypher('memory_graph', $$ "
            f"MATCH (a:{label} {{name: {from_json}}}), (b:{label} {{name: {to_json}}}) "
            f"MERGE (a)-[r:{rtype}]->(b) SET r.weight = {w_json} "
            f"RETURN r $$) AS (r agtype)"
        )
        cur.execute(
            "INSERT INTO relations (source_id, target_id, relation_type, weight) "
            "SELECT e1.id, e2.id, %s, %s "
            "FROM entities e1, entities e2 "
            "WHERE e1.name = %s AND e2.name = %s "
            "AND e1.project = %s AND e2.project = %s "
            "ON CONFLICT (source_id, target_id, relation_type) DO UPDATE SET weight = EXCLUDED.weight",
            (rtype, e["weight"], e["from_entity"], e["to_entity"], scope, scope),
        )
        # The context sentence of an edge carries the edge weight in its
        # metadata. A later weight change would leave the old value behind
        # and contradict the graph, so the two are kept in sync.
        _sync_context_weight(cur, project, e["from_entity"], e["to_entity"],
                             rtype, e["weight"])
        results.append({
            "from_entity": e["from_entity"],
            "to_entity": e["to_entity"],
            "relation_type": rtype,
            "weight": e["weight"],
            "status": "updated" if existed else "created",
            "context_memory": None,
            "self_loop": e["from_entity"] == e["to_entity"],
        })

    context_edges = [(e, r) for e, r in zip(prepared, results) if e["context"]]
    if context_edges:
        embeddings = embed([e["context"] for e, _ in context_edges])
        ctx_table = _mem_table(project)
        for (e, r), emb in zip(context_edges, embeddings):
            ctx_entity = f"{e['from_entity']}_{e['relation_type']}_{e['to_entity']}"
            cur.execute(
                f"SELECT id FROM {ctx_table} WHERE entity = %s "
                f"AND superseded_by IS NULL AND deleted_at IS NULL LIMIT 1",
                (ctx_entity,),
            )
            row = cur.fetchone()
            mem_id = _store_memory(
                cur, e["context"], emb, "context", ctx_entity, 0.8,
                {"relation_context": True, "relation_type": e["relation_type"],
                 "from": e["from_entity"],
                 "to": e["to_entity"], "weight": e["weight"]},
                project, None,
            )
            if row:
                _supersede(cur, project, mem_id, row[0])
            r["context_memory"] = mem_id

    return results


def relate(
    from_entity: Annotated[
        str,
        Field(description="Source entity; created automatically if unknown.")],
    to_entity: Annotated[
        str,
        Field(description="Target entity; created automatically if unknown.")],
    relation_type: Annotated[
        str,
        Field(description="Relation label, e.g. 'depends_on'. Only letters, "
                          "digits and '_'.")],
    context: Annotated[
        str | None,
        Field(description="Sentence describing the relation; it is stored as a "
                          "memory of the relation.")] = None,
    project: Annotated[
        str | None,
        Field(description="Project scope of the graph. Without it the global "
                          "graph is used.")] = None,
    weight: Annotated[
        float,
        Field(description="Strength of the edge in 0.0-1.0 (default 1.0). It "
                          "scales the retrieval boost of graph_boost.")] = 1.0,
) -> dict[str, Any]:
    """Create a directed relation between two entities in the AGE graph.

    `weight` is the strength of the edge in 0.0-1.0; it scales the retrieval
    boost that `graph_boost` gives to the connected memories.
    """
    logger.info("relate %s ->[%s]-> %s weight=%s", from_entity, relation_type, to_entity, weight)
    try:
        project = _norm_project(project)
    except ValueError as e:
        return fail(STATUS_ERROR, str(e))
    if not relation_type:
        return fail(STATUS_ERROR, "relation_type is required")

    conn = get_conn()
    try:
        cur = conn.cursor()
        if project:
            # Every project that gets a graph node also gets its memory table
            # and its registry entry. Without that the scope existed only in
            # the graph: invisible in `list_projects` and not removable with
            # `drop_project`, which left its nodes behind as orphans.
            _ensure_project_table(cur, project)
        cur.execute("SET search_path TO ag_catalog, public")
        results = _relate_edges(cur, [{
            "from_entity": from_entity,
            "to_entity": to_entity,
            "relation_type": relation_type,
            "weight": weight,
            "context": context,
        }], project)
        conn.commit()
    except Exception as e:
        conn.rollback()
        return fail(STATUS_ERROR, str(e))
    finally:
        conn.close()

    edge = results[0]
    warnings: list[str] = []
    if edge.get("self_loop"):
        warnings.append(
            f"self-loop: {edge['from_entity']!r} points to itself via "
            f"{edge['relation_type']!r}; graph traversal cannot leave that "
            "node through this edge")
    return ok(edge["status"], **{
        "from_entity": edge["from_entity"],
        "to_entity": edge["to_entity"],
        "relation_type": edge["relation_type"],
        "weight": edge["weight"],
        "context_memory": edge["context_memory"],
        "warnings": warnings,
    })


def relate_batch(
    edges: Annotated[
        list[RelationEdge],
        Field(description="Edges to create/update. Each needs from_entity, "
                          "to_entity and relation_type; weight, context and "
                          "project are optional.")],
    project: Annotated[
        str | None,
        Field(description="Project scope for all edges (a per-edge project "
                          "wins).")] = None,
) -> dict[str, Any]:
    """Create or update multiple directed relations in one transaction.

    Each edge needs: from_entity, to_entity, relation_type.
    Optional: weight (default 1.0), context (stored as a memory), project
    (overrides the tool-level project for that edge).
    Entities are auto-created; existing edges are updated (MERGE), so the
    call is idempotent and both directions of a relation can coexist.
    """
    logger.info("relate_batch count=%s project=%s", len(edges or []), project)
    try:
        project = _norm_project(project)
    except ValueError as e:
        return fail(STATUS_ERROR, str(e))
    if not edges:
        return fail(STATUS_ERROR, "edges must not be empty: pass at least one "
                                  "edge or omit the call")
    conn = get_conn()
    try:
        cur = conn.cursor()
        # Every project that gets a graph node also gets its memory table and
        # its registry entry, so the scope is visible in `list_projects` and
        # removable with `drop_project`.
        for p in {_norm_project(e.get("project", project)) for e in edges}:
            if p:
                _ensure_project_table(cur, p)
        cur.execute("SET search_path TO ag_catalog, public")
        # Edges may target different scopes; keep the caller's order.
        indexed: dict[Any, list[tuple[int, dict]]] = {}
        for i, edge in enumerate(edges):
            indexed.setdefault(_norm_project(edge.get("project", project)),
                               []).append((i, edge))
        results: list[Any] = [None] * len(edges)
        for scope, pairs in indexed.items():
            batch = _relate_edges(cur, [edge for _, edge in pairs], scope)
            for (pos, _), result in zip(pairs, batch):
                results[pos] = result
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.exception("relate_batch failed")
        return fail(STATUS_ERROR, str(e))
    finally:
        conn.close()
    warnings: list[str] = []
    self_loops = [r for r in results if r.get("self_loop")]
    if self_loops:
        warnings.append(
            f"{len(self_loops)} self-loop edge(s) stored (an entity points to "
            "itself); they are valid but cannot expand a graph traversal")
    return ok(STATUS_OK, count=len(results), results=results,
              warnings=warnings)


def unrelate(
    from_entity: Annotated[
        str,
        Field(description="Source entity of the edge to delete.")],
    to_entity: Annotated[
        str,
        Field(description="Target entity of the edge to delete.")],
    relation_type: Annotated[
        str,
        Field(description="Relation label of the edge to delete.")],
    project: Annotated[
        str | None,
        Field(description="Project scope of the graph.")] = None,
    keep_context: Annotated[
        bool,
        Field(description="Keep the relation-context memory that `relate` "
                          "stored for this edge. By default it is "
                          "soft-forgotten together with the edge, so a "
                          "deleted relation does not keep describing a "
                          "connection that no longer exists.")] = False,
) -> dict[str, Any]:
    """Delete a single directed relation between two entities.

    The context sentence of the edge is soft-forgotten too unless
    `keep_context=True`: it describes an edge that no longer exists.
    """
    logger.info("unrelate %s -[%s]-> %s project=%s",
                from_entity, relation_type, to_entity, project)
    try:
        project = _norm_project(project)
        from_entity = _norm_entity(from_entity)
        to_entity = _norm_entity(to_entity)
    except ValueError as e:
        return fail(STATUS_ERROR, str(e))
    if not from_entity or not to_entity:
        return fail(STATUS_ERROR, "from_entity and to_entity are required")
    if not relation_type:
        return fail(STATUS_ERROR, "relation_type is required")
    try:
        _safe_ident(relation_type, "relation_type")
    except ValueError as e:
        return fail(STATUS_ERROR, str(e))

    label = _scope_label(project)
    scope = _scope_key(project)
    from_json = json.dumps(from_entity)
    to_json = json.dumps(to_entity)
    conn = get_conn()
    context_removed = 0
    try:
        cur = conn.cursor()
        cur.execute("SET search_path TO ag_catalog, public")
        cur.execute(
            f"SELECT * FROM cypher('memory_graph', $$ "
            f"MATCH (a:{label} {{name: {from_json}}})-[r:{relation_type}]->"
            f"(b:{label} {{name: {to_json}}}) "
            f"DELETE r RETURN 1 $$) AS (x agtype)"
        )
        deleted = bool(cur.fetchall())
        if deleted:
            cur.execute(
                "DELETE FROM relations "
                "WHERE relation_type = %s "
                "AND source_id = (SELECT id FROM entities WHERE name = %s AND project = %s) "
                "AND target_id = (SELECT id FROM entities WHERE name = %s AND project = %s)",
                (relation_type, from_entity, scope, to_entity, scope),
            )
            if not keep_context:
                context_removed = _forget_relation_context(
                    cur, project, from_entity, to_entity, relation_type,
                )
        conn.commit()
    except Exception as e:
        conn.rollback()
        return fail(STATUS_ERROR, str(e))
    finally:
        conn.close()

    if not deleted:
        return fail(STATUS_NOT_FOUND,
                    f"no {relation_type} edge from {from_entity} to {to_entity}")
    return ok(STATUS_DELETED, from_entity=from_entity, to_entity=to_entity,
              relation_type=relation_type,
              context_memories_removed=context_removed,
              context_kept=keep_context)


def find_connections(
    entity_name: Annotated[
        str | None,
        Field(description="Entity to traverse from. Alias: `entity`.")] = None,
    direction: Annotated[
        Literal["outgoing", "incoming", "both"],
        Field(description="Direction to traverse (default: both). Each result "
                          "reports its own direction.")] = "both",
    relation_type: Annotated[
        str | None,
        Field(description="Only follow edges of this relation type.")] = None,
    limit: Annotated[
        int,
        Field(description="Maximum number of results (default 20).")] = 20,
    offset: Annotated[
        int,
        Field(description="Skip this many connections (default 0). Use it "
                          "with `limit` to page; the answer reports `total` "
                          "and `has_more`.")] = 0,
    project: Annotated[
        str | None,
        Field(description="Project scope of the graph. Without it the global "
                          "graph is traversed.")] = None,
    entity: Annotated[
        str | None,
        Field(description="Entity to traverse from. Alias: `entity_name`.")] = None,
) -> dict[str, Any]:
    """Traverse the graph from an entity to find connected entities.

    The start entity can be passed as `entity` or `entity_name`.
    `direction` is "outgoing", "incoming" or "both"; each result reports its
    own direction. `project` selects that project's graph; without it the
    global graph is traversed. `offset` pages through the connections; the
    answer reports `total` and `has_more`, so a truncated list is never
    mistaken for a complete neighbourhood.
    """
    if entity and entity_name and entity.strip() != entity_name.strip():
        return fail(
            STATUS_ERROR,
            f"entity and entity_name were both given but differ "
            f"({entity!r} vs {entity_name!r}); pass only one of the aliases")
    entity_name = entity or entity_name
    logger.info("find_connections entity=%s direction=%s relation_type=%s limit=%s project=%s",
                entity_name, direction, relation_type, limit, project)
    try:
        project = _norm_project(project)
        entity_name = _norm_entity(entity_name)
    except ValueError as e:
        return fail(STATUS_ERROR, str(e))
    if not entity_name:
        return fail(STATUS_ERROR, "entity is required")
    if limit < 1:
        return fail(STATUS_ERROR, "limit must be >= 1")
    if offset < 0:
        return fail(STATUS_ERROR, "offset must be >= 0")
    if direction not in ("outgoing", "incoming", "both"):
        return fail(STATUS_ERROR,
                    "direction must be 'outgoing', 'incoming' or 'both'")
    if relation_type:
        try:
            _safe_ident(relation_type, "relation_type")
        except ValueError as e:
            return fail(STATUS_ERROR, str(e))
    label = _scope_label(project)
    rtype_cypher = f":{relation_type}" if relation_type else ""
    name_json = json.dumps(entity_name)
    if direction == "outgoing":
        pattern = f"(a:{label})-[r{rtype_cypher}]->(b)"
    elif direction == "incoming":
        pattern = f"(a:{label})<-[r{rtype_cypher}]-(b)"
    else:
        pattern = f"(a:{label})-[r{rtype_cypher}]-(b)"
    base_query = f"MATCH {pattern} WHERE a.name = {name_json}"
    count_query = f"{base_query} RETURN count(r)"
    data_query = (
        f"{base_query} "
        f"RETURN b.name AS connected_entity, type(r) AS relation_type, "
        f"coalesce(r.weight, 1.0) AS w, startNode(r).name AS src, "
        f"endNode(r).name AS tgt "
        f"ORDER BY coalesce(r.weight, 1.0) DESC, b.name, type(r) "
        f"SKIP {int(offset)} LIMIT {int(limit)}"
    )

    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT 1 FROM entities WHERE name = %s AND project = %s LIMIT 1",
            (entity_name, _scope_key(project)),
        )
        if cur.fetchone() is None:
            conn.rollback()
            return fail(STATUS_NOT_FOUND,
                        _entity_not_found_hint(cur, entity_name, project),
                        entity=entity_name, scope=_scope_key(project))
        cur.execute("SET search_path TO ag_catalog, public")
        cur.execute(
            f"SELECT * FROM cypher('memory_graph', $$ {count_query} $$) AS "
            f"(total agtype)"
        )
        row = cur.fetchone()
        total = int(str(row[0])) if row else 0
        cur.execute(
            f"SELECT * FROM cypher('memory_graph', $$ {data_query} $$) AS "
            f"(connected_entity agtype, relation_type agtype, weight agtype, "
            f"src agtype, tgt agtype)",
        )
        results = []
        for row in cur.fetchall():
            src = _ag_str(row[3])
            tgt = _ag_str(row[4])
            if direction == "outgoing":
                edge_dir = "outgoing"
            elif direction == "incoming":
                edge_dir = "incoming"
            else:
                edge_dir = "outgoing" if src == entity_name else "incoming"
            results.append({
                "entity": _ag_str(row[0]),
                "relation": _ag_str(row[1]),
                "weight": float(str(row[2])),
                "direction": edge_dir,
                "from_entity": src,
                "to_entity": tgt,
            })
        conn.commit()
    except Exception as e:
        conn.rollback()
        return fail(STATUS_ERROR, str(e))
    finally:
        conn.close()
    has_more = total > offset + limit
    warnings: list[str] = []
    if offset > 0 and total > 0 and not results:
        warnings.append(
            f"no results on this page: offset={offset} is beyond the "
            f"{total} connection(s); lower the offset")
    return ok(STATUS_OK, count=len(results), total=total, offset=offset,
              has_more=has_more,
              next_offset=(offset + limit) if has_more else None,
              warnings=warnings, results=results)


def find_paths(from_entity: Annotated[
        str,
        Field(description="Start entity of the path.")],
    to_entity: Annotated[
        str,
        Field(description="End entity of the path.")],
    max_hops: Annotated[
        int,
        Field(description="Maximum number of hops to search (default 5, max "
                          "10).")] = 5,
    project: Annotated[
        str | None,
        Field(description="Project scope of the graph. Without it the global "
                          "graph is used.")] = None,
) -> dict[str, Any]:
    """Find paths between two entities in the graph.

    The canonical collection is ``results``.
    `project` selects that project's graph; without it the global graph is used.
    When no path is found the response says so in `warnings`: an empty
    `results` list then means "no connection", not "nothing looked up".
    """
    logger.info("find_paths from=%s to=%s max_hops=%s project=%s",
                from_entity, to_entity, max_hops, project)
    try:
        project = _norm_project(project)
        from_entity = _norm_entity(from_entity)
        to_entity = _norm_entity(to_entity)
    except ValueError as e:
        return fail(STATUS_ERROR, str(e))
    if not from_entity or not to_entity:
        return fail(STATUS_ERROR, "from_entity and to_entity are required")
    try:
        max_hops = int(max_hops)
    except (TypeError, ValueError):
        return fail(STATUS_ERROR, "max_hops must be an integer")
    if max_hops < 1:
        return fail(STATUS_ERROR, "max_hops must be >= 1")
    warnings: list[str] = []
    if max_hops > 10:
        warnings.append(
            f"max_hops={max_hops} was clamped to 10: longer paths are not "
            "searched")
        max_hops = 10
    label = _scope_label(project)
    conn = get_conn()
    try:
        cur = conn.cursor()
        scope = _scope_key(project)
        cur.execute(
            "SELECT name FROM entities WHERE project = %s AND name IN (%s, %s)",
            (scope, from_entity, to_entity),
        )
        known_entities = {row[0] for row in cur.fetchall()}
        missing = [name for name in (from_entity, to_entity)
                   if name not in known_entities]
        if missing:
            conn.rollback()
            return fail(STATUS_NOT_FOUND,
                        _entity_not_found_hint(cur, missing[0], project),
                        entity=missing[0], scope=scope)
        if from_entity == to_entity:
            cur.execute(
                "SELECT id FROM entities WHERE project = %s AND name = %s",
                (scope, from_entity))
            row = cur.fetchone()
            conn.rollback()
            return ok(STATUS_OK, count=1, warnings=warnings, results=[{
                "nodes": [{"id": row[0] if row else None, "name": from_entity,
                           "label": label}],
                "relations": [], "hops": 0}])
        cur.execute("SET search_path TO ag_catalog, public")
        fe_json = json.dumps(from_entity)
        te_json = json.dumps(to_entity)
        cur.execute(
            f"""SELECT * FROM cypher('memory_graph', $$
                MATCH p = (a:{label})-[*1..{max_hops}]-(b:{label})
                WHERE a.name = {fe_json} AND b.name = {te_json}
                RETURN nodes(p) AS ns, relationships(p) AS rs
                LIMIT 20
            $$) AS (ns agtype, rs agtype)""")
        rows = cur.fetchall()
        conn.commit()
    except Exception as e:
        conn.rollback()
        return fail(STATUS_ERROR, str(e))
    finally:
        conn.close()
    paths = []
    seen_paths: set[tuple] = set()
    conn = get_conn()
    entity_ids: dict[str, int] = {}
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT name, id FROM entities WHERE project = %s",
            (_scope_key(project),),
        )
        entity_ids = {name: entity_id for name, entity_id in cur.fetchall()}
    finally:
        conn.close()
    for r in rows:
        nodes_str = str(r[0]).replace("::vertex", "").replace("::edge", "")
        rels_str = str(r[1]).replace("::vertex", "").replace("::edge", "")
        try:
            parsed = json.loads(nodes_str)
            parsed_rels = json.loads(rels_str)
        except json.JSONDecodeError:
            continue
        names = [n.get("properties", {}).get("name", "?") for n in parsed]
        ids = [n.get("id") for n in parsed]
        if len(names) != len(set(names)):
            continue
        signature = tuple(ids)
        if signature in seen_paths:
            continue
        seen_paths.add(signature)
        nodes = [{"id": entity_ids.get(names[i]), "name": names[i],
                  "label": parsed[i].get("label")}
                 for i in range(len(names))]
        relations = []
        for rel in parsed_rels:
            props = rel.get("properties", {})
            start = rel.get("start_id", rel.get("start"))
            end = rel.get("end_id", rel.get("end"))
            relations.append({
                "type": rel.get("label"),
                "from": names[ids.index(start)] if start in ids else start,
                "to": names[ids.index(end)] if end in ids else end,
                "weight": props.get("weight", 1.0),
            })
        paths.append({"nodes": nodes, "relations": relations,
                      "hops": len(nodes) - 1})
    if paths:
        shortest_hops = min(path["hops"] for path in paths)
        paths = [path for path in paths if path["hops"] == shortest_hops]
    else:
        warnings.append(
            f"no path found between {from_entity!r} and {to_entity!r} "
            f"(at most {max_hops} hop(s) in the {scope} graph): the two "
            f"entities are not connected")
        return ok(STATUS_OK, count=0, results=[], warnings=warnings)
    return ok(STATUS_OK, count=len(paths), results=paths, warnings=warnings)
