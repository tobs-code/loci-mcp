import json
import logging
from typing import Any

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
    _norm_project,
    _safe_ident,
    _scope_key,
    _scope_label,
)
from mcp_server.storage import _store_memory, _supersede

logger = logging.getLogger(__name__)

def _ensure_graph_entity(cur, name: str, project: str | None,
                         ent_type: str | None = None) -> None:
    """Create the AGE node and SQL entity row for one scope (idempotent)."""
    label = _scope_label(project)
    scope = _scope_key(project)
    ent_json = json.dumps(name)
    scope_json = json.dumps(scope)
    cur.execute(
        f"SELECT * FROM cypher('memory_graph', $$ "
        f"MATCH (n:{label} {{name: {ent_json}}}) RETURN n $$) AS (n agtype)"
    )
    if not cur.fetchone():
        cur.execute(
            f"SELECT * FROM cypher('memory_graph', $$ "
            f"CREATE (n:{label} {{name: {ent_json}, project: {scope_json}}}) RETURN n $$) AS (n agtype)"
        )
    cur.execute(
        "INSERT INTO entities (name, type, project) VALUES (%s, %s, %s) "
        "ON CONFLICT (name, project) DO UPDATE SET type = EXCLUDED.type",
        (name, ent_type or label, scope),
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
        frm = edge.get("from_entity")
        to = edge.get("to_entity")
        rtype = edge.get("relation_type")
        if not frm or not to:
            raise ValueError("each edge needs 'from_entity' and 'to_entity'")
        _safe_ident(rtype or "", "relation_type")
        prepared.append({
            "from_entity": frm,
            "to_entity": to,
            "relation_type": rtype,
            "weight": float(edge.get("weight", 1.0)),
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
        results.append({
            "from_entity": e["from_entity"],
            "to_entity": e["to_entity"],
            "relation_type": rtype,
            "weight": e["weight"],
            "status": "updated" if existed else "created",
            "context_memory": None,
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
                cur, e["context"], emb, "fact", ctx_entity, 0.8,
                {"relation_type": e["relation_type"], "from": e["from_entity"],
                 "to": e["to_entity"], "weight": e["weight"]},
                project, None,
            )
            if row:
                _supersede(cur, project, mem_id, row[0])
            r["context_memory"] = mem_id

    return results


def relate(
    from_entity: str,
    to_entity: str,
    relation_type: str,
    context: str | None = None,
    project: str | None = None,
    weight: float = 1.0,
) -> dict[str, Any]:
    """Create a directed relation between two entities in the AGE graph."""
    logger.info("relate %s ->[%s]-> %s weight=%s", from_entity, relation_type, to_entity, weight)
    project = _norm_project(project)
    if not relation_type:
        return fail(STATUS_ERROR, "relation_type is required")

    conn = get_conn()
    try:
        cur = conn.cursor()
        if context and project:
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
    return ok(edge["status"], **{
        "from_entity": edge["from_entity"],
        "to_entity": edge["to_entity"],
        "relation_type": edge["relation_type"],
        "weight": edge["weight"],
        "context_memory": edge["context_memory"],
    })


def relate_batch(
    edges: list[dict],
    project: str | None = None,
) -> dict[str, Any]:
    """Create or update multiple directed relations in one transaction.

    Each edge needs: from_entity, to_entity, relation_type.
    Optional: weight (default 1.0), context (stored as a memory), project
    (overrides the tool-level project for that edge).
    Entities are auto-created; existing edges are updated (MERGE), so the
    call is idempotent and both directions of a relation can coexist.
    """
    logger.info("relate_batch count=%s project=%s", len(edges or []), project)
    project = _norm_project(project)
    if not edges:
        return ok(STATUS_OK, count=0, results=[])
    conn = get_conn()
    try:
        cur = conn.cursor()
        if any(e.get("context") for e in edges):
            for p in {_norm_project(e.get("project", project))
                      for e in edges if e.get("context")}:
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
    return ok(STATUS_OK, count=len(results), results=results)


def unrelate(
    from_entity: str,
    to_entity: str,
    relation_type: str,
    project: str | None = None,
) -> dict[str, Any]:
    """Delete a single directed relation between two entities."""
    logger.info("unrelate %s -[%s]-> %s project=%s",
                from_entity, relation_type, to_entity, project)
    project = _norm_project(project)
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
              relation_type=relation_type)


def find_connections(
    entity_name: str | None = None,
    direction: str = "both",
    relation_type: str | None = None,
    limit: int = 20,
    project: str | None = None,
    entity: str | None = None,
) -> dict[str, Any]:
    """Traverse the graph from an entity to find connected entities.

    The start entity can be passed as `entity` or `entity_name`.
    `direction` is "outgoing", "incoming" or "both"; each result reports its
    own direction. `project` selects that project's graph; without it the
    global graph is traversed.
    """
    entity_name = entity or entity_name
    logger.info("find_connections entity=%s direction=%s relation_type=%s limit=%s project=%s",
                entity_name, direction, relation_type, limit, project)
    project = _norm_project(project)
    if not entity_name:
        return fail(STATUS_ERROR, "entity is required")
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
    query = f"""
        MATCH (a:{label})-[r{rtype_cypher}]-(b)
        WHERE a.name = {name_json}
        RETURN b.name AS connected_entity, type(r) AS relation_type,
               coalesce(r.weight, 1.0) AS w, startNode(r).name AS src,
               endNode(r).name AS tgt
        LIMIT {int(limit) * 4}
    """

    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("SET search_path TO ag_catalog, public")
        cur.execute(
            f"SELECT * FROM cypher('memory_graph', $$ {query} $$) AS "
            f"(connected_entity agtype, relation_type agtype, weight agtype, "
            f"src agtype, tgt agtype)",
        )
        results = []
        for row in cur.fetchall():
            src = _ag_str(row[3])
            tgt = _ag_str(row[4])
            edge_dir = "outgoing" if src == entity_name else "incoming"
            if direction != "both" and edge_dir != direction:
                continue
            results.append({
                "entity": _ag_str(row[0]),
                "relation": _ag_str(row[1]),
                "weight": float(str(row[2])),
                "direction": edge_dir,
                "from_entity": src,
                "to_entity": tgt,
            })
        # Strongest edges first, then alphabetical for stable output.
        results.sort(key=lambda r: (-r["weight"], r["entity"], r["relation"]))
        results = results[:limit]
        conn.commit()
    except Exception as e:
        conn.rollback()
        return fail(STATUS_ERROR, str(e))
    finally:
        conn.close()
    return ok(STATUS_OK, count=len(results), results=results)


def find_paths(from_entity: str, to_entity: str, max_hops: int = 5,
               project: str | None = None) -> dict[str, Any]:
    """Find paths between two entities in the graph.

    `project` selects that project's graph; without it the global graph is used.
    """
    logger.info("find_paths from=%s to=%s max_hops=%s project=%s",
                from_entity, to_entity, max_hops, project)
    project = _norm_project(project)
    max_hops = max(1, min(int(max_hops), 10))
    label = _scope_label(project)
    conn = get_conn()
    try:
        cur = conn.cursor()
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
        nodes = [{"id": ids[i], "name": names[i],
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
    return ok(STATUS_OK, count=len(paths), results=paths, paths=paths)
