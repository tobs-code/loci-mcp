"""Graph storage: AGE Cypher plus the SQL entity/relation mirror.

Every Cypher statement and every mirror-row write lives here, so a caller
works with entities and edges instead of AGE syntax. The two stores are
updated together: a node without its mirror row or an edge without its
relation row would make `list_entities` and `find_connections` disagree.
"""

from __future__ import annotations

import json

from mcp_server.scopes import (
    _ag_str,
    _mem_table,
    _scope_key,
    _scope_label,
    _table_exists,
)


def _cypher(cur, body: str, columns: str):
    """Run one Cypher statement against the shared AGE graph."""
    cur.execute("SET search_path TO ag_catalog, public")
    cur.execute(
        f"SELECT * FROM cypher('memory_graph', $$ {body} $$) AS ({columns})"
    )
    return cur


def ensure_entity(cur, name: str, project: str | None,
                  ent_type: str | None = None) -> bool:
    """Create the AGE node and SQL entity row for one scope (idempotent)."""
    label = _scope_label(project)
    scope = _scope_key(project)
    name_json = json.dumps(name)
    scope_json = json.dumps(scope)
    _cypher(cur, f"MATCH (n:{label} {{name: {name_json}}}) RETURN n",
            "n agtype")
    created = cur.fetchone() is None
    if created:
        _cypher(
            cur,
            f"CREATE (n:{label} {{name: {name_json}, project: {scope_json}}}) "
            f"RETURN n",
            "n agtype",
        )
    cur.execute(
        "INSERT INTO entities (name, type, project) VALUES (%s, %s, %s) "
        "ON CONFLICT (name, project) DO UPDATE SET type = EXCLUDED.type",
        (name, ent_type or label, scope),
    )
    return created


def edge_weight(cur, project: str | None, from_entity: str, to_entity: str,
                relation_type: str) -> float | None:
    """Stored weight of an edge, or None when it does not exist."""
    label = _scope_label(project)
    frm = json.dumps(from_entity)
    to = json.dumps(to_entity)
    _cypher(
        cur,
        f"MATCH (a:{label} {{name: {frm}}})-[r:{relation_type}]->"
        f"(b:{label} {{name: {to}}}) RETURN r.weight",
        "w agtype",
    )
    row = cur.fetchone()
    if row is None:
        return None
    try:
        return float(_ag_str(row[0]))
    except (TypeError, ValueError):
        return None


def upsert_edge(cur, project: str | None, from_entity: str, to_entity: str,
                relation_type: str, weight: float) -> bool:
    """MERGE one directed edge and its mirror row; True when it existed."""
    label = _scope_label(project)
    scope = _scope_key(project)
    existed = edge_weight(cur, project, from_entity, to_entity, relation_type) \
        is not None
    frm = json.dumps(from_entity)
    to = json.dumps(to_entity)
    _cypher(
        cur,
        f"MATCH (a:{label} {{name: {frm}}}), (b:{label} {{name: {to}}}) "
        f"MERGE (a)-[r:{relation_type}]->(b) SET r.weight = {json.dumps(weight)} "
        f"RETURN r",
        "r agtype",
    )
    cur.execute(
        "INSERT INTO relations (source_id, target_id, relation_type, weight) "
        "SELECT e1.id, e2.id, %s, %s "
        "FROM entities e1, entities e2 "
        "WHERE e1.name = %s AND e2.name = %s "
        "AND e1.project = %s AND e2.project = %s "
        "ON CONFLICT (source_id, target_id, relation_type) "
        "DO UPDATE SET weight = EXCLUDED.weight",
        (relation_type, weight, from_entity, to_entity, scope, scope),
    )
    return existed


def delete_edge(cur, project: str | None, from_entity: str, to_entity: str,
                relation_type: str) -> bool:
    """Delete one directed edge and its mirror row; False when absent."""
    label = _scope_label(project)
    scope = _scope_key(project)
    frm = json.dumps(from_entity)
    to = json.dumps(to_entity)
    _cypher(
        cur,
        f"MATCH (a:{label} {{name: {frm}}})-[r:{relation_type}]->"
        f"(b:{label} {{name: {to}}}) DELETE r RETURN 1",
        "x agtype",
    )
    deleted = bool(cur.fetchall())
    if deleted:
        cur.execute(
            "DELETE FROM relations "
            "WHERE relation_type = %s "
            "AND source_id = (SELECT id FROM entities "
            "WHERE name = %s AND project = %s) "
            "AND target_id = (SELECT id FROM entities "
            "WHERE name = %s AND project = %s)",
            (relation_type, from_entity, scope, to_entity, scope),
        )
    return deleted


def entity_row(cur, project: str | None, name: str) -> tuple | None:
    """Mirror row of one entity in this scope."""
    cur.execute(
        "SELECT id, name, type, project FROM entities "
        "WHERE name = %s AND project = %s",
        (name, _scope_key(project)),
    )
    return cur.fetchone()


def entity_exists(cur, project: str | None, name: str) -> bool:
    cur.execute(
        "SELECT 1 FROM entities WHERE name = %s AND project = %s LIMIT 1",
        (name, _scope_key(project)),
    )
    return cur.fetchone() is not None


def node_exists(cur, project: str | None, name: str) -> bool:
    label = _scope_label(project)
    _cypher(
        cur,
        f"MATCH (n:{label} {{name: {json.dumps(name)}}}) RETURN n",
        "n agtype",
    )
    return cur.fetchone() is not None


def find_entity(cur, project: str | None, name: str) -> tuple | None:
    """Entity row, falling back to a graph-only node (AGE, no mirror row)."""
    row = entity_row(cur, project, name)
    if row:
        return row
    if node_exists(cur, project, name):
        return (None, name, _scope_label(project), _scope_key(project))
    return None


def connections(cur, project: str | None, name: str) -> list[dict]:
    """All edges touching an entity, with direction."""
    label = _scope_label(project)
    _cypher(
        cur,
        f"MATCH (a:{label} {{name: {json.dumps(name)}}})-[r]-(b) "
        f"RETURN b.name, type(r), coalesce(r.weight, 1.0), "
        f"startNode(r).name",
        "neighbor agtype, rel agtype, w agtype, src agtype",
    )
    return [
        {"entity": _ag_str(row[0]), "relation": _ag_str(row[1]),
         "weight": float(str(row[2])),
         "direction": "outgoing" if _ag_str(row[3]) == name else "incoming"}
        for row in cur.fetchall()
    ]


def traverse(cur, project: str | None, name: str, direction: str,
             relation_type: str | None, limit: int,
             offset: int) -> tuple[int, list[dict]]:
    """Paged neighbourhood of one entity."""
    label = _scope_label(project)
    rtype = f":{relation_type}" if relation_type else ""
    name_json = json.dumps(name)
    if direction == "outgoing":
        pattern = f"(a:{label})-[r{rtype}]->(b)"
    elif direction == "incoming":
        pattern = f"(a:{label})<-[r{rtype}]-(b)"
    else:
        pattern = f"(a:{label})-[r{rtype}]-(b)"
    base = f"MATCH {pattern} WHERE a.name = {name_json}"
    _cypher(cur, f"{base} RETURN count(r)", "total agtype")
    row = cur.fetchone()
    total = int(str(row[0])) if row else 0
    _cypher(
        cur,
        f"{base} RETURN b.name AS connected_entity, "
        f"type(r) AS relation_type, coalesce(r.weight, 1.0) AS w, "
        f"startNode(r).name AS src, endNode(r).name AS tgt "
        f"ORDER BY coalesce(r.weight, 1.0) DESC, b.name, type(r) "
        f"SKIP {int(offset)} LIMIT {int(limit)}",
        "connected_entity agtype, relation_type agtype, weight agtype, "
        "src agtype, tgt agtype",
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
            edge_dir = "outgoing" if src == name else "incoming"
        results.append({
            "entity": _ag_str(row[0]),
            "relation": _ag_str(row[1]),
            "weight": float(str(row[2])),
            "direction": edge_dir,
            "from_entity": src,
            "to_entity": tgt,
        })
    return total, results


def paths(cur, project: str | None, from_entity: str, to_entity: str,
          max_hops: int) -> list:
    """Raw AGE paths between two entities (both directions)."""
    label = _scope_label(project)
    _cypher(
        cur,
        f"MATCH p = (a:{label})-[*1..{int(max_hops)}]-(b:{label}) "
        f"WHERE a.name = {json.dumps(from_entity)} "
        f"AND b.name = {json.dumps(to_entity)} "
        f"RETURN nodes(p) AS ns, relationships(p) AS rs LIMIT 20",
        "ns agtype, rs agtype",
    )
    return list(cur.fetchall())


def entity_ids(cur, project: str | None) -> dict[str, int]:
    cur.execute("SELECT name, id FROM entities WHERE project = %s",
                (_scope_key(project),))
    return {name: entity_id for name, entity_id in cur.fetchall()}


def edge_count(cur, project: str | None, name: str) -> int:
    label = _scope_label(project)
    _cypher(
        cur,
        f"MATCH (n:{label} {{name: {json.dumps(name)}}})-[r]-() "
        f"RETURN count(r)",
        "c agtype",
    )
    row = cur.fetchone()
    return int(str(row[0])) if row else 0


def delete_node(cur, project: str | None, name: str) -> None:
    """Remove one node and its mirror row."""
    label = _scope_label(project)
    scope = _scope_key(project)
    _cypher(
        cur,
        f"MATCH (n:{label} {{name: {json.dumps(name)}}}) DETACH DELETE n",
        "x agtype",
    )
    cur.execute("DELETE FROM entities WHERE name = %s AND project = %s",
                (name, scope))


def drop_orphan_nodes(cur, project: str | None,
                      names: list[str]) -> list[str]:
    """Remove nodes an unrelate left without an edge or a memory.

    A node is only removed when it has no remaining edge in this scope and no
    memory row at all, so a node that still backs a fact is never dropped.
    """
    scope = _scope_key(project)
    table = _mem_table(project)
    has_table = _table_exists(cur, table)
    removed: list[str] = []
    for name in dict.fromkeys(names):
        if edge_count(cur, project, name) > 0:
            continue
        if has_table:
            cur.execute(
                f"SELECT 1 FROM {table} WHERE entity = %s LIMIT 1", (name,))
            if cur.fetchone():
                continue
        cur.execute(
            "SELECT id FROM entities WHERE name = %s AND project = %s",
            (name, scope))
        mirror = cur.fetchone()
        if mirror:
            cur.execute(
                "SELECT 1 FROM relations WHERE source_id = ANY(%s) "
                "OR target_id = ANY(%s) LIMIT 1",
                ([mirror[0]], [mirror[0]]))
            if cur.fetchone():
                continue
        delete_node(cur, project, name)
        removed.append(name)
    return removed


def edge_context(cur, project: str | None, from_entity: str, to_entity: str,
                 relation_type: str) -> str | None:
    """Latest active context sentence of one edge."""
    table = _mem_table(project)
    if not _table_exists(cur, table):
        return None
    cur.execute(
        f"SELECT content FROM {table} "
        "WHERE entity = %s AND superseded_by IS NULL "
        "AND deleted_at IS NULL "
        "AND metadata->>'relation_context' = 'true' "
        "AND metadata->>'relation_type' = %s "
        "ORDER BY updated_at DESC LIMIT 1",
        (f"{from_entity}_{relation_type}_{to_entity}", relation_type),
    )
    row = cur.fetchone()
    return row[0] if row else None


def delete_scope_graph(cur, project: str | None) -> None:
    """Remove a scope's nodes and every mirror row that belongs to it."""
    label = _scope_label(project)
    scope = _scope_key(project)
    _cypher(cur, f"MATCH (n:{label}) DETACH DELETE n", "r agtype")
    cur.execute(
        "DELETE FROM relations WHERE source_id IN "
        "(SELECT id FROM entities WHERE project = %s) OR target_id IN "
        "(SELECT id FROM entities WHERE project = %s)",
        (scope, scope),
    )
    cur.execute("DELETE FROM entities WHERE project = %s", (scope,))


def delete_scope_nodes(cur, project: str | None) -> None:
    """Detach-delete a scope's nodes and their mirror rows, keep relations."""
    label = _scope_label(project)
    scope = _scope_key(project)
    _cypher(cur, f"MATCH (n:{label}) DETACH DELETE n", "r agtype")
    cur.execute("DELETE FROM entities WHERE project = %s", (scope,))


def delete_entity_everywhere(cur, name: str) -> None:
    """Detach-delete one entity name from every scope and its mirror rows."""
    _cypher(
        cur,
        f"MATCH (n {{name: {json.dumps(name)}}}) DETACH DELETE n",
        "r agtype",
    )
    cur.execute("DELETE FROM entities WHERE name = %s", (name,))


def boost_neighbors(cur, label: str, name: str) -> list[tuple]:
    """Up to ten edges around a node in one scope label (graph boost)."""
    _cypher(
        cur,
        f"MATCH (a:{label} {{name: {json.dumps(name)}}})-[r]-(b) "
        f"RETURN b.name, coalesce(r.weight, 1.0), type(r), a.name LIMIT 10",
        "name agtype, weight agtype, rtype agtype, source agtype",
    )
    return list(cur.fetchall())


def label_node_exists(cur, label: str, name: str) -> bool:
    _cypher(
        cur,
        f"MATCH (a:{label} {{name: {json.dumps(name)}}}) RETURN a LIMIT 1",
        "a agtype",
    )
    return cur.fetchone() is not None


def duplicate_edge_groups(cur) -> list[tuple]:
    """Directed (source, target, type) groups that AGE stored more than once."""
    _cypher(
        cur,
        "MATCH (a)-[r1]->(b) "
        "WITH a, b, type(r1) AS t, collect(r1) AS rels "
        "WHERE size(rels) > 1 "
        "RETURN a.name, b.name, t, size(rels)",
        "src agtype, tgt agtype, rel_type agtype, cnt agtype",
    )
    return list(cur.fetchall())


def delete_duplicate_edge(cur, from_entity: str, to_entity: str,
                          relation_type: str) -> None:
    """Delete one duplicate AGE edge without the mirror row."""
    _cypher(
        cur,
        f"MATCH (a {{name: {json.dumps(from_entity)}}})"
        f"-[r:{relation_type}]->"
        f"(b {{name: {json.dumps(to_entity)}}}) DELETE r RETURN 1 LIMIT 1",
        "x agtype",
    )
