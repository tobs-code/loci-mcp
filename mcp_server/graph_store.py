"""Graph storage: the one owner of the entity/relation lifecycle.

Every Cypher statement and every mirror-row write lives here, so a caller
works with entities and edges instead of AGE syntax. `GraphStore` is the
public seam: it owns node/edge creation, the relation-context memory, weight
inheritance, the mirror rows and orphan cleanup. The module-level helpers
below are private primitives; only `GraphStore` (and its `validate_weight`
static method) is part of the caller-facing interface.

The graph and the SQL mirror are updated together: a node without its mirror
row or an edge without its relation row would make `list_entities` and
`find_connections` disagree.
"""

from __future__ import annotations

import json
import re
from typing import Any

from mcp_server.config import SENTINEL_FORGOTTEN
from mcp_server.inference_gateway import embed
from mcp_server.scope_catalog import (
    _ag_str,
    _mem_table,
    _norm_entity,
    _safe_ident,
    _scope_key,
    _scope_label,
    _table_exists,
)
from mcp_server.schema_migrator import ensure_project_table
from mcp_server.storage import _store_memory, _supersede


def _dollar_quote(body: str) -> str:
    """Wrap a Cypher body in a dollar-quoted literal with a collision-free tag.

    AGE's `cypher()` requires its query argument to be a dollar-quoted string
    literal at parse time (it rejects a bind parameter), so the body cannot be
    passed via `%s`. A fixed `$$` delimiter lets a user-controlled value such
    as an entity name containing `$$` close the literal and inject SQL. The
    tag is therefore generated and verified not to occur in the body: no value
    can terminate the literal early.
    """
    tag = "$loci$"
    suffix = 0
    while tag in body:
        suffix += 1
        tag = f"$loci{suffix}$"
    return f"{tag}{body}{tag}"


def _cypher(cur, body: str, columns: str):
    """Run one Cypher statement against the shared AGE graph."""
    cur.execute("SET search_path TO ag_catalog, public")
    cur.execute(
        f"SELECT * FROM cypher('memory_graph', {_dollar_quote(body)}) "
        f"AS ({columns})"
    )
    # AGE needs ag_catalog on the path, but leaving it there makes later
    # unqualified SQL resolve against ag_catalog first: a project table
    # `mem_alpha` would be shadowed by an AGE label table of the same name,
    # so a write or forget would silently miss its rows. Reset on a second
    # cursor: it changes the same session without discarding the cypher
    # result the caller is about to fetch.
    reset = cur.connection.cursor()
    try:
        reset.execute("SET search_path TO public")
    finally:
        reset.close()
    return cur


_AGE_TYPE_SUFFIX = re.compile(r"::(?:vertex|edge)(?=\s*[,}\]])")


def _age_json(value: Any) -> Any:
    """Decode AGE's JSON-ish output without mangling string values.

    AGE appends `::vertex`/`::edge` to objects, which makes the value invalid
    JSON. A blind `.replace()` also rewrites those characters when they occur
    inside a quoted name, so only a type annotation that directly follows a
    value (before a comma, brace or bracket) is stripped.
    """
    return json.loads(_AGE_TYPE_SUFFIX.sub("", str(value)))


def _ensure_entity(cur, name: str, project: str | None,
                   ent_type: str | None = None) -> bool:
    """Create the AGE node and SQL entity row for one scope (idempotent)."""
    label = _scope_label(project)
    scope = _scope_key(project)
    name_json = json.dumps(name)
    scope_json = json.dumps(scope)
    # MATCH-then-CREATE is a time-of-check/time-of-use race: two concurrent
    # `relate` calls for the same new entity could both see no node and create
    # one each, leaving duplicate graph nodes. Serialize the check and the
    # creation for the rest of the transaction.
    #
    # One lock per SCOPE, not per (label, name): a per-entity lock taken in
    # caller order deadlocks when two transactions touch the same two entities
    # in opposite order (`relate(A, B)` vs `relate(B, A)`). A single scope key
    # has no ordering to invert and graph writes are short.
    cur.execute(
        "SELECT pg_advisory_xact_lock(hashtext(%s))",
        (label,),
    )
    _cypher(
        cur,
        f"MERGE (n:{label} {{name: {name_json}}}) "
        f"SET n.project = {scope_json} RETURN n",
        "n agtype",
    )
    cur.execute(
        "INSERT INTO entities (name, type, project) VALUES (%s, %s, %s) "
        "ON CONFLICT (name, project) DO NOTHING RETURNING 1",
        (name, ent_type or label, scope),
    )
    # `RETURNING (xmax = 0)` is not a reliable created-flag: under READ
    # COMMITTED a row just committed by another transaction can also report
    # xmax = 0, which made an update look like a creation. The insert either
    # produced a row (created) or hit the conflict and must update the type.
    created = cur.fetchone() is not None
    if not created:
        cur.execute(
            "UPDATE entities SET type = %s WHERE name = %s AND project = %s",
            (ent_type or label, name, scope),
        )
    return created


def _edge_weight(cur, project: str | None, from_entity: str, to_entity: str,
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


def _upsert_edge(cur, project: str | None, from_entity: str, to_entity: str,
                 relation_type: str, weight: float) -> bool:
    """MERGE one directed edge and its mirror row; True when it existed."""
    label = _scope_label(project)
    scope = _scope_key(project)
    existed = _edge_weight(cur, project, from_entity, to_entity, relation_type) \
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


def _delete_edge(cur, project: str | None, from_entity: str, to_entity: str,
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


def _entity_row(cur, project: str | None, name: str) -> tuple | None:
    """Mirror row of one entity in this scope."""
    cur.execute(
        "SELECT id, name, type, project FROM entities "
        "WHERE name = %s AND project = %s",
        (name, _scope_key(project)),
    )
    return cur.fetchone()


def _entity_exists(cur, project: str | None, name: str) -> bool:
    cur.execute(
        "SELECT 1 FROM entities WHERE name = %s AND project = %s LIMIT 1",
        (name, _scope_key(project)),
    )
    return cur.fetchone() is not None


def _node_exists(cur, project: str | None, name: str) -> bool:
    label = _scope_label(project)
    _cypher(
        cur,
        f"MATCH (n:{label} {{name: {json.dumps(name)}}}) RETURN n",
        "n agtype",
    )
    return cur.fetchone() is not None


def _find_entity(cur, project: str | None,
                 name: str) -> tuple | None:
    """Entity row, falling back to a graph-only node (AGE, no mirror row)."""
    row = _entity_row(cur, project, name)
    if row:
        return row
    if _node_exists(cur, project, name):
        return (None, name, _scope_label(project), _scope_key(project))
    return None


def _connections(cur, project: str | None, name: str) -> list[dict]:
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


def _traverse(cur, project: str | None, name: str, direction: str,
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


def _paths(cur, project: str | None, from_entity: str, to_entity: str,
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


def _entity_ids(cur, project: str | None) -> dict[str, int]:
    cur.execute("SELECT name, id FROM entities WHERE project = %s",
                (_scope_key(project),))
    return {name: entity_id for name, entity_id in cur.fetchall()}


def _edge_count(cur, project: str | None, name: str) -> int:
    label = _scope_label(project)
    _cypher(
        cur,
        f"MATCH (n:{label} {{name: {json.dumps(name)}}})-[r]-() "
        f"RETURN count(r)",
        "c agtype",
    )
    row = cur.fetchone()
    return int(str(row[0])) if row else 0


def _delete_node(cur, project: str | None, name: str) -> None:
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


def _drop_orphan_nodes(cur, project: str | None,
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
        if _edge_count(cur, project, name) > 0:
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
        _delete_node(cur, project, name)
        removed.append(name)
    return removed


def _edge_context(cur, project: str | None, from_entity: str, to_entity: str,
                  relation_type: str) -> str | None:
    """Latest active context sentence of one edge."""
    table = _mem_table(project)
    if not _table_exists(cur, table):
        return None
    cur.execute(
        f"SELECT content FROM {table} "
        "WHERE entity = %s AND superseded_by IS NULL "
        "AND deleted_at IS NULL "
        "AND (expires_at IS NULL OR expires_at > NOW()) "
        "AND metadata->>'relation_context' = 'true' "
        "AND metadata->>'from' = %s "
        "AND metadata->>'to' = %s "
        "AND metadata->>'relation_type' = %s "
        "ORDER BY updated_at DESC LIMIT 1",
        (f"{from_entity}_{relation_type}_{to_entity}",
         from_entity, to_entity, relation_type),
    )
    row = cur.fetchone()
    return row[0] if row else None


def _delete_scope_graph(cur, project: str | None) -> None:
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


def _delete_scope_nodes(cur, project: str | None) -> None:
    """Detach-delete a scope's nodes and their mirror rows, keep relations."""
    label = _scope_label(project)
    scope = _scope_key(project)
    _cypher(cur, f"MATCH (n:{label}) DETACH DELETE n", "r agtype")
    cur.execute("DELETE FROM entities WHERE project = %s", (scope,))


def _delete_entity_everywhere(cur, name: str) -> None:
    """Detach-delete one entity name from every scope and its mirror rows."""
    _cypher(
        cur,
        f"MATCH (n {{name: {json.dumps(name)}}}) DETACH DELETE n",
        "r agtype",
    )
    cur.execute("DELETE FROM entities WHERE name = %s", (name,))


def _boost_neighbors(cur, label: str, name: str) -> list[tuple]:
    """Up to ten edges around a node in one scope label (graph boost)."""
    _cypher(
        cur,
        f"MATCH (a:{label} {{name: {json.dumps(name)}}})-[r]-(b) "
        f"RETURN b.name, coalesce(r.weight, 1.0), type(r), a.name LIMIT 10",
        "name agtype, weight agtype, rtype agtype, source agtype",
    )
    return list(cur.fetchall())


def _label_node_exists(cur, label: str, name: str) -> bool:
    _cypher(
        cur,
        f"MATCH (a:{label} {{name: {json.dumps(name)}}}) RETURN a LIMIT 1",
        "a agtype",
    )
    return cur.fetchone() is not None


def _duplicate_edge_groups(cur) -> list[tuple]:
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


def _delete_duplicate_edge(cur, from_entity: str, to_entity: str,
                           relation_type: str) -> None:
    """Delete one duplicate AGE edge without the mirror row."""
    _cypher(
        cur,
        f"MATCH (a {{name: {json.dumps(from_entity)}}})"
        f"-[r:{relation_type}]->"
        f"(b {{name: {json.dumps(to_entity)}}}) DELETE r RETURN 1 LIMIT 1",
        "x agtype",
    )


def _forget_relation_context(cur, project: str | None, from_entity: str,
                             to_entity: str, relation_type: str) -> int:
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


class GraphStore:
    """One owner of the graph lifecycle for a caller's transaction.

    The store never commits or closes: the caller (a tool, a batch entry or a
    savepoint around `remember(links=...)`) owns the transaction. Construct it
    from a `MemoryStore` or, when a caller already holds a cursor inside an
    open transaction, with `GraphStore(cur=cur)`.
    """

    def __init__(self, db=None, *, cur=None):
        if cur is None:
            if db is None:
                raise ValueError(
                    "GraphStore needs a MemoryStore or an open cursor")
            cur = db.cur
        self.db = db
        self.cur = cur

    # -- validation ---------------------------------------------------------

    @staticmethod
    def validate_weight(value: Any) -> float:
        return _validate_weight(value)

    def ensure_scope(self, project: str | None) -> None:
        """A graph scope always has a memory table (and a registry entry)."""
        if project:
            ensure_project_table(self.cur, project)

    # -- write lifecycle ----------------------------------------------------

    def ensure_entity(self, name: str, project: str | None,
                      ent_type: str | None = None) -> bool:
        """Create the node and mirror row; True when it did not exist."""
        return _ensure_entity(self.cur, name, project, ent_type)

    def relate(self, edges: list[dict], project: str | None) -> list[dict]:
        """Create/update directed edges, mirror rows and context memories.

        Idempotent (MERGE). An omitted weight preserves the stored strength
        (1.0 only for a new edge). A repeated edge with the same context
        sentence reuses the existing memory (`context_reused`).
        """
        self.ensure_scope(project)
        prepared: list[dict] = []
        for edge in edges:
            frm = _norm_entity(edge.get("from_entity"))
            to = _norm_entity(edge.get("to_entity"))
            rtype = edge.get("relation_type")
            if not frm or not to:
                raise ValueError(
                    "each edge needs 'from_entity' and 'to_entity'")
            _safe_ident(rtype or "", "relation_type")
            raw_weight = edge.get("weight")
            prepared.append({
                "from_entity": frm,
                "to_entity": to,
                "relation_type": rtype,
                # None means "preserve what the edge already has"; validation
                # is skipped until the existing weight has been read.
                "weight": (None if raw_weight is None
                           else _validate_weight(raw_weight)),
                "context": edge.get("context"),
            })

        entities: list[str] = []
        seen: set[str] = set()
        for e in prepared:
            for ent in (e["from_entity"], e["to_entity"]):
                if ent not in seen:
                    seen.add(ent)
                    entities.append(ent)

        for ent in entities:
            _ensure_entity(self.cur, ent, project)

        results: list[dict] = []
        for e in prepared:
            rtype = e["relation_type"]
            existing_weight = _edge_weight(
                self.cur, project, e["from_entity"], e["to_entity"], rtype)
            existed = existing_weight is not None
            if e["weight"] is None:
                # No weight passed: keep the stored strength (default 1.0 only
                # for a brand-new edge). Otherwise an idempotent `relate` call
                # - e.g. adding a context sentence later - silently reset a
                # deliberately weak edge to 1.0.
                weight = existing_weight if existed else 1.0
            else:
                weight = e["weight"]
            e["weight"] = weight
            _upsert_edge(self.cur, project, e["from_entity"],
                         e["to_entity"], rtype, weight)
            # The context sentence of an edge carries the edge weight in its
            # metadata. A later weight change would leave the old value behind
            # and contradict the graph, so the two are kept in sync.
            _sync_context_weight(self.cur, project, e["from_entity"],
                                 e["to_entity"], rtype, weight)
            results.append({
                "from_entity": e["from_entity"],
                "to_entity": e["to_entity"],
                "relation_type": rtype,
                "weight": weight,
                "status": "updated" if existed else "created",
                "context_memory": None,
                "context_reused": False,
                "self_loop": e["from_entity"] == e["to_entity"],
            })

        context_edges = [(e, r) for e, r in zip(prepared, results)
                        if e["context"]]
        if context_edges:
            embeddings = embed([e["context"] for e, _ in context_edges])
            ctx_table = _mem_table(project)
            for (e, r), emb in zip(context_edges, embeddings):
                ctx_entity = (f"{e['from_entity']}_{e['relation_type']}_"
                              f"{e['to_entity']}")
                self.cur.execute(
                    f"SELECT id, content FROM {ctx_table} WHERE entity = %s "
                    f"AND superseded_by IS NULL AND deleted_at IS NULL "
                    f"AND (expires_at IS NULL OR expires_at > NOW()) "
                    f"AND metadata->>'from' = %s "
                    f"AND metadata->>'to' = %s "
                    f"AND metadata->>'relation_type' = %s "
                    f"ORDER BY updated_at DESC LIMIT 1",
                    (ctx_entity, e["from_entity"], e["to_entity"],
                     e["relation_type"]),
                )
                row = self.cur.fetchone()
                if row and row[1] == e["context"]:
                    # Re-relating the same edge with the same sentence is a
                    # no-op for the context memory. Storing a new copy and
                    # superseding the old one changed the id on every call, so
                    # an idempotent `relate` still produced a new memory
                    # version each time.
                    mem_id = row[0]
                    self.cur.execute(
                        f"UPDATE {ctx_table} SET updated_at = NOW() "
                        f"WHERE id = %s", (mem_id,))
                    r["context_reused"] = True
                else:
                    mem_id = _store_memory(
                        self.cur, e["context"], emb, "context", ctx_entity,
                        0.8,
                        {"relation_context": True,
                         "relation_type": e["relation_type"],
                         "from": e["from_entity"],
                         "to": e["to_entity"], "weight": e["weight"]},
                        project, None,
                    )
                    if row:
                        _supersede(self.cur, project, mem_id, row[0])
                    r["context_reused"] = False
                r["context_memory"] = mem_id

        return results

    # Backwards-compatible name for callers that describe the edge batch.
    relate_edges = relate

    def unrelate(self, from_entity: str, to_entity: str, relation_type: str,
                 project: str | None, keep_context: bool = False) -> dict:
        """Delete one edge, forget its sentence and remove orphan nodes."""
        deleted = _delete_edge(
            self.cur, project, from_entity, to_entity, relation_type)
        context_removed = 0
        orphans: list[str] = []
        if deleted:
            if not keep_context:
                context_removed = _forget_relation_context(
                    self.cur, project, from_entity, to_entity, relation_type)
            orphans = _drop_orphan_nodes(
                self.cur, project, [from_entity, to_entity])
        return {"deleted": deleted,
                "context_memories_removed": context_removed,
                "orphans_removed": orphans}

    def delete_entity(self, name: str, project: str | None = None, *,
                      every_scope: bool | None = None) -> None:
        """Remove an entity's node and mirror row (one scope or all)."""
        if every_scope is None:
            every_scope = project is None
        if every_scope:
            _delete_entity_everywhere(self.cur, name)
        else:
            _delete_node(self.cur, project, name)

    def delete_scope(self, project: str | None) -> None:
        """Remove a scope's nodes and every mirror row that belongs to it."""
        _delete_scope_graph(self.cur, project)

    def delete_scope_nodes(self, project: str | None) -> None:
        """Remove a scope's nodes and entity rows but keep relation rows."""
        _delete_scope_nodes(self.cur, project)

    # -- reads --------------------------------------------------------------

    def find_entity(self, project: str | None, name: str) -> tuple | None:
        return _find_entity(self.cur, project, name)

    def entity_row(self, project: str | None, name: str) -> tuple | None:
        return _entity_row(self.cur, project, name)

    def entity_exists(self, project: str | None, name: str) -> bool:
        return _entity_exists(self.cur, project, name)

    def entity_ids(self, project: str | None) -> dict[str, int]:
        return _entity_ids(self.cur, project)

    def scopes_with_entity(self, name: str) -> list[str]:
        self.cur.execute(
            "SELECT DISTINCT project FROM entities WHERE name = %s", (name,))
        return [row[0] for row in self.cur.fetchall()]

    def entity_details(self, project: str | None, name: str) -> dict | None:
        """The entity row plus its raw connections, or None when absent."""
        row = _find_entity(self.cur, project, name)
        if not row:
            return None
        return {"row": row, "connections": _connections(self.cur, project,
                                                         name)}

    def connections(self, entity: str, direction: str, project: str | None,
                    relation_type: str | None = None, limit: int = 20,
                    offset: int = 0) -> dict:
        """Paged neighbourhood with the relation-context sentence attached."""
        if not _entity_exists(self.cur, project, entity):
            return {"found": False, "total": 0, "results": [],
                    "scope": _scope_key(project)}
        total, results = _traverse(
            self.cur, project, entity, direction, relation_type, limit,
            offset)
        # An edge optionally carries a sentence explaining *why* the two
        # entities are connected. Traversal that returns only the label makes
        # the agent guess; the sentence is already stored as a relation-context
        # memory, so attach it to the edge it belongs to.
        for result in results:
            sentence = _edge_context(
                self.cur, project, result["from_entity"],
                result["to_entity"], result["relation"])
            if sentence:
                result["context"] = sentence
        return {"found": True, "total": total, "results": results,
                "scope": _scope_key(project)}

    def relation_page(self, entity: str | None = None,
                      relation_type: str | None = None,
                      project: str | None = None, limit: int = 50,
                      offset: int = 0,
                      all_scopes: bool | None = None) -> dict:
        """Browse mirror edges (strongest first) without a start entity."""
        if all_scopes is None:
            all_scopes = project is None or str(project).strip() == "*"
        conditions: list[str] = []
        params: list[Any] = []
        if not all_scopes:
            conditions.append("e1.project = %s")
            params.append(_scope_key(project))
        if entity:
            conditions.append("(e1.name = %s OR e2.name = %s)")
            params.extend([entity, entity])
        if relation_type:
            conditions.append("r.relation_type = %s")
            params.append(relation_type)
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        from_sql = (f"FROM relations r JOIN entities e1 ON e1.id = r.source_id "
                    f"JOIN entities e2 ON e2.id = r.target_id {where}")
        self.cur.execute(f"SELECT count(*) {from_sql}", params)
        total = int(self.cur.fetchone()[0] or 0)
        self.cur.execute(
            f"SELECT e1.name, e2.name, r.relation_type, r.weight, e1.project "
            f"{from_sql} "
            f"ORDER BY coalesce(r.weight, 1.0) DESC, e1.name, e2.name, "
            f"r.relation_type OFFSET %s LIMIT %s",
            params + [offset, limit],
        )
        results = [{
            "from_entity": row[0],
            "to_entity": row[1],
            "relation_type": row[2],
            "weight": float(row[3] or 1.0),
            "project": "global" if row[4] == "global" else row[4],
        } for row in self.cur.fetchall()]
        # Same rule as `find_connections`: an edge may carry a sentence that
        # explains why the entities are connected. Browsing without it made
        # the caller fetch every edge again just to learn the reason.
        for result in results:
            sentence = _edge_context(
                self.cur, None if result["project"] == "global"
                else result["project"], result["from_entity"],
                result["to_entity"], result["relation_type"])
            if sentence:
                result["context"] = sentence
        return {"total": total, "results": results}

    def paths(self, from_entity: str, to_entity: str, project: str | None,
              max_hops: int) -> dict:
        """Shortest AGE paths between two known entities of one scope."""
        scope = _scope_key(project)
        label = _scope_label(project)
        known_entities = set(_entity_ids(self.cur, project))
        missing = [name for name in (from_entity, to_entity)
                   if name not in known_entities]
        if missing:
            return {"missing": missing, "results": [], "scope": scope}
        if from_entity == to_entity:
            row = _entity_row(self.cur, project, from_entity)
            return {"missing": [], "scope": scope, "results": [{
                "nodes": [{"id": row[0] if row else None,
                           "name": from_entity, "label": label}],
                "relations": [], "hops": 0}]}

        rows = _paths(self.cur, project, from_entity, to_entity, max_hops)
        entity_ids = _entity_ids(self.cur, project)
        results: list[dict] = []
        seen_paths: set[tuple] = set()
        for r in rows:
            try:
                parsed = _age_json(r[0])
                parsed_rels = _age_json(r[1])
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
                edge = {
                    "relation_type": rel.get("label"),
                    "from": names[ids.index(start)] if start in ids else start,
                    "to": names[ids.index(end)] if end in ids else end,
                    "weight": props.get("weight", 1.0),
                }
                sentence = _edge_context(
                    self.cur, project, edge["from"], edge["to"],
                    edge["relation_type"])
                if sentence:
                    edge["context"] = sentence
                relations.append(edge)
            results.append({"nodes": nodes, "relations": relations,
                            "hops": len(nodes) - 1})
        return {"missing": [], "scope": scope, "results": results}

    # -- mirror bookkeeping -------------------------------------------------

    def entity_delete_preview(self, name: str,
                              project: str | None = None) -> dict:
        """What a delete would touch: entity rows, scopes and relation rows."""
        scope_clause = " AND project = %s" if project else ""
        params: tuple[Any, ...] = ((name, _scope_key(project)) if project
                                   else (name,))
        self.cur.execute(
            f"SELECT id, project FROM entities WHERE name = %s{scope_clause}",
            params,
        )
        rows = self.cur.fetchall()
        entity_ids = [row[0] for row in rows]
        scopes: list[str] = []
        for row in rows:
            scope_hit = row[1] or "global"
            if scope_hit not in scopes:
                scopes.append(scope_hit)
        relations = 0
        if entity_ids:
            self.cur.execute(
                "SELECT COUNT(*) FROM relations "
                "WHERE source_id = ANY(%s) OR target_id = ANY(%s)",
                (entity_ids, entity_ids),
            )
            relation_row = self.cur.fetchone()
            relations = relation_row[0] if relation_row else 0
        return {"entities": len(entity_ids), "relations": relations,
                "scopes": scopes}

    def scope_counts(self, project: str | None) -> dict:
        """Mirror row counts of one scope (for a scope-wide preview)."""
        scope = _scope_key(project)
        self.cur.execute(
            "SELECT COUNT(*) FROM entities WHERE project = %s", (scope,))
        entities = self.cur.fetchone()[0]
        self.cur.execute(
            "SELECT COUNT(*) FROM relations WHERE source_id IN "
            "(SELECT id FROM entities WHERE project = %s) OR target_id IN "
            "(SELECT id FROM entities WHERE project = %s)",
            (scope, scope),
        )
        relations = self.cur.fetchone()[0]
        return {"entities": entities, "relations": relations}

    # -- maintenance / search ----------------------------------------------

    def duplicate_groups(self) -> list[tuple]:
        return _duplicate_edge_groups(self.cur)

    def delete_duplicate(self, from_entity: str, to_entity: str,
                         relation_type: str) -> None:
        _delete_duplicate_edge(self.cur, from_entity, to_entity,
                               relation_type)

    def boost(self, label: str, name: str) -> list[tuple]:
        """Up to ten raw neighbours of a node in one scope label."""
        return _boost_neighbors(self.cur, label, name)

    def label_exists(self, label: str, name: str) -> bool:
        return _label_node_exists(self.cur, label, name)
