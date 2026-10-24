"""Callers reach the graph through GraphStore, not through private helpers."""

import json

from mcp_server.db import get_conn
from mcp_server.graph_store import GraphStore
from mcp_server.server import find_connections, get_entity, remember
from mcp_server.store import MemoryStore


def _run(fn):
    db = MemoryStore()
    try:
        result = fn(GraphStore(db))
        db.commit()
        return result
    finally:
        db.close()


def _edge_count(cur, a: str, b: str, rtype: str) -> int:
    cur.execute("SET search_path TO ag_catalog, public")
    cur.execute(
        "SELECT * FROM cypher('memory_graph', $$ "
        f"MATCH (a:Entity {{name: {json.dumps(a)}}})-[r:{rtype}]->"
        f"(b:Entity {{name: {json.dumps(b)}}}) RETURN count(r) $$) "
        "AS (n agtype)"
    )
    return int(str(cur.fetchone()[0]))


def test_get_entity_reads_the_graph_through_graph_store(monkeypatch):
    project = "seam-read"
    remember("Seam read fact", project=project, entity="SeamHub")

    calls: list[str] = []
    original = GraphStore.entity_details

    def _spy(self, project_arg, name):
        calls.append(name)
        return original(self, project_arg, name)

    monkeypatch.setattr(GraphStore, "entity_details", _spy)

    result = get_entity(entity="SeamHub", project=project)

    assert result["ok"] is True
    assert calls == ["SeamHub"]
    assert result["details"]["type"] == "memory-only"


def test_remember_links_go_through_graph_store(monkeypatch):
    project = "seam-link"
    calls: list[tuple] = []
    original = GraphStore.relate

    def _spy(self, edges, project_arg):
        calls.append(tuple(edge["to_entity"] for edge in edges))
        return original(self, edges, project_arg)

    monkeypatch.setattr(GraphStore, "relate", _spy)

    result = remember(
        "Seam link fact", project=project, entity="SeamFrom",
        links=[{"to_entity": "SeamTo", "relation_type": "LINKS"}],
    )

    assert result["ok"] is True
    assert calls == [("SeamTo",)]
    assert result["links"][0]["status"] == "created"


def test_graph_store_accepts_a_caller_cursor_and_is_idempotent():
    conn = get_conn()
    try:
        cur = conn.cursor()
        graph = GraphStore(cur=cur)

        assert graph.ensure_entity("SeamEntity", "seam-cursor") is True
        assert graph.ensure_entity("SeamEntity", "seam-cursor") is False
        assert graph.entity_exists("seam-cursor", "SeamEntity") is True
        conn.commit()
    finally:
        conn.close()


def test_relate_preserves_an_existing_weight_when_omitted():
    def _body(graph):
        first = graph.relate([{
            "from_entity": "WeightA", "to_entity": "WeightB",
            "relation_type": "USES", "weight": 0.4,
        }], None)
        again = graph.relate([{
            "from_entity": "WeightA", "to_entity": "WeightB",
            "relation_type": "USES",
        }], None)
        fresh = graph.relate([{
            "from_entity": "WeightC", "to_entity": "WeightD",
            "relation_type": "USES",
        }], None)
        return first, again, fresh

    first, again, fresh = _run(_body)

    assert first[0]["status"] == "created"
    assert again[0]["status"] == "updated"
    assert again[0]["weight"] == 0.4
    assert fresh[0]["weight"] == 1.0


def test_relate_reuses_an_identical_context_sentence():
    edges = [{
        "from_entity": "ContextA", "to_entity": "ContextB",
        "relation_type": "EXPLAINS", "context": "A explains B.",
    }]

    first = _run(lambda graph: graph.relate(edges, None))
    second = _run(lambda graph: graph.relate(edges, None))

    assert first[0]["context_reused"] is False
    assert second[0]["context_reused"] is True
    assert first[0]["context_memory"] == second[0]["context_memory"]


def test_unrelate_forgets_the_context_and_removes_orphans():
    _run(lambda graph: graph.relate([{
        "from_entity": "OrphanA", "to_entity": "OrphanB",
        "relation_type": "LINKS", "context": "OrphanA links OrphanB.",
    }], None))

    result = _run(lambda graph: graph.unrelate(
        "OrphanA", "OrphanB", "LINKS", None))

    assert result["deleted"] is True
    assert result["context_memories_removed"] == 1
    assert sorted(result["orphans_removed"]) == ["OrphanA", "OrphanB"]


def test_self_loop_is_stored_with_a_flag():
    result = _run(lambda graph: graph.relate([{
        "from_entity": "LoopNode", "to_entity": "LoopNode",
        "relation_type": "SELF",
    }], None))

    assert result[0]["self_loop"] is True


def test_project_and_global_graphs_stay_isolated():
    _run(lambda graph: graph.relate([{
        "from_entity": "Isolated", "to_entity": "ProjectPeer",
        "relation_type": "LINKS",
    }], "seam-iso"))
    _run(lambda graph: graph.relate([{
        "from_entity": "Isolated", "to_entity": "GlobalPeer",
        "relation_type": "LINKS",
    }], None))

    project = find_connections(entity="Isolated", project="seam-iso")
    global_view = find_connections(entity="Isolated")

    project_names = {row["entity"] for row in project["results"]}
    global_names = {row["entity"] for row in global_view["results"]}
    assert project_names == {"ProjectPeer"}
    assert global_names == {"GlobalPeer"}


def test_duplicate_groups_are_reported_and_cleaned_up():
    db = MemoryStore()
    try:
        graph = GraphStore(db)
        graph.relate([{
            "from_entity": "DupA", "to_entity": "DupB",
            "relation_type": "DUP",
        }], None)
        for _ in range(2):
            graph.cur.execute(
                "SET search_path TO ag_catalog, public")
            graph.cur.execute(
                "SELECT * FROM cypher('memory_graph', $$ "
                "MATCH (a:Entity {name: 'DupA'}), (b:Entity {name: 'DupB'}) "
                "CREATE (a)-[:DUP]->(b) RETURN 1 $$) AS (x agtype)"
            )
        db.commit()

        groups = [(str(src).strip('"'), str(tgt).strip('"'),
                   str(rtype).strip('"'), cnt)
                  for src, tgt, rtype, cnt in graph.duplicate_groups()]
        group = next(g for g in groups if g[1] == "DupB")
        assert int(str(group[3])) == 3

        for _ in range(int(str(group[3])) - 1):
            graph.delete_duplicate(group[0], group[1], group[2])
        db.commit()

        assert _edge_count(graph.cur, "DupA", "DupB", "DUP") == 1
    finally:
        db.close()
