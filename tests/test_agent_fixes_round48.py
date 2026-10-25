"""Round-48 write-pipeline consistency fixes.

Covers:
- `links` entity names are validated with the same rules as
  `GraphStore.relate` before anything is stored;
- a failed link savepoint does not report rolled-back edges as created;
- a project-scoped duplicate supersede does not touch a colliding id in the
  global scope (and a real cross-scope supersede still hides the global fact,
  even when an inactive project copy shares the id);
- an omitted link weight preserves an existing edge's weight on the duplicate
  path instead of silently resetting it to 1.0.
"""

from mcp_server.db import get_conn
from mcp_server.graph_store import GraphStore
from mcp_server.scope_catalog import _mem_table
from mcp_server.server import (
    export_memories,
    find_connections,
    forget,
    import_memories,
    list_memories,
    list_relations,
    remember,
)


def _row(project, memory_id):
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            f"SELECT superseded_by, deleted_at FROM {_mem_table(project)} "
            f"WHERE id = %s",
            (memory_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return {"superseded_by": row[0], "deleted_at": row[1]}
    finally:
        conn.close()


def _clone_scope_to_global(project):
    """Re-import a project's rows into the global scope, keeping their ids."""
    exported = export_memories(project=project, include_graph=False)
    assert exported["ok"] is True, exported.get("error")
    payload = dict(exported["data"])
    payload["records"] = [dict(record, project="global")
                          for record in payload["records"]]
    imported = import_memories(data=payload, format="json")
    assert imported["ok"] is True, imported.get("error")


def test_links_reject_a_reserved_entity_character_before_the_write():
    content = "Round48 fact with an invalid link target"
    result = remember(
        content,
        project="round48-links-invalid",
        entity="ValidSource",
        links=[{"to_entity": "Bad$Target", "relation_type": "USES"}],
    )
    assert result["ok"] is False
    assert result["id"] is None
    assert "links[0]" in result["error"]
    assert "$" in result["error"]

    stored = list_memories(project="round48-links-invalid", contains=content)
    assert stored["total"] == 0


def test_links_reject_a_reserved_from_entity_character_before_the_write():
    result = remember(
        "Round48 fact with an invalid link source",
        project="round48-links-invalid-from",
        entity="ValidTarget",
        links=[{"from_entity": "Bad`Source",
                "to_entity": "Target",
                "relation_type": "USES"}],
    )
    assert result["ok"] is False
    assert result["id"] is None
    assert "links[0]" in result["error"]
    assert "from_entity" in result["error"]


def test_successful_links_are_still_reported():
    result = remember(
        "Round48 fact with two valid links",
        project="round48-links-ok",
        entity="GoodSource",
        links=[
            {"to_entity": "TargetA", "relation_type": "USES"},
            {"to_entity": "TargetB", "relation_type": "USES", "weight": 0.4},
        ],
    )
    assert result["ok"] is True, result.get("error")
    assert result["status"] == "stored"
    assert [(edge["to_entity"], edge["status"], edge["weight"])
            for edge in result["links"]] == [
        ("TargetA", "created", 1.0),
        ("TargetB", "created", 0.4),
    ]

    edges = find_connections("GoodSource", project="round48-links-ok")
    assert edges["ok"] is True
    assert {(edge["entity"], edge["relation"]) for edge in edges["results"]} == {
        ("TargetA", "USES"), ("TargetB", "USES")}


def test_link_failure_reports_no_rolled_back_edge(monkeypatch):
    real_relate = GraphStore.relate

    def flaky_relate(self, edges, project):
        if edges[0]["to_entity"] == "BrokenTarget":
            raise RuntimeError("graph write failed")
        return real_relate(self, edges, project)

    monkeypatch.setattr(GraphStore, "relate", flaky_relate)

    content = "Round48 fact whose second edge fails"
    result = remember(
        content,
        project="round48-links-rollback",
        entity="RollbackSource",
        links=[
            {"to_entity": "KeptTarget", "relation_type": "CREATES"},
            {"to_entity": "RollbackSource", "relation_type": "CREATES"},
            {"to_entity": "BrokenTarget", "relation_type": "CREATES"},
        ],
    )
    assert result["ok"] is True, result.get("error")
    assert result["status"] == "stored"
    assert "links" not in result
    assert any("links failed" in warning for warning in result["warnings"])
    # A self-loop warning from an edge that the rollback removed is not part
    # of the persisted state either.
    assert not any("self-loop" in warning for warning in result["warnings"])

    # The fact survived the link savepoint, but no edge did - including the
    # first one that succeeded before the failing edge.
    stored = list_memories(project="round48-links-rollback", contains=content)
    assert stored["total"] == 1
    relations = list_relations(project="round48-links-rollback")
    assert relations["ok"] is True
    assert relations["results"] == []


def test_duplicate_without_weight_preserves_the_existing_edge_weight():
    content = "Round48 weighted edge is kept on duplicate writes"
    project = "round48-link-weight"
    first = remember(
        content,
        project=project,
        entity="WeightSource",
        links=[{"to_entity": "WeightTarget", "relation_type": "LINKS",
                "weight": 0.2}],
    )
    assert first["ok"] is True, first.get("error")
    assert first["links"][0]["weight"] == 0.2

    again = remember(
        content,
        project=project,
        entity="WeightSource",
        links=[{"to_entity": "WeightTarget", "relation_type": "LINKS"}],
    )
    assert again["status"] == "duplicate"
    assert again["links"][0]["status"] == "updated"
    assert again["links"][0]["weight"] == 0.2

    reweighted = remember(
        content,
        project=project,
        entity="WeightSource",
        links=[{"to_entity": "WeightTarget", "relation_type": "LINKS",
                "weight": 1.0}],
    )
    assert reweighted["links"][0]["status"] == "updated"
    assert reweighted["links"][0]["weight"] == 1.0

    fresh = remember(
        content,
        project=project,
        entity="WeightSource",
        links=[{"to_entity": "FreshTarget", "relation_type": "LINKS"}],
    )
    assert fresh["links"][0]["status"] == "created"
    assert fresh["links"][0]["weight"] == 1.0

    targets = {(edge["entity"], edge["weight"])
               for edge in find_connections("WeightSource",
                                            project=project)["results"]}
    assert ("WeightTarget", 1.0) in targets
    assert ("FreshTarget", 1.0) in targets


def test_duplicate_supersede_in_a_project_leaves_a_colliding_global_id_alone():
    project = "round48-dup-supersede"
    old = remember("Round48 old project fact", project=project,
                   entity="ScopeOld")
    duplicate = remember("Round48 duplicate project fact", project=project,
                         entity="ScopeDuplicate")
    assert old["ok"] is True and duplicate["ok"] is True

    # An import can create the same ids in the global scope. The global rows
    # are separate memories, so a project-scoped supersede must not touch them.
    _clone_scope_to_global(project)
    assert _row(None, old["id"])["superseded_by"] is None

    result = remember(
        "Round48 duplicate project fact",
        project=project,
        entity="ScopeDuplicate",
        supersede=old["id"],
    )
    assert result["ok"] is True, result.get("error")
    assert result["status"] == "duplicate"
    assert result["id"] == duplicate["id"]
    assert result["superseded"] == [old["id"]]

    assert _row(project, old["id"])["superseded_by"] == duplicate["id"]
    assert _row(None, old["id"])["superseded_by"] is None


def test_duplicate_supersede_still_hides_a_real_global_fact():
    project = "round48-dup-supersede-global"
    global_fact = remember("Round48 old global fact", entity="GlobalOld")
    assert global_fact["ok"] is True, global_fact.get("error")
    duplicate = remember("Round48 project duplicate fact", project=project,
                         entity="GlobalDuplicate")
    assert duplicate["ok"] is True

    result = remember(
        "Round48 project duplicate fact",
        project=project,
        entity="GlobalDuplicate",
        supersede=global_fact["id"],
    )
    assert result["ok"] is True, result.get("error")
    assert result["status"] == "duplicate"
    assert result["superseded"] == [global_fact["id"]]
    assert result["cross_scope_superseded"] == [global_fact["id"]]
    assert _row(None, global_fact["id"])["superseded_by"] == duplicate["id"]

def test_duplicate_supersede_finds_an_active_global_target_behind_an_inactive_project_copy():
    project = "round48-dup-supersede-shadow"
    global_fact = remember("Round48 target lives globally", entity="ShadowOld")
    assert global_fact["ok"] is True, global_fact.get("error")

    exported = export_memories(project="global", include_graph=False)
    assert exported["ok"] is True, exported.get("error")
    imported = import_memories(data=exported["data"], format="json",
                               project=project)
    assert imported["ok"] is True, imported.get("error")
    # The project copy becomes inactive; the global row with the same id
    # stays active and is therefore the only valid supersede target.
    assert forget(global_fact["id"], project=project)["ok"] is True

    duplicate = remember("Round48 duplicate lives in project", project=project,
                         entity="ShadowDuplicate")
    assert duplicate["ok"] is True, duplicate.get("error")
    result = remember(
        "Round48 duplicate lives in project",
        project=project,
        entity="ShadowDuplicate",
        supersede=global_fact["id"],
    )
    assert result["ok"] is True, result.get("error")
    assert result["status"] == "duplicate"
    assert result["superseded"] == [global_fact["id"]]
    assert result["cross_scope_superseded"] == [global_fact["id"]]

    assert _row(None, global_fact["id"])["superseded_by"] == duplicate["id"]
    project_copy = _row(project, global_fact["id"])
    assert project_copy["deleted_at"] is not None
    assert project_copy["superseded_by"] != duplicate["id"]
