"""Round-48 duplicate-supersede scope isolation.

Covers:
- a project-scoped duplicate supersede does not touch a colliding id in the
  global scope (and a real cross-scope supersede still hides the global fact,
  even when an inactive project copy shares the id).
"""

from mcp_server.db import get_conn
from mcp_server.scope_catalog import _mem_table
from mcp_server.server import (
    export_memories,
    forget,
    import_memories,
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
