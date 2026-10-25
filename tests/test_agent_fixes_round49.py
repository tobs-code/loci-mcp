"""Round-49 normal-supersede scope resolution.

An id is unique per scope, so an import can leave an inactive project copy of
a fact whose global row is still active. Target resolution for the normal
`remember`/`remember_batch` supersede path must only consider active rows:
otherwise the inactive copy hides the real target, `mark_superseded` runs
against the wrong table, and the write reports a replacement that never
happened.
"""

from mcp_server.db import get_conn
from mcp_server.scope_catalog import _mem_table
from mcp_server.server import (
    export_memories,
    forget,
    import_memories,
    remember,
    remember_batch,
)


def _row(project, memory_id):
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            f"SELECT superseded_by, deleted_at, updated_at FROM "
            f"{_mem_table(project)} WHERE id = %s",
            (memory_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return {
            "superseded_by": row[0],
            "deleted_at": row[1],
            "updated_at": row[2],
        }
    finally:
        conn.close()


def _copy_scope(source_project, target_project, memory_id):
    """Re-import one row into another scope, keeping its id."""
    exported = export_memories(project=source_project, include_graph=False)
    assert exported["ok"] is True, exported.get("error")
    payload = dict(exported["data"])
    payload["records"] = [
        dict(record, project=target_project)
        for record in payload["records"]
        if record.get("id") == memory_id
    ]
    assert len(payload["records"]) == 1
    imported = import_memories(data=payload, format="json")
    assert imported["ok"] is True, imported.get("error")


def _inactive_project_shadow(content, entity):
    """Global active fact + inactive project copy sharing its id."""
    project = "round49-normal-shadow"
    global_fact = remember(content, entity=entity)
    assert global_fact["ok"] is True, global_fact.get("error")
    _copy_scope("global", project, global_fact["id"])
    forgotten = forget(global_fact["id"], project=project)
    assert forgotten["ok"] is True, forgotten.get("error")
    before = _row(project, global_fact["id"])
    assert before["deleted_at"] is not None
    return project, global_fact["id"], before


def test_normal_supersede_reaches_the_active_global_target():
    project, old_id, before = _inactive_project_shadow(
        "Round49 normal shadow target lives globally", "ShadowGlobal")

    result = remember(
        "Round49 normal shadow replacement lives in the project",
        project=project,
        supersede=old_id,
    )

    assert result["ok"] is True, result.get("error")
    assert result["status"] == "stored"
    assert result["superseded"] == [old_id]
    assert result["cross_scope_superseded"] == [old_id]
    assert any("cross-scope supersede" in w for w in result["warnings"])

    global_row = _row(None, old_id)
    assert global_row["superseded_by"] == result["id"]
    # The inactive copy is neither revived, overwritten nor superseded again.
    assert _row(project, old_id) == before


def test_supersede_content_reaches_the_active_global_target():
    project, old_id, before = _inactive_project_shadow(
        "Round49 content shadow target lives globally", "ShadowContent")
    target_content = "Round49 content shadow target lives globally"

    result = remember(
        "Round49 content shadow replacement lives in the project",
        project=project,
        supersede_content=target_content,
    )

    assert result["ok"] is True, result.get("error")
    assert result["status"] == "stored"
    assert result["superseded"] == [old_id]
    assert result["cross_scope_superseded"] == [old_id]

    assert _row(None, old_id)["superseded_by"] == result["id"]
    assert _row(project, old_id) == before


def test_normal_supersede_keeps_an_inactive_global_copy_untouched():
    project = "round49-normal-reverse-shadow"
    project_fact = remember(
        "Round49 reverse shadow target lives in the project",
        entity="ShadowReverse",
        project=project,
    )
    assert project_fact["ok"] is True, project_fact.get("error")
    _copy_scope(project, "global", project_fact["id"])
    assert forget(project_fact["id"], project="global")["ok"] is True
    before = _row(None, project_fact["id"])
    assert before["deleted_at"] is not None

    result = remember(
        "Round49 reverse shadow replacement lives in the project",
        project=project,
        supersede=project_fact["id"],
    )

    assert result["ok"] is True, result.get("error")
    assert result["status"] == "stored"
    assert result["superseded"] == [project_fact["id"]]
    assert "cross_scope_superseded" not in result

    assert _row(project, project_fact["id"])["superseded_by"] == result["id"]
    assert _row(None, project_fact["id"]) == before


def test_same_scope_supersede_still_replaces_the_target():
    project = "round49-normal-same-scope"
    old = remember("Round49 same-scope relay is enabled", project=project)
    assert old["ok"] is True, old.get("error")

    result = remember(
        "Round49 same-scope relay is disabled",
        project=project,
        supersede=old["id"],
    )

    assert result["ok"] is True, result.get("error")
    assert result["status"] == "stored"
    assert result["superseded"] == [old["id"]]
    assert "cross_scope_superseded" not in result
    assert _row(project, old["id"])["superseded_by"] == result["id"]


def test_legitimate_cross_scope_supersede_is_still_reported():
    old = remember("Round49 cross-scope relay is enabled", scope="global")
    assert old["ok"] is True, old.get("error")

    result = remember(
        "Round49 cross-scope relay is disabled",
        project="round49-normal-cross-scope",
        supersede=old["id"],
    )

    assert result["ok"] is True, result.get("error")
    assert result["status"] == "stored"
    assert result["superseded"] == [old["id"]]
    assert result["cross_scope_superseded"] == [old["id"]]
    assert any("cross-scope supersede" in w for w in result["warnings"])
    assert _row(None, old["id"])["superseded_by"] == result["id"]


def test_remember_batch_uses_the_same_active_target_resolution():
    project, old_id, before = _inactive_project_shadow(
        "Round49 batch shadow target lives globally", "ShadowBatch")

    result = remember_batch(facts=[{
        "content": "Round49 batch shadow replacement lives in the project",
        "project": project,
        "supersede": old_id,
    }])

    assert result["ok"] is True, result.get("error")
    entry = result["results"][0]
    assert entry["status"] == "stored"
    assert entry["superseded"] == [old_id]
    assert entry["cross_scope_superseded"] == [old_id]
    assert _row(None, old_id)["superseded_by"] == entry["id"]
    assert _row(project, old_id) == before


def test_contradiction_auto_supersede_reaches_the_active_global_target():
    project, old_id, before = _inactive_project_shadow(
        "Round49 contradiction relay is not enabled", "ShadowContradiction")

    result = remember(
        "Round49 contradiction relay is enabled",
        entity="ShadowContradiction",
        project=project,
        check_contradictions=True,
    )

    assert result["ok"] is True, result.get("error")
    assert result["status"] == "stored"
    assert result["superseded"] == [old_id]
    assert result["cross_scope_superseded"] == [old_id]
    assert [pair["existing_id"] for pair
            in result["contradictions"]["superseded"]] == [old_id]

    assert _row(None, old_id)["superseded_by"] == result["id"]
    assert _row(project, old_id) == before
