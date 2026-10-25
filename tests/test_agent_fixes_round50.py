"""Round-50 narrow follow-ups around supersede scope handling.

Covers:
- the legacy graph-store relation-context supersede must not touch a
  colliding active id in the global scope;
- metadata carryover after a cross-scope supersede reads the scope that
  actually owned the replaced row, not an expired copy in the write scope;
- an id that is active in the write scope and in global resolves to the
  write scope explicitly (the other active copy is never touched), including
  when the replacement is an automatic near-duplicate supersede.
"""

from datetime import UTC, datetime

from mcp_server.db import get_conn
from mcp_server.scope_catalog import _mem_table
from mcp_server.server import (
    export_memories,
    import_memories,
    relate,
    remember,
)


def _row(project, memory_id):
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            f"SELECT superseded_by, deleted_at, metadata FROM "
            f"{_mem_table(project)} WHERE id = %s",
            (memory_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return {
            "superseded_by": row[0],
            "deleted_at": row[1],
            "metadata": row[2],
        }
    finally:
        conn.close()


def _export_one(project, memory_id):
    exported = export_memories(project=project, include_graph=False)
    assert exported["ok"] is True, exported.get("error")
    records = [record for record in exported["data"]["records"]
               if record.get("id") == memory_id]
    assert len(records) == 1
    return dict(records[0])


def _import_one(project, record):
    payload = {"records": [dict(record, project=project)]}
    imported = import_memories(data=payload, format="json")
    assert imported["ok"] is True, imported.get("error")


def test_relate_context_supersede_does_not_touch_a_colliding_global_id():
    project = "round50-legacy-context"
    old = relate(
        "Round50LegacyFrom", "Round50LegacyTo", "ROUND50_LEGACY",
        context="Round50 relation context old", project=project,
    )
    assert old["ok"] is True, old.get("error")
    context_id = old["context_memory"]
    assert context_id

    # Put an active global row with the same id next to the project context
    # memory: an import can produce exactly this collision.
    _import_one("global", _export_one(project, context_id))
    before = _row("global", context_id)
    assert before["superseded_by"] is None

    changed = relate(
        "Round50LegacyFrom", "Round50LegacyTo", "ROUND50_LEGACY",
        context="Round50 relation context new", project=project,
    )
    assert changed["ok"] is True, changed.get("error")
    assert changed["context_reused"] is False
    assert changed["context_memory"] != context_id

    # Only the project context row was replaced. The global row is a
    # different memory that merely shares the id.
    assert _row(project, context_id)["superseded_by"] == changed["context_memory"]
    assert _row("global", context_id) == before


def test_metadata_carryover_uses_the_scope_that_owned_the_target():
    project = "round50-metadata-carry"
    global_fact = remember(
        "Round50 metadata target lives globally",
        scope="global",
        metadata={"origin": "global", "keep": "global"},
    )
    assert global_fact["ok"] is True, global_fact.get("error")

    expired_copy = _export_one("global", global_fact["id"])
    expired_copy["metadata"] = {"origin": "project-expired"}
    expired_copy["expires_at"] = datetime(2000, 1, 1, tzinfo=UTC).isoformat()
    _import_one(project, expired_copy)
    expired_before = _row(project, global_fact["id"])
    assert expired_before["deleted_at"] is None
    assert expired_before["metadata"]["origin"] == "project-expired"

    result = remember(
        "Round50 metadata replacement lives in the project",
        project=project,
        supersede=global_fact["id"],
    )

    assert result["ok"] is True, result.get("error")
    assert result["superseded"] == [global_fact["id"]]
    assert result["cross_scope_superseded"] == [global_fact["id"]]

    carried = _row(project, result["id"])["metadata"]
    assert carried["origin"] == "global"
    assert carried["keep"] == "global"
    assert _row(project, global_fact["id"]) == expired_before


def test_an_id_active_in_both_scopes_resolves_to_the_write_scope():
    project = "round50-active-active"
    global_fact = remember(
        "Round50 active-active seed lives globally", scope="global")
    assert global_fact["ok"] is True, global_fact.get("error")
    _import_one(project, _export_one("global", global_fact["id"]))
    before_global = _row("global", global_fact["id"])
    assert _row(project, global_fact["id"])["superseded_by"] is None
    assert before_global["superseded_by"] is None

    result = remember(
        "Round50 active-active replacement lives in the project",
        project=project,
        supersede=global_fact["id"],
    )

    assert result["ok"] is True, result.get("error")
    assert result["status"] == "stored"
    assert result["superseded"] == [global_fact["id"]]
    assert "cross_scope_superseded" not in result
    assert any(
        "active in both" in warning and "took precedence" in warning
        for warning in result["warnings"])
    assert _row(project, global_fact["id"])["superseded_by"] == result["id"]
    assert _row("global", global_fact["id"]) == before_global


def test_automatic_near_duplicate_supersede_warns_about_the_precedence():
    project = "round50-auto-near-active-active"
    global_fact = remember(
        "Round50 auto-near relay listens on port 7500",
        scope="global",
        entity="Round50AutoNear",
    )
    assert global_fact["ok"] is True, global_fact.get("error")
    _import_one(project, _export_one("global", global_fact["id"]))
    before_global = _row("global", global_fact["id"])

    result = remember(
        "Round50 auto-near relay now listens on port 7500",
        project=project,
        entity="Round50AutoNear",
        dedupe="off",
        on_near_duplicate="supersede",
    )

    assert result["ok"] is True, result.get("error")
    assert result["status"] == "stored"
    assert result["superseded"] == [global_fact["id"]]
    assert "cross_scope_superseded" not in result
    assert any(
        "active in both" in warning and "took precedence" in warning
        for warning in result["warnings"])
    assert _row(project, global_fact["id"])["superseded_by"] == result["id"]
    assert _row("global", global_fact["id"]) == before_global
