"""Regression tests for the seventeenth round of fixes.

Covers:
- `list_projects(include_counts=false)` no longer mixes a real active count
  with zero placeholders for the other counts;
- an unknown `entity_type` is an argument error that names the valid labels;
- `update_batch` accepts `touch` like `update_memory`;
- a non-object metadata value names its actual type;
- restoring an existing but active/superseded memory explains why it cannot
  be restored;
- `verify_integrity` lists supersede chains instead of only counting them;
- `export_memories(entity=...)` keeps the other endpoints of its edges;
- a write that duplicates content across scopes warns instead of hiding the
  fork.
"""

from mcp_server.db import get_conn
from mcp_server.scopes import _mem_table
from mcp_server.server import (
    export_memories,
    forget,
    list_entities,
    list_memories,
    list_projects,
    relate,
    remember,
    remember_batch,
    update_batch,
    update_memory,
    verify_integrity,
)


def _exec(sql: str, params: tuple = ()) -> None:
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(sql, params)
        conn.commit()
    finally:
        conn.close()


def test_list_projects_without_counts_omits_count_keys():
    project = "round17-counts"
    remember("Round17 count probe", project=project)

    rows = {row["project"]: row for row in
            list_projects(include_counts=False)["results"]}
    row = rows[project]
    assert row["has_table"] is True
    assert "memories" not in row
    assert "total_memories" not in row
    assert "superseded_memories" not in row
    assert "forgotten_memories" not in row
    assert "entities" not in row
    assert "relations" not in row

    full = {r["project"]: r for r in list_projects()["results"]}[project]
    assert full["memories"] == 1
    assert full["total_memories"] == 1


def test_unknown_entity_type_is_rejected_with_valid_labels():
    project = "round17-entity-type"
    relate("Alpha", "Beta", "uses", project=project)

    bad = list_entities(entity_type="NotARealLabel", project=project)
    assert bad["ok"] is False
    assert "unknown entity_type" in bad["error"]
    assert "NotARealLabel" in bad["error"]

    labels = {row["label"] for row in
              list_entities(entity_type="graph", project=project)["results"]}
    assert labels
    raw = list_entities(entity_type=sorted(labels)[0], project=project)
    assert raw["ok"] is True


def test_update_batch_supports_touch():
    project = "round17-touch"
    stored = remember("Round17 touch probe", project=project,
                      importance=0.4)
    before = list_memories(memory_id=stored["id"], project=project,
                           fields=["id", "updated_at", "content"])["memory"]

    result = update_batch([{"memory_id": stored["id"], "touch": True}],
                          project=project)
    assert result["ok"] is True
    assert result["results"][0]["status"] == "updated"

    after = list_memories(memory_id=stored["id"], project=project,
                          fields=["id", "updated_at", "content"])["memory"]
    assert after["content"] == before["content"]
    assert after["updated_at"] >= before["updated_at"]

    conflict = update_batch(
        [{"memory_id": stored["id"], "touch": True, "restore": True}],
        project=project)
    assert conflict["results"][0]["status"] == "error"
    assert "cannot be combined" in conflict["results"][0]["error"]


def test_metadata_with_wrong_type_names_the_type():
    result = remember("Round17 metadata type probe", metadata=["env", "test"],
                      project="round17-metadata")
    assert result["ok"] is False
    assert "got list" in result["error"]
    assert "plain string" not in result["error"]


def test_restore_explains_active_and_superseded_rows():
    project = "round17-restore"
    old = remember("Round17 restore old fact", project=project)
    active = update_memory(old["id"], restore=True, project=project)
    assert active["ok"] is False
    assert active["status"] == "conflict"
    assert "is active" in active["error"]
    assert "nothing to restore" in active["error"]

    replacement = remember("Round17 restore new fact", project=project,
                           supersede=old["id"])
    assert replacement["ok"] is True
    superseded = update_memory(old["id"], restore=True, project=project)
    assert superseded["ok"] is False
    assert "superseded by" in superseded["error"]
    assert "does not undo a supersede" in superseded["error"]


def test_verify_integrity_lists_supersede_chains():
    project = "round17-chains"
    a = remember("Round17 chain A", project=project)
    b = remember("Round17 chain B", project=project)
    c = remember("Round17 chain C", project=project)
    table = _mem_table(project)
    _exec(f"UPDATE {table} SET superseded_by = %s WHERE id = %s",
          (b["id"], a["id"]))
    _exec(f"UPDATE {table} SET superseded_by = %s WHERE id = %s",
          (c["id"], b["id"]))

    report = verify_integrity(project=project)
    assert report["ok"] is True
    assert report["chains"] >= 1
    chain_rows = [r for r in report["results"]
                  if r["problem"] == "supersede_chain"]
    assert chain_rows
    row = next(r for r in chain_rows if r["id"] == a["id"])
    assert row["superseded_by"] == b["id"]
    assert row["chain_to"] == c["id"]


def test_export_entity_filter_keeps_edge_endpoints():
    project = "round17-export-graph"
    relate("BillingService", "PostgresDB", "uses", project=project,
           context="BillingService writes to PostgresDB")
    relate("ReportingService", "PostgresDB", "uses", project=project,
           context="ReportingService reads PostgresDB")

    exported = export_memories(project=project, entity="BillingService")
    graph = exported["data"]["graph"]
    assert {e["name"] for e in graph["entities"]} == {
        "BillingService", "PostgresDB"}
    assert {(e["from_entity"], e["to_entity"]) for e in graph["relations"]} == {
        ("BillingService", "PostgresDB")}


def test_cross_scope_duplicate_write_warns():
    text = "Round17 cross-scope write probe 7712"
    first = remember(text)
    assert first["ok"] is True
    assert "cross_scope_duplicates" not in first

    second = remember(text, project="round17-dup")
    assert second["ok"] is True
    assert second["status"] == "stored"
    assert second["cross_scope_duplicates"] == [
        {"id": first["id"], "project": "global"}]
    assert any("another scope" in w for w in second["warnings"])

    batched = remember_batch(
        [{"content": "Round17 cross-scope batch probe 9915"}], project="round17-dup-batch")
    assert batched["ok"] is True
    duplicate = remember_batch(
        [{"content": "Round17 cross-scope batch probe 9915"}], project="round17-dup-batch-2")
    assert duplicate["ok"] is True
    entry = duplicate["results"][0]
    assert entry["cross_scope_duplicates"][0]["project"] == "round17-dup-batch"
    assert any("another scope" in w for w in entry["warnings"])

    forget(entity="*", project="round17-dup")
    forget(entity="*", project="round17-dup-batch")
    forget(entity="*", project="round17-dup-batch-2")
