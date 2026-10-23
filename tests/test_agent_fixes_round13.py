"""Regression tests for the fourteenth round of fixes.

Covers:
- `recall` can return forgotten and expired rows on request;
- `graph_boost` warns when the entity has no graph node;
- hidden rows explain why they are hidden in `update_memory`;
- batch answers distinguish entries from successes;
- `forget(entity=...)` reports an already-forgotten entity;
- `verify_integrity` finds and repairs dangling supersede links;
- the graph export honours the `entity` filter;
- JSONL warns that `include_graph` is ignored;
- `import_memories(dry_run=True)` names what it would create;
- `remember_batch` carries metadata over a supersede;
- a TTL in the past is accepted but warned about;
- expired rows do not leak through `include_superseded`;
- `list_entities` count matches total.
"""

from datetime import datetime, timedelta, timezone

from mcp_server.db import get_conn
from mcp_server.scopes import _mem_table
from mcp_server.server import (
    export_memories,
    forget,
    get_entity,
    import_memories,
    list_entities,
    list_memories,
    recall,
    relate,
    remember,
    remember_batch,
    update_batch,
    update_memory,
    verify_integrity,
)

PAST = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
FUTURE = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()


def _exec(query, params=None):
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(query, params)
        conn.commit()
    finally:
        conn.close()


def test_recall_can_return_forgotten_and_expired_rows():
    forgotten = remember("The staging cluster listens on port 5433",
                         project="hidden-recall")
    remember("The staging backup window starts at 02:00",
             project="hidden-recall", expires_at=PAST)
    forget(forgotten["id"], project="hidden-recall", soft=True)

    active = recall("staging cluster port", project="hidden-recall")
    assert active["results"] == []

    shown = recall("staging cluster port", project="hidden-recall",
                   include_deleted=True)
    assert any(r["id"] == forgotten["id"] for r in shown["results"])
    assert any(r.get("forgotten") is True for r in shown["results"])

    expired = recall("staging backup window", project="hidden-recall",
                     include_expired=True)
    assert any(r.get("expired") is True for r in expired["results"])
    assert any("expired TTL" in w for w in expired["warnings"])


def test_graph_boost_warns_when_entity_has_no_node():
    remember("The ghost service runs nowhere", project="boost-warn",
             entity="GhostService")
    result = recall("ghost service", project="boost-warn",
                    entity="GhostService", graph_boost=True)
    assert result["ok"] is True
    assert any("no graph node" in w for w in result["warnings"])


def test_update_and_touch_explain_hidden_state():
    gone = remember("The retired queue name is legacy", project="hidden-state")
    forget(gone["id"], project="hidden-state", soft=True)

    updated = update_memory(gone["id"], project="hidden-state",
                            importance=0.5)
    assert updated["ok"] is False
    assert "soft-forgotten" in updated["error"]
    assert "restore=true" in updated["error"]

    touched = update_memory(gone["id"], project="hidden-state", touch=True)
    assert touched["ok"] is False
    assert "soft-forgotten" in touched["error"]

    old = remember("The primary region is eu-west-1", project="hidden-state")
    remember("The primary region is us-east-1", project="hidden-state",
             supersede=old["id"])
    replaced = update_memory(old["id"], project="hidden-state",
                             importance=0.5)
    assert replaced["ok"] is False
    assert "superseded by" in replaced["error"]


def test_batch_answers_split_entries_and_successes():
    stored = remember_batch(
        [{"content": "Batch alpha is enabled"},
         {"content": "Batch beta is enabled"},
         {"content": ""}],
        project="batch-counts")
    assert stored["count"] == 3
    assert stored["succeeded"] == 2
    assert stored["failed"] == 1

    updated = update_batch(
        [{"memory_id": stored["results"][0]["id"], "importance": 0.4},
         {"memory_id": "does-not-exist", "importance": 0.4}],
        project="batch-counts")
    assert updated["status"] == "partial"
    assert updated["count"] == 2
    assert updated["succeeded"] == 1
    assert updated["failed"] == 1


def test_forget_already_forgotten_entity_is_not_not_found():
    fact = remember("The archive bucket is cold storage",
                    project="forget-entity", entity="ArchiveBucket")
    forget(fact["id"], project="forget-entity", soft=True)

    again = forget(entity="ArchiveBucket", project="forget-entity")
    assert again["ok"] is True
    assert again["status"] == "ok"
    assert "already soft-forgotten" in again["message"]


def test_verify_integrity_finds_and_repairs_dangling_link():
    fact = remember("The deploy target is blue", project="integrity")
    _exec(
        f"UPDATE {_mem_table('integrity')} SET superseded_by = %s "
        f"WHERE id = %s", ("missingreplacement", fact["id"]))

    assert list_memories(memory_id=fact["id"], project="integrity")["ok"] is False

    report = verify_integrity(project="integrity")
    assert report["ok"] is True
    assert report["dangling_superseded_by"] == 1
    assert report["fixed"] == 0
    assert report["results"][0]["problem"] == "dangling_superseded_by"

    repaired = verify_integrity(project="integrity", fix=True)
    assert repaired["fixed"] == 1
    assert list_memories(memory_id=fact["id"], project="integrity")["ok"] is True


def test_export_entity_filter_narrows_the_graph():
    relate("BillingService", "PostgresDB", "uses", project="export-graph",
           context="BillingService writes to PostgresDB")
    relate("ReportingService", "PostgresDB", "uses", project="export-graph",
           context="ReportingService reads PostgresDB")

    exported = export_memories(project="export-graph", entity="BillingService")
    names = {e["name"] for e in exported["data"]["graph"]["entities"]}
    # The incident edge stays in the backup, so both of its endpoints must be
    # present; otherwise the exported graph is incomplete.
    assert names == {"BillingService", "PostgresDB"}
    edges = exported["data"]["graph"]["relations"]
    assert {(e["from_entity"], e["to_entity"]) for e in edges} == {
        ("BillingService", "PostgresDB")}


def test_jsonl_export_warns_that_the_graph_is_ignored():
    remember("The JSONL backup has no graph", project="jsonl-graph")
    exported = export_memories(project="jsonl-graph", format="jsonl")
    assert exported["graph_included"] is False
    assert any("has no graph" in w for w in exported["warnings"])


def test_import_dry_run_names_would_create():
    remember("The dry run counts this fact", project="dry-import")
    exported = export_memories(project="dry-import")
    preview = import_memories(data=exported["data"], project="dry-import-copy",
                              dry_run=True)
    assert preview["ok"] is True
    assert preview["would_create"] == 1
    assert preview["imported"] == 1
    assert list_memories(project="dry-import-copy")["results"] == []


def test_batch_supersede_carries_metadata():
    first = remember("The ingest pipeline uses two workers",
                     project="batch-meta",
                     metadata={"ticket": "OPS-42", "owner": "data"})
    stored = remember_batch(
        [{"content": "The ingest pipeline uses four workers",
          "supersede": first["id"]}],
        project="batch-meta")
    assert stored["succeeded"] == 1
    new = list_memories(memory_id=stored["results"][0]["id"], project="batch-meta")
    assert new["memory"]["metadata"]["ticket"] == "OPS-42"
    assert new["memory"]["metadata"]["owner"] == "data"


def test_past_ttl_is_stored_but_warned():
    stored = remember("The maintenance window already passed",
                      project="ttl-warn", expires_at=PAST)
    assert stored["ok"] is True
    assert any("immediately inactive" in w for w in stored["warnings"])
    assert list_memories(project="ttl-warn")["results"] == []
    assert len(list_memories(project="ttl-warn",
                             include_expired=True)["results"]) == 1


def test_expired_rows_do_not_leak_through_include_superseded():
    remember("The expired build cache is off", project="expired-leak",
             entity="ExpiredCache", expires_at=PAST)
    remember("The active build cache is on", project="expired-leak",
             entity="ExpiredCache")
    result = get_entity("ExpiredCache", project="expired-leak",
                        include_superseded=True)
    contents = [m["content"] for m in result["memories"]]
    assert contents == ["The active build cache is on"]


def test_list_entities_count_matches_total():
    remember("Only expired memory for this entity", project="entity-count",
             entity="ExpiredOnly", expires_at=PAST)
    remember("Active memory for this entity", project="entity-count",
             entity="ActiveOnly")
    listed = list_entities(project="entity-count")
    assert listed["count"] == listed["total"]
    names = [row["name"] for row in listed["results"]]
    assert "ExpiredOnly" not in names
    assert "ActiveOnly" in names
