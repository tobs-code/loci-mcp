"""Round-47 correctness fixes for inactive targets, explicit scopes and
cross-scope contradiction resolution.

Bug A: an inactive (soft-forgotten/superseded/expired) id is not a valid
`supersede` target. Writing a real supersede id over the forgotten sentinel
made a later restore claim success while the row stayed invalidated.

Bug B: `forget(entity=..., project="global")` must stay in the shared scope;
the reserved name normalizes to None and used to widen the graph delete and
the memory loop to every project.

Bug C: an explicit `project="global"` must stay a bounded lookup in
`update_memory`/`update_batch`/restore, not turn into "every scope".

Bug D: `contradictions_report(resolve="newest")` must not update a same-id
copy in another scope without closing `valid_to`, and cross-scope pairs are
reported as non-resolvable instead of silently invalidating one scope.
"""

from mcp_server.config import SENTINEL_FORGOTTEN
from mcp_server.db import get_conn
from mcp_server.server import (
    contradictions_report,
    export_memories,
    forget,
    import_memories,
    relate,
    remember,
    remember_batch,
    update_batch,
    update_memory,
)
from mcp_server.scope_catalog import _mem_table
from mcp_server.store import MemoryStore


def _row(project, memory_id):
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(f"SELECT * FROM {_mem_table(project)} WHERE id = %s",
                    (memory_id,))
        row = cur.fetchone()
        if row is None:
            return None
        return dict(zip([desc[0] for desc in cur.description], row))
    finally:
        conn.close()


def _clone_scope(project):
    exported = export_memories(project=project, include_graph=False)
    assert exported["ok"] is True, exported.get("error")
    payload = dict(exported["data"])
    payload["records"] = [dict(record, project="global")
                          for record in payload["records"]]
    imported = import_memories(data=payload, format="json")
    assert imported["ok"] is True, imported.get("error")


def test_supersede_skips_an_inactive_target_and_keeps_it_restorable():
    project = "round47-supersede"
    old = remember("Round47 retired queue is legacy", project=project)
    assert forget(old["id"], project=project)["ok"] is True

    before = _row(project, old["id"])
    assert before["deleted_at"] is not None
    assert before["superseded_by"] == SENTINEL_FORGOTTEN

    replacement = remember("Round47 retired queue is replaced",
                           project=project, supersede=old["id"],
                           dedupe="off")
    assert replacement["ok"] is True, replacement.get("error")
    assert replacement["superseded"] == []
    assert any("no active memory with id" in w
               for w in replacement["warnings"])

    after = _row(project, old["id"])
    assert after["superseded_by"] == SENTINEL_FORGOTTEN
    assert after["deleted_at"] is not None

    restored = update_memory(old["id"], project=project, restore=True)
    assert restored["ok"] is True, restored.get("error")
    revived = _row(project, old["id"])
    assert revived["deleted_at"] is None
    assert revived["superseded_by"] is None


def test_supersede_guard_also_covers_duplicate_and_batch_paths():
    project = "round47-supersede-other-paths"
    inactive = remember("Round47 inactive batch target", project=project)
    assert forget(inactive["id"], project=project)["ok"] is True
    remember("Round47 active duplicate body", project=project)

    duplicate = remember("Round47 active duplicate body", project=project,
                         supersede=inactive["id"])
    assert duplicate["ok"] is True, duplicate.get("error")
    assert duplicate["status"] == "duplicate"
    assert any("no active memory with id" in w
               for w in duplicate["warnings"])

    batch_target = remember("Round47 inactive batch target two",
                            project=project)
    assert forget(batch_target["id"], project=project)["ok"] is True
    batch = remember_batch([{
        "content": "Round47 batch replacement fact",
        "supersede": batch_target["id"],
    }], project=project)
    assert batch["ok"] is True, batch.get("error")
    assert batch["succeeded"] == 1

    for target in (inactive, batch_target):
        row = _row(project, target["id"])
        assert row["superseded_by"] == SENTINEL_FORGOTTEN
        assert row["deleted_at"] is not None


def test_store_supersede_helpers_skip_inactive_rows():
    project = "round47-helper-guard"
    target = remember("Round47 helper guard target", project=project)
    assert forget(target["id"], project=project)["ok"] is True

    db = MemoryStore()
    try:
        assert db.mark_superseded(target["id"], "replacement",
                                  project=project,
                                  target="project") == 0
        assert db.supersede_many([target["id"]], "replacement",
                                 project=project) == 0
        db.commit()
    finally:
        db.close()

    row = _row(project, target["id"])
    assert row["superseded_by"] == SENTINEL_FORGOTTEN
    assert row["deleted_at"] is not None


def test_forget_entity_global_is_bounded_to_the_shared_scope():
    project = "round47-entity-scope"
    entity = "Round47SharedEntity"
    global_fact = remember("Round47 shared entity global fact", entity=entity,
                           scope="global")
    project_fact = remember("Round47 shared entity project fact", entity=entity,
                            project=project)
    relate(from_entity=entity, to_entity="Round47GlobalPeer",
           relation_type="USES")
    relate(from_entity=entity, to_entity="Round47ProjectPeer",
           relation_type="USES", project=project)

    result = forget(entity=entity, project="global")
    assert result["ok"] is True, result.get("error")
    assert result["counts"]["entities"] == 1
    assert result["counts"]["relations"] == 1

    assert _row("global", global_fact["id"])["deleted_at"] is not None
    assert _row(project, project_fact["id"])["deleted_at"] is None


def test_update_memory_and_batch_treat_global_as_an_explicit_scope():
    project = "round47-update-global"
    stored = remember("Round47 scoped update probe", project=project)
    _clone_scope(project)

    updated = update_memory(stored["id"], project="global", importance=0.91)
    assert updated["ok"] is True, updated.get("error")
    assert updated["project"] == "global"
    assert _row("global", stored["id"])["importance"] == 0.91
    assert _row(project, stored["id"])["importance"] == 0.5

    batched = update_batch(
        [{"memory_id": stored["id"], "importance": 0.77}],
        project="global")
    assert batched["ok"] is True, batched.get("error")
    assert batched["results"][0]["status"] == "updated"
    assert _row("global", stored["id"])["importance"] == 0.77
    assert _row(project, stored["id"])["importance"] == 0.5


def test_explicit_global_does_not_fall_back_to_a_project_copy():
    project = "round47-global-missing"
    stored = remember("Round47 global bound missing probe", project=project)

    updated = update_memory(stored["id"], project="global", importance=0.93)
    assert updated["ok"] is False
    assert updated["status"] == "not_found"
    assert _row(project, stored["id"])["importance"] == 0.5

    batched = update_batch(
        [{"memory_id": stored["id"], "importance": 0.93}],
        project="global")
    assert batched["ok"] is False
    assert batched["results"][0]["status"] == "not_found"
    assert _row(project, stored["id"])["importance"] == 0.5


def test_batch_global_scope_binds_errors_and_keeps_other_entries():
    project = "round47-global-batch"
    shared = remember("Round47 shared batch success", scope="global")
    only_project = remember("Round47 project-only batch entry",
                            project=project)

    result = update_batch([
        {"memory_id": only_project["id"], "importance": 0.88},
        {"memory_id": shared["id"], "importance": 0.66},
    ], project="global")

    assert result["status"] == "partial", result
    assert [entry["status"] for entry in result["results"]] == [
        "not_found", "updated"]
    assert _row(project, only_project["id"])["importance"] == 0.5
    assert _row("global", shared["id"])["importance"] == 0.66


def test_restore_global_fallback_never_revives_a_project_row():
    project = "round47-global-restore-missing"
    stored = remember("Round47 project-only restore probe", project=project)
    assert forget(stored["id"], project=project)["ok"] is True

    restored = update_memory(stored["id"], project="global", restore=True)
    assert restored["ok"] is False
    assert restored["status"] == "not_found"
    assert _row(project, stored["id"])["deleted_at"] is not None


def test_restore_with_explicit_global_does_not_need_an_unambiguous_id():
    project = "round47-restore-global"
    stored = remember("Round47 scoped restore probe", project=project)
    _clone_scope(project)
    forget(memory_ids=[stored["id"]], project=project)
    forget(memory_ids=[stored["id"]], project="global")

    restored = update_memory(stored["id"], project="global", restore=True)
    assert restored["ok"] is True, restored.get("error")
    assert _row("global", stored["id"])["deleted_at"] is None
    assert _row(project, stored["id"])["deleted_at"] is not None


def test_contradiction_resolution_updates_only_the_pairs_scope_and_ttl():
    project = "round47-resolve-scope"
    entity = "Round47ResolveRelay"
    old = remember("Round47 resolve relay answers on port 7410",
                   project=project, entity=entity)
    newer = remember("Round47 resolve relay does not answer on port 7410",
                     project=project, entity=entity, dedupe="off")
    _clone_scope(project)

    report = contradictions_report(entity=entity, project=project,
                                   resolve="newest")
    assert report["ok"] is True, report.get("error")
    assert report["resolved"] == 1, report

    project_old = _row(project, old["id"])
    global_old = _row("global", old["id"])
    assert project_old["superseded_by"] == newer["id"]
    assert project_old["valid_to"] is not None
    assert global_old["superseded_by"] is None
    assert global_old["valid_to"] is None


def test_cross_scope_contradictions_are_flagged_and_not_resolved():
    project = "round47-resolve-cross"
    entity = "Round47CrossRelay"
    global_fact = remember(
        "Round47 cross relay answers on port 7420 in production",
        scope="global", entity=entity)
    project_fact = remember(
        "Round47 cross relay does not answer on port 7420 in production",
        project=project, entity=entity, dedupe="off")

    report = contradictions_report(entity=entity, project=project,
                                   resolve="newest")
    assert report["ok"] is True, report.get("error")
    pair = report["results"][0]
    assert {pair["a"]["id"], pair["b"]["id"]} == {
        global_fact["id"], project_fact["id"]}
    assert pair["cross_scope"] is True
    assert pair.get("resolution") is None
    assert report["resolved"] == 0
    assert any("cross-scope" in w for w in report["warnings"])

    assert _row("global", global_fact["id"])["superseded_by"] is None
    assert _row(project, project_fact["id"])["superseded_by"] is None
