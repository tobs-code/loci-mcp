"""Regression tests for the twenty-first round of fixes (agent audit).

Covers:
- unknown top-level arguments are refused by the envelope instead of being
  silently ignored (pydantic `extra="ignore"`);
- project/global counts do not mistake the `__forgotten__` sentinel in
  `superseded_by` for a supersede marker;
- `import_memories` clears self-referencing and internal `superseded_by`
  values so the restored rows stay readable history;
- `verify_integrity(fix=true)` re-scans, so the counts and `results` describe
  the state AFTER the repair;
- `supersede_content` finds a global fact from inside a project (cross-scope
  supersede, like `supersede=<id>` already does);
- an entity-scoped `contradictions_report` includes untagged neighbours;
- `update_memory`/`update_batch` near-duplicate warnings suggest an
  executable action (a remember supersede), not a `supersede` flag an update
  call does not have.
"""

import asyncio

import mcp_server.tools.maintenance as maintenance
from mcp_server.app import app
from mcp_server.db import get_conn
from mcp_server.scopes import _mem_table
from mcp_server.server import (
    contradictions_report,
    export_memories,
    forget,
    import_memories,
    init_context,
    list_memories,
    list_projects,
    remember,
    update_batch,
    update_memory,
    verify_integrity,
)


def _call(tool: str, arguments: dict):
    """Call a tool through the MCP tool manager (as a client would)."""
    return asyncio.run(app._tool_manager.call_tool(tool, arguments))


def _fake_nli(contradiction: float = 0.95):
    def _nli(*args, **kwargs):
        documents = kwargs.get("documents") or (
            args[1] if len(args) > 1 else [])
        return [contradiction] * len(documents), None, []

    return _nli


def _sql(query: str, params: tuple = ()) -> None:
    """Run one write statement directly (test seeding)."""
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("SET search_path TO public")
        cur.execute(query, params)
        conn.commit()
    finally:
        conn.close()


def test_unknown_argument_is_refused_instead_of_ignored():
    # `list_memories` has no `only_superseded` parameter (that is a `recall`
    # flag): before the schema check it vanished silently and the caller read
    # a plain browse, very likely as "only superseded".
    result = _call("list_memories", {"only_superseded": True})

    assert result["ok"] is False
    assert result["status"] == "error"
    assert "only_superseded" in result["error"]
    assert "unknown argument" in result["error"].lower()
    # The accepted arguments are named, so the caller can correct on their own.
    assert "include_superseded" in result["error"]

    # A schema-correct call still works: the check must not break real calls.
    stored = remember("Round21 accepts exactly the documented arguments")
    assert stored["ok"] is True


def test_unknown_argument_on_update_path_is_refused_with_envelope():
    result = _call("update_memory", {"memory_id": "x", "only_touch": True})

    assert result["ok"] is False
    assert "only_touch" in result["error"]


def test_forgotten_sentinel_is_not_counted_as_superseded():
    project = "round21-count"
    first = remember("Round21 counting probe one", project=project)["id"]
    remember("Round21 counting probe two", project=project,
             supersede=first)

    stats = init_context(project=project, include_global=False)
    row = next(r for r in stats["project_stats"] if r["project"] == project)
    assert row["superseded_memories"] == 1
    assert row["forgotten_memories"] == 0

    active = list_memories(project=project, include_global=False,
                           fields=["id"], limit=50)["results"]
    second, = [r["id"] for r in active]
    forget(memory_id=second, project=project)

    stats = init_context(project=project, include_global=False)
    row = next(r for r in stats["project_stats"] if r["project"] == project)
    # A soft-forgotten row carries the `__forgotten__` sentinel in
    # `superseded_by`; the superseded count must stay 1 (the real supersede
    # above) instead of 2.
    assert row["forgotten_memories"] == 1
    assert row["superseded_memories"] == 1

    listed = list_projects(include_empty=True)
    scope, = [r for r in listed["projects"] if r["project"] == project]
    assert scope["superseded_memories"] == 1
    assert scope["forgotten_memories"] == 1
    assert scope["total_memories"] == 2


def test_import_clears_self_referencing_superseded_by():
    project = "round21-selfref"
    records = [
        {"id": "selfref0000000001",
         "content": "Round21 self-referencing import probe",
         "type": "fact",
         "project": project,
         "superseded_by": "selfref0000000001"},
        {"id": "sentinel000000001",
         "content": "Round21 sentinel-carrying import probe",
         "type": "fact",
         "project": project,
         "superseded_by": "__forgotten__"},
    ]

    result = import_memories(data={"records": records}, project=project)

    assert result["ok"] is True
    assert result["imported"] == 2
    assert result.get("repaired_self_references") == 2
    assert any("self-referencing" in w for w in result["warnings"])

    active = list_memories(project=project, include_global=False,
                           fields=["id"], limit=50)["results"]
    ids = {row["id"] for row in active}
    assert {"selfref0000000001", "sentinel000000001"} <= ids


def test_existing_supersede_history_survives_import():
    # A record with sane history keeps its supersede link: the normalization
    # only touches self-references and the internal sentinel.
    project = "round21-keep"
    old_id = remember("Round21 keeps valid history",
                      project=project)["id"]
    newer = remember(
        "Round21 keeps valid history older",
        project=project,
        dedupe="off",
        supersede=old_id)
    superseded_ids = newer.get("superseded", [])
    assert superseded_ids == [old_id]

    backup = export_memories(project=project, include_superseded=True)
    record = next(r for r in backup["data"]["records"]
                  if r["id"] in superseded_ids)
    assert record["superseded_by"] == newer["id"]

    other = "round21-keep-target"
    result = import_memories(data={"records": [record]}, project=other)
    assert result["ok"] is True
    assert "repaired_self_references" not in result

    rows = list_memories(project=other, include_global=False,
                         include_superseded=True, limit=50)["results"]
    keep = next(r for r in rows if r["id"] in superseded_ids)
    assert keep["superseded_by"] == newer["id"]


def test_verify_integrity_fix_rescans_after_repair():
    project = "round21-integrity"
    probe = remember("Round21 dangling link probe", project=project)["id"]
    target = remember("Round21 dangling link target",
                      project=project)["id"]

    table = _mem_table(project)
    # Hard-delete the target and plant the dangling link by hand, so `probe`
    # points at an id that no longer exists in any scope.
    _sql(f"UPDATE {table} SET superseded_by = %s WHERE id = %s",
         (target, probe))
    _sql(f"DELETE FROM {table} WHERE id = %s", (target,))
    _sql("UPDATE memories SET superseded_by = NULL WHERE id = %s", (target,))

    before = verify_integrity(project=project)
    assert before["ok"] is True
    assert before["dangling_superseded_by"] >= 1

    fixed = verify_integrity(project=project, fix=True)

    assert fixed["ok"] is True
    assert fixed["fixed"] >= 1
    # After the fix the report describes the post-fix state, so a caller can
    # read it as "is it clean now?".
    assert fixed["dangling_superseded_by"] == 0
    assert fixed["self_references"] == 0
    assert any("cleared" in w for w in fixed["warnings"])
    # The repaired row is readable history again.
    listed = list_memories(project=project, include_global=False,
                           limit=50)["results"]
    assert probe in {row["id"] for row in listed}


def test_cross_scope_supersede_content_matches_global_fact():
    project = "round21-cross"
    global_id = remember(
        "Round21 cross scope: the cache stores JSON blobs")["id"]

    result = remember(
        "Round21 cross scope: the cache stores new content",
        project=project,
        supersede_content="Round21 cross scope: the cache stores JSON blobs")

    assert result["ok"] is True
    assert global_id in result.get("superseded", [])
    # The replacement hides the global fact in every scope, and the response
    # names the cross-scope supersede instead of a silent global deletion.
    assert global_id in result.get("cross_scope_superseded", [])
    assert any("cross-scope" in w for w in result["warnings"])

    # The superseded global fact needs the history flag to be readable.
    direct = list_memories(memory_id=global_id)
    assert direct["ok"] is False
    assert "include_superseded" in str(direct.get("error", ""))

    history = list_memories(memory_id=global_id, include_superseded=True)
    assert history["ok"] is True
    row = history["memory"]
    assert row["superseded_by"] == result["id"]


def test_entity_scan_sees_untagged_conflict(monkeypatch):
    monkeypatch.setattr(maintenance, "nli_detailed", _fake_nli(0.95))
    project = "round21-ent"
    tagged = remember(
        "Round21 relay republishes queue events at 21:00 timer",
        entity="Relay21", project=project, dedupe="off")["id"]
    tagged_b = remember(
        "Round21 relay republishes queue events at 21:00 timer again",
        entity="Relay21", project=project, dedupe="off")["id"]
    # The untagged twin: before the fix an entity-scoped scan hid it and the
    # report looked clean while the conflict was still active.
    untagged = remember(
        "Round21 relay republishes queue events at 21:00 timer not",
        project=project, dedupe="off")["id"]

    report = contradictions_report(project=project, entity="Relay21")

    assert report["ok"] is True
    ids_in_pairs = ({p["a"]["id"] for p in report["results"]}
                    | {p["b"]["id"] for p in report["results"]})
    assert {tagged, tagged_b} <= ids_in_pairs
    assert untagged in ids_in_pairs


def test_entity_scan_with_single_seed_still_compares_untagged(monkeypatch):
    # 1 tagged seed used to trip the "nothing to compare" gate before the
    # untagged neighbours ever got a candidate seat.
    monkeypatch.setattr(maintenance, "nli_detailed", _fake_nli(0.95))
    project = "round21-single"
    tagged = remember(
        "Round21 indexer rebuilds search vectors at noon",
        entity="Indexer21", project=project)["id"]
    untagged = remember(
        "Round21 indexer never rebuilds search vectors at noon",
        project=project)["id"]

    report = contradictions_report(project=project, entity="Indexer21")

    assert report["ok"] is True
    assert not report["warnings"] or all(
        "nothing to compare" not in w for w in report["warnings"])
    ids_in_pairs = ({p["a"]["id"] for p in report["results"]}
                    | {p["b"]["id"] for p in report["results"]})
    assert {tagged, untagged} <= ids_in_pairs
    assert all(p.get("cross_entity") for p in report["results"])


def test_entity_scan_with_lone_scope_still_says_nothing_to_compare():
    # An entity scan of a scope that holds only that entity's single memory
    # has genuinely nothing to compare: the scope-wide gate does not apply,
    # but the empty answer must still say why it is empty.
    project = "round21-lone"
    remember(
        "Round21 hermit cache flushes once per day",
        entity="Hermit21", project=project)

    report = contradictions_report(project=project, entity="Hermit21")

    assert report["ok"] is True
    assert report["count"] == 0
    assert any("nothing to compare" in w for w in report["warnings"])


def test_update_near_duplicate_warning_suggests_actual_action():
    project = "round21-upd"
    target = remember(
        "Round21 worker retries every 3 seconds", entity="W21",
        project=project, dedupe="off")["id"]
    other = remember(
        "Round21 worker retries every 2 seconds", entity="W21",
        project=project, dedupe="off")["id"]

    result = update_memory(
        memory_id=target,
        content="Round21 worker retries every 2 seconds now",
        project=project)

    assert result["ok"] is True
    warnings = result.get("warnings", [])
    # The update path has no `supersede` flag: the advice must point at an
    # action the caller can execute (a remember supersede), not at a flag of
    # the tool that is already running.
    assert any("separate remember call" in w for w in warnings)

    batch = update_batch(
        updates=[{"memory_id": other,
                  "content": "Round21 worker retries every 3 seconds now"}],
        project=project)
    assert batch["ok"] is True
    entry = next(e for e in batch["results"] if e["id"] == other)
    assert any("separate remember call" in w
               for w in entry.get("warnings", []))
