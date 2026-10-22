"""Regression tests for the ninth round of fixes (audit follow-up).

Covers:
- a stored write that is still very similar to an active memory returns a
  `near-duplicate` warning (and the semantic dedupe floor is 0.85 now);
- a project write that invalidates a shared (global) fact reports a
  `cross-scope supersede` warning and `cross_scope_superseded`;
- `import_memories` names why records were skipped;
- `contradictions_report` exposes `nli_failures` (which pairs were not
  judged) instead of only a count;
- `update_batch` (savepoints per entry) and `update_memory(touch=true)` (decay refresh).
"""

from mcp_server.server import (
    contradictions_report,
    import_memories,
    list_memories,
    remember,
    remember_batch,
    update_batch,
    update_memory,
)


# --------------------------------------------------------------------------
# Near-duplicate warning + the semantic dedupe floor
# --------------------------------------------------------------------------

def test_remember_warns_about_a_near_duplicate_but_stores_it():
    first = remember("Near-dup probe: the invoice is paid monthly")["id"]

    result = remember("Near-dup probe: the invoice is paid every month")

    assert result["status"] == "stored"
    assert any("near-duplicate" in w and first in w
               for w in result["warnings"])


def test_remember_does_not_warn_for_unrelated_content():
    remember("Near-dup probe: the invoice is paid monthly")

    result = remember("Zebra migration: the herd moved north in spring")

    assert result["status"] == "stored"
    assert result["warnings"] == []


def test_dedupe_off_suppresses_the_near_duplicate_warning():
    remember("Near-dup off probe: the invoice is paid monthly")

    result = remember("Near-dup off probe: the invoice is paid monthly.",
                      dedupe="off")

    assert result["status"] == "stored"
    assert result["warnings"] == []


def test_semantic_dedupe_catches_a_rewrite_at_the_new_floor():
    # Fake-embedding cosine of these two texts is 0.875: below the old 0.9
    # floor (stored as a new fact) and above the new 0.85 one (duplicate).
    first = remember("Dedupe floor probe: invoice paid monthly in euros")["id"]

    result = remember("Dedupe floor probe: invoice paid monthly in dollars",
                      dedupe="semantic")

    assert result["status"] == "duplicate"
    assert result["id"] == first
    assert result["duplicate_mode"] == "semantic"


def test_remember_batch_warns_about_a_near_duplicate_per_entry():
    existing = remember("Batch near probe: the report is weekly")["id"]

    result = remember_batch([
        {"content": "Batch near probe: the report is monthly"},
    ])

    entry = result["results"][0]
    assert entry["status"] == "stored"
    assert any("near-duplicate" in w and existing in w
               for w in entry["warnings"])


# --------------------------------------------------------------------------
# Cross-scope supersede
# --------------------------------------------------------------------------

def test_project_write_warns_when_it_invalidates_a_global_fact():
    global_id = remember(
        "Cross-scope probe: the relay does not support TLS",
        scope="global")["id"]

    result = remember("Cross-scope probe: the relay supports TLS",
                      project="cross-scope", check_contradictions=True)

    assert result["ok"] is True
    assert result["status"] == "stored"
    assert result["cross_scope_superseded"] == [global_id]
    assert any("cross-scope supersede" in w for w in result["warnings"])
    # The shared fact is gone for every scope, so a plain global read no
    # longer returns it (that is exactly what the warning announces).
    active = list_memories(project="global", limit=20)["results"]
    assert all(row["id"] != global_id for row in active)


def test_explicit_supersede_of_a_global_fact_is_reported_as_cross_scope():
    global_id = remember("Explicit cross probe: retention is 30 days",
                         scope="global")["id"]

    result = remember("Explicit cross probe: retention is 90 days",
                      project="cross-explicit", supersede=global_id)

    assert result["superseded"] == [global_id]
    assert result["cross_scope_superseded"] == [global_id]
    assert any("cross-scope supersede" in w for w in result["warnings"])


def test_remember_batch_reports_a_cross_scope_supersede_per_entry():
    global_id = remember("Batch cross probe: the cache does not persist",
                         scope="global")["id"]

    result = remember_batch([
        {"content": "Batch cross probe: the cache persists"},
    ], project="batch-cross", check_contradictions=True)

    entry = result["results"][0]
    assert entry["cross_scope_superseded"] == [global_id]
    assert any("cross-scope" in w for w in entry["warnings"])


# --------------------------------------------------------------------------
# import_memories names why it skipped records
# --------------------------------------------------------------------------

def test_reimport_warns_about_skipped_ids():
    payload = {"records": [
        {"id": "aaaaaaaaaaaaaaa1", "content": "Reimport probe one"},
        {"id": "aaaaaaaaaaaaaaa2", "content": "Reimport probe two"},
    ]}
    first = import_memories(payload)
    assert first["imported"] == 2
    assert first["skipped"] == 0
    assert not any("import skipped" in w for w in first["warnings"])

    second = import_memories(payload)

    assert second["imported"] == 0
    assert second["skipped"] == 2
    assert second["skipped_reasons"]["id_exists"] == 2
    assert any("import skipped" in w and "already existed" in w
               for w in second["warnings"])


def test_import_warns_about_a_skipped_content_duplicate():
    remember("Import dup probe: the invoice is paid monthly")

    result = import_memories({"records": [
        {"content": "Import dup probe: the invoice is paid monthly"},
    ]})

    assert result["skipped"] == 1
    assert result["skipped_reasons"]["duplicate_content"] == 1
    assert any("skip_duplicates=false" in w for w in result["warnings"])


# --------------------------------------------------------------------------
# Structured NLI failures in contradictions_report
# --------------------------------------------------------------------------

def test_contradictions_report_names_the_pairs_nli_could_not_judge(monkeypatch):
    a = remember("Nli failure probe: the relay answers on port 8000",
                 entity="FailureRelay")["id"]
    b = remember("Nli failure probe: the relay answers on port 8100",
                 entity="FailureRelay")["id"]

    def fake_nli(query, documents, *args, **kwargs):
        return ([0.0] * len(documents),
                f"NLI unavailable for {len(documents)} of "
                f"{len(documents)} document(s): boom",
                list(range(len(documents))))

    monkeypatch.setattr("mcp_server.tools.maintenance.nli_detailed",
                        fake_nli)

    report = contradictions_report(entity="FailureRelay")

    assert report["ok"] is True
    failures = report["nli_failures"]
    assert failures
    entry = failures[0]
    assert entry["fact_id"] in {a, b}
    assert set(entry["neighbor_ids"]) <= {a, b}
    assert entry["reason"] == "boom"
    assert any("scan partially incomplete" in w for w in report["warnings"])


# --------------------------------------------------------------------------
# update_batch
# --------------------------------------------------------------------------

def test_update_batch_updates_several_memories_and_reports_each():
    one = remember("Batch update probe one")["id"]
    two = remember("Batch update probe two")["id"]

    result = update_batch([
        {"memory_id": one, "importance": 0.3},
        {"memory_id": two, "content": "Batch update probe two (revised)"},
    ])

    assert result["ok"] is True
    assert result["count"] == 2
    assert [r["status"] for r in result["results"]] == ["updated", "updated"]
    row_one = list_memories(memory_id=one, fields=["importance"])["results"][0]
    row_two = list_memories(memory_id=two, fields=["content"])["results"][0]
    assert row_one["importance"] == 0.3
    assert row_two["content"] == "Batch update probe two (revised)"


def test_update_batch_keeps_going_after_a_bad_entry():
    good = remember("Batch update survivor")["id"]

    result = update_batch([
        {"memory_id": "zzunknownid00", "importance": 0.5},
        {"memory_id": good, "importance": 0.2},
        {"importance": 0.1},
    ])

    assert result["ok"] is True
    assert result["status"] == "partial"
    assert [r["status"] for r in result["results"]] == [
        "not_found", "updated", "error"]
    assert "memory_id is required" in result["results"][2]["error"]


def test_update_batch_fails_when_nothing_could_be_updated():
    result = update_batch([{"memory_id": "zzunknownid01", "importance": 0.5}])

    assert result["ok"] is False
    assert result["status"] == "error"
    assert result["results"][0]["status"] == "not_found"


def test_update_batch_refuses_an_ambiguous_id_per_entry():
    memory_id = remember("Ambiguous batch probe", project="amb-a")["id"]
    import_memories({"records": [
        {"id": memory_id, "content": "Ambiguous batch probe other"},
    ]}, project="amb-b")

    result = update_batch([{"memory_id": memory_id, "importance": 0.4}])

    entry = result["results"][0]
    assert entry["status"] == "error"
    assert set(entry["scopes"]) == {"amb-a", "amb-b"}


# --------------------------------------------------------------------------
# update_memory(touch=True)
# --------------------------------------------------------------------------

def test_update_memory_touch_resets_the_decay_age():
    stored = remember("Touch probe: the lease renews on monday")
    before = list_memories(memory_id=stored["id"],
                           fields=["updated_at"])["results"][0]["updated_at"]

    result = update_memory(stored["id"], touch=True)

    assert result["ok"] is True
    assert result["status"] == "updated"
    assert result["project"] == "global"
    after = list_memories(memory_id=stored["id"],
                          fields=["updated_at"])["results"][0]["updated_at"]
    assert after > before


def test_update_memory_touch_can_set_importance_and_leaves_content_alone():
    stored = remember("Touch importance probe")

    assert update_memory(stored["id"], importance=0.8, touch=True)["ok"] is True

    row = list_memories(memory_id=stored["id"],
                        fields=["importance", "content"])["results"][0]
    assert row["importance"] == 0.8
    assert row["content"] == "Touch importance probe"


def test_update_memory_touch_reports_an_unknown_id_as_not_found():
    result = update_memory("zzunknownid02", touch=True)

    assert result["ok"] is False
    assert result["status"] == "not_found"
    assert "NOT FOUND" in result["error"]


def test_update_memory_touch_reports_the_project_scope():
    remember("Touch project probe", project="touch-project")
    row = list_memories(project="touch-project",
                        contains="Touch project probe")["results"][0]

    result = update_memory(row["id"], touch=True)

    assert result["project"] == "touch-project"
