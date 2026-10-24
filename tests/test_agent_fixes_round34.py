"""Regression tests for the round-34 live-audit fixes.

1. `remember_batch` duplicate entries name the blocking row in a dedicated
   `duplicate_of` key instead of only reusing the existing id; `remember`
   stays symmetric for exact duplicates.
2. A kept contradiction warns with an executable `supersede='<id>'` advice
   in both `remember` and `remember_batch`.
3. `import_memories` names the supersede links whose target lives in another
   scope (`cross_scope_supersede_links` + warning) instead of silently
   keeping them, and a dry run flags links that the real import would clear.
"""
from mcp_server.server import (
    export_memories,
    import_memories,
    list_memories,
    remember,
    remember_batch,
)


def test_batch_duplicate_entry_names_duplicate_of():
    project = "round34-dup-entry"
    first = remember("Round34 duplicate-entry probe content",
                     project=project)
    assert first["ok"] is True

    batch = remember_batch(facts=[
        {"content": "Round34 duplicate-entry probe content",
         "project": project},
    ])

    entry = batch["results"][0]
    assert entry["status"] == "duplicate"
    assert entry["duplicate_mode"] == "exact"
    # `id` already points at the existing row; the explicit key keeps the
    # entry shape symmetric with the near-duplicate path.
    assert entry["duplicate_of"] == first["id"]
    assert entry["id"] == first["id"]


def test_remember_duplicate_reports_duplicate_of():
    project = "round34-dup-single"
    first = remember("Round34 remember duplicate probe content",
                     project=project)
    dup = remember("Round34 remember duplicate probe content",
                   project=project)
    assert dup["status"] == "duplicate"
    assert dup["duplicate_of"] == first["id"]
    assert dup["duplicate_mode"] == "exact"


def test_remember_kept_contradiction_warning_is_executable():
    project = "round34-kept-single"
    remember("Round34 the relay does not use port 8081",
             entity="Round34Relay", project=project)
    remember("Round34 the relay never records port 8081 audits",
             entity="Round34Relay", project=project)

    # `warn` keeps the contradicting memories active; the warning must offer
    # a copy-paste-able supersede directive for one of them.
    result = remember("Round34 the relay uses port 8081",
                      entity="Round34Relay", project=project,
                      check_contradictions=True, contradiction_mode="warn")
    kept = result["contradictions"]["kept"]
    assert kept
    kept_ids = [c["existing_id"] for c in kept]
    assert any(f"supersede='{kept_ids[0]}'" in w for w in result["warnings"])


def test_batch_kept_contradiction_warning_is_executable():
    project = "round34-kept-batch"
    remember("Round34 batch relay does not use port 7071",
             entity="Round34RelayB", project=project)
    remember("Round34 batch relay never records port 7071 audits",
             entity="Round34RelayB", project=project)

    result = remember_batch(facts=[
        {"content": "Round34 batch relay uses port 7071",
         "entity": "Round34RelayB", "project": project},
    ], check_contradictions=True, max_contradiction_supersedes=0)

    entry = result["results"][0]
    assert entry["status"] == "stored"
    kept = entry["contradictions"]["kept"]
    assert kept
    kept_ids = [c["existing_id"] for c in kept]
    assert any(f"supersede='{kept_ids[0]}'" in w
               for w in entry["warnings"])


def test_import_names_kept_cross_scope_supersede_links():
    # The old fact lives in project A, its replacement is NOT re-imported:
    # the restored link is valid (the target exists in the still-live scope
    # A) but invisible inside the restored scope B unless the import says so.
    project_a = "round34-xscope-source"
    old_id = remember("Round34 cross-scope history fact",
                      project=project_a)["id"]
    newer = remember("Round34 cross-scope history fact newer",
                     project=project_a,
                     dedupe="off",
                     supersede=old_id)
    newer_id = newer["id"]
    exported = export_memories(project=project_a, include_superseded=True)
    record = next(r for r in exported["data"]["records"]
                  if r["id"] == old_id)
    assert record["superseded_by"] == newer_id

    project_b = "round34-xscope-target"
    result = import_memories(data={"records": [record]}, project=project_b)
    assert result["ok"] is True
    assert result["imported"] == 1
    assert result["cross_scope_supersede_links"] == 1
    assert any("another scope" in w and "cross-scope supersedes are valid"
               in w for w in result["warnings"])

    # The link is kept (not repaired): the history row stays explicit.
    rows = list_memories(project=project_b, include_global=False,
                         include_superseded=True, limit=50)["results"]
    restored = next(r for r in rows if r["id"] == old_id)
    assert restored["superseded_by"] == newer_id


def test_import_dry_run_flags_lost_supersede_links():
    # The target id exists nowhere in the database: the real import clears
    # it, and the dry run must say so instead of looking like a clean run.
    project = "round34-lost-link"
    record = [{"id": "round34lost0000001",
               "content": "Round34 lost-link history record",
               "type": "fact",
               "project": project,
               "superseded_by": "round34target9000001"}]
    dry = import_memories(data={"records": record}, project=project,
                          dry_run=True)
    assert dry["ok"] is True
    assert dry["would_create"] == 1
    assert any("not stored anywhere" in w for w in dry["warnings"])

    result = import_memories(data={"records": record}, project=project)
    assert result["ok"] is True
    assert result["repaired_superseded_links"] == 1
