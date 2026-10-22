"""Regression tests for the thirteenth round of fixes.

Covers the audit findings and the features added with them:
- TTL survives export/import (JSON and JSONL);
- expired rows are visible through `include_expired`/`list_memories(memory_id=...)`, but never
  leak into `include_deleted`;
- `metadata_mode` controls the metadata carry-over on a supersede;
- `on_near_duplicate` can warn, supersede or reject;
- `unrelate(keep_context=True)` keeps the edge sentence;
- `find_connections` reports `total`/`has_more` and pages with `offset`;
- export filters and `import_memories(dry_run=True)`;
- TTL updates and bulk restore.
"""

from datetime import datetime, timedelta, timezone

from mcp_server.server import (
    export_memories,
    find_connections,
    forget,
    get_entity,
    import_memories,
    list_memories,
    relate,
    remember,
    unrelate,
    update_batch,
    update_memory,
)

FUTURE = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
PAST = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()


def test_export_import_preserves_ttl():
    remember("The nightly job keeps the cache warm",
             project="ttl-io", expires_at=FUTURE)
    remember("The old cache policy is retired",
             project="ttl-io", expires_at=PAST)
    exported = export_memories(project="ttl-io")
    by_content = {r["content"]: r for r in exported["data"]["records"]}
    assert by_content["The nightly job keeps the cache warm"]["expires_at"]
    assert by_content["The old cache policy is retired"]["expires_at"]

    result = import_memories(data=exported["data"], project="ttl-io-copy")
    assert result["ok"] is True
    assert result["imported"] == 2

    active = list_memories(project="ttl-io-copy")
    assert [r["content"] for r in active["results"]] == [
        "The nightly job keeps the cache warm"]
    expired = list_memories(project="ttl-io-copy", include_expired=True)
    assert len(expired["results"]) == 2
    markers = {r["content"]: r.get("expired") for r in expired["results"]}
    assert markers["The old cache policy is retired"] is True


def test_include_deleted_does_not_show_expired_rows():
    remember("Temporary maintenance window", project="ttl-flags",
             expires_at=PAST)
    forgotten = remember("Forgotten but not expired", project="ttl-flags")
    forget(forgotten["id"], project="ttl-flags", soft=True)

    listed = list_memories(project="ttl-flags", include_deleted=True)
    contents = [r["content"] for r in listed["results"]]
    assert "Forgotten but not expired" in contents
    assert "Temporary maintenance window" not in contents


def test_list_memories_by_id_explains_hidden_expired_row():
    stored = remember("Rotating credential expires tonight",
                      project="ttl-get", expires_at=PAST)
    hidden = list_memories(memory_id=stored["id"], project="ttl-get")
    assert hidden["ok"] is False
    assert "include_expired" in hidden["error"]

    visible = list_memories(memory_id=stored["id"], project="ttl-get",
                         include_expired=True)
    assert visible["ok"] is True
    assert visible["memory"]["expired"] is True
    assert visible["memory"]["expires_at"] is not None


def test_list_memories_memory_id_accepts_star_scope():
    stored = remember("One memory one scope", project="ttl-get-star")
    result = list_memories(memory_id=stored["id"], project="*")
    assert result["ok"] is True
    assert result["memory"]["id"] == stored["id"]
    assert result["memory"]["project"] == "ttl-get-star"


def test_metadata_mode_replace_drops_old_keys():
    old = remember("Payments API runs on port 9090",
                   project="meta-mode", entity="Payments",
                   metadata={"ticket": "OPS-1", "port": 9090})
    replaced = remember("Payments API runs on port 9443",
                        project="meta-mode", entity="Payments",
                        metadata={"port": 9443}, supersede=old["id"],
                        metadata_mode="replace")
    assert replaced["ok"] is True
    row = list_memories(memory_id=replaced["id"], project="meta-mode")["memory"]
    assert row["metadata"] == {"port": 9443}


def test_near_duplicate_reject_and_supersede():
    first = remember("Orders service listens on port 8100",
                     project="near-policy", entity="Orders")
    rejected = remember("Orders service listens on port 8100 today",
                        project="near-policy", entity="Orders",
                        on_near_duplicate="reject")
    assert rejected["status"] == "duplicate"
    assert rejected["duplicate_mode"] == "near"
    assert list_memories(project="near-policy")["count"] == 1

    replaced = remember("Orders service listens on port 8100 nowadays",
                        project="near-policy", entity="Orders",
                        on_near_duplicate="supersede")
    assert replaced["status"] == "stored"
    assert first["id"] in replaced["superseded"]
    assert list_memories(project="near-policy")["count"] == 1


def test_unrelate_keep_context_keeps_the_edge_sentence():
    relate("GC", "S3", "writes_to", project="keep-ctx",
           context="The garbage collector archives reports to S3")
    result = unrelate("GC", "S3", "writes_to", project="keep-ctx",
                      keep_context=True)
    assert result["ok"] is True
    assert result["context_kept"] is True
    assert result["context_memories_removed"] == 0
    assert list_memories(project="keep-ctx", contains="archives reports",
                         include_relation_context=True)["count"] == 1

    entity = get_entity("GC", project="keep-ctx")
    assert entity["relation_context_count"] == 1


def test_find_connections_reports_total_and_pages():
    relate("Hub", "Alpha", "LINKS", project="conn-page")
    relate("Hub", "Beta", "LINKS", project="conn-page")
    relate("Gamma", "Hub", "LINKS", project="conn-page")

    first = find_connections("Hub", project="conn-page", limit=1)
    assert first["total"] == 3
    assert first["count"] == 1
    assert first["has_more"] is True
    second = find_connections("Hub", project="conn-page", limit=1,
                              offset=first["next_offset"])
    assert second["count"] == 1
    beyond = find_connections("Hub", project="conn-page", offset=10)
    assert beyond["count"] == 0
    assert beyond["warnings"]


def test_export_filters_and_import_dry_run():
    remember("Deploy runs from the pipeline", project="export-filter",
             entity="Deploy", mem_type="runbook",
             metadata={"team": "platform"})
    remember("Deploy needs an approval", project="export-filter",
             entity="Deploy", mem_type="convention")

    exported = export_memories(project="export-filter", entity="Deploy",
                               mem_type="runbook",
                               metadata_filter={"team": "platform"})
    assert exported["count"] == 1
    assert exported["data"]["records"][0]["mem_type"] == "runbook"

    dry = import_memories(data=exported["data"], project="export-filter-dry",
                          dry_run=True)
    assert dry["ok"] is True
    assert dry["dry_run"] is True
    assert dry["imported"] == 1
    assert any("dry-run" in w for w in dry["warnings"])
    assert list_memories(project="export-filter-dry")["count"] == 0


def test_update_memory_sets_and_clears_ttl():
    stored = remember("Session tokens rotate", project="ttl-update")
    set_ttl = update_memory(stored["id"], project="ttl-update",
                            expires_at=FUTURE)
    assert set_ttl["ok"] is True
    assert list_memories(memory_id=stored["id"], project="ttl-update")[
        "memory"]["expires_at"] is not None

    cleared = update_memory(stored["id"], project="ttl-update",
                            clear_expires_at=True)
    assert cleared["ok"] is True
    assert list_memories(memory_id=stored["id"], project="ttl-update")[
        "memory"]["expires_at"] is None


def test_update_batch_sets_and_clears_ttl():
    one = remember("Batch TTL one", project="ttl-batch-update")
    two = remember("Batch TTL two", project="ttl-batch-update")

    result = update_batch(updates=[
        {"memory_id": one["id"], "expires_at": FUTURE},
        {"memory_id": two["id"], "expires_at": FUTURE},
    ], project="ttl-batch-update")
    assert result["ok"] is True
    assert result["count"] == 2
    assert all(list_memories(memory_id=m["id"], project="ttl-batch-update")[
        "memory"]["expires_at"] for m in (one, two))

    cleared = update_batch(updates=[
        {"memory_id": one["id"], "clear_expires_at": True},
        {"memory_id": two["id"], "clear_expires_at": True},
    ], project="ttl-batch-update")
    assert cleared["ok"] is True
    assert all(list_memories(memory_id=m["id"], project="ttl-batch-update")[
        "memory"]["expires_at"] is None for m in (one, two))


def test_jsonl_export_keeps_ttl():
    remember("JSONL TTL fact", project="ttl-jsonl", expires_at=FUTURE)
    exported = export_memories(format="jsonl", project="ttl-jsonl")
    assert exported["count"] == 1
    assert FUTURE[:10] in exported["data"]


def test_bulk_restore():
    one = remember("Bulk restore one", project="bulk-restore")
    two = remember("Bulk restore two", project="bulk-restore")
    forget(memory_ids=[one["id"], two["id"]], project="bulk-restore")

    result = update_batch(
        [{"memory_id": one["id"], "restore": True},
         {"memory_id": two["id"], "restore": True}],
        project="bulk-restore")
    assert result["ok"] is True
    assert result["succeeded"] == 2
    assert {r["status"] for r in result["results"]} == {"restored"}
    assert {r["content"] for r in
            list_memories(project="bulk-restore")["results"]} == {
                "Bulk restore one", "Bulk restore two"}
