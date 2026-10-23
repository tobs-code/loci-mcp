from mcp_server.server import (
    contradictions_report,
    forget,
    recall,
    remember,
    update_memory,
)


def test_forget_soft_hides_the_fact_but_keeps_its_history():
    memory_id = remember("The secret token is abc")["id"]

    result = forget(memory_id)
    assert result["ok"] is True
    assert result["status"] == "soft_forgotten"
    assert result["counts"]["memories"] == 1
    assert recall("secret token")["results"] == []

    history = recall("secret token", include_superseded=True)
    assert all(r["id"] != memory_id for r in history["results"])

    missing = forget("missing-id")
    assert missing["ok"] is False
    assert missing["status"] == "not_found"
    assert missing["error"] == "NOT FOUND: missing-id"


def test_forget_hard_deletes_the_fact():
    memory_id = remember("A fact that will be purged")["id"]

    result = forget(memory_id, soft=False)
    assert result["status"] == "deleted"
    assert recall("purged", include_superseded=True)["results"] == []


def test_forget_entity_only_touches_the_selected_scope():
    global_id = remember("Bob likes tea", entity="Bob")["id"]
    project_id = remember("Bob works on alpha", entity="Bob", project="alpha")["id"]

    result = forget(entity_name="Bob", project="alpha")

    assert result["ok"] is True
    assert result["status"] == "soft_forgotten"
    project_hits = recall("Bob", project="alpha", limit=10)["results"]
    assert any(r["id"] == global_id for r in project_hits)
    assert all(r["id"] != project_id for r in project_hits)

    history = recall("Bob", project="alpha", include_superseded=True, limit=10)["results"]
    assert any(r["id"] == global_id for r in history)
    assert all(r["id"] != project_id for r in history)


def test_supersede_hides_the_old_fact_from_normal_recall():
    old_id = remember("The server runs on port 8000")["id"]
    new_id = remember("The server runs on port 9000", supersede=old_id)["id"]

    active = recall("server port", limit=10)["results"]
    active_ids = {r["id"] for r in active}
    assert new_id in active_ids
    assert old_id not in active_ids

    history = recall("server port", include_superseded=True, limit=10)["results"]
    old = next(r for r in history if r["id"] == old_id)
    assert old["superseded_by"] == new_id


def test_update_rejects_superseded_and_forgotten_memories():
    old_id = remember("The old deployment uses blue green")["id"]
    new_id = remember("The new deployment uses canary", supersede=old_id)["id"]

    superseded = update_memory(old_id, content="Should not be editable")
    assert superseded["ok"] is False
    assert superseded["status"] == "conflict"

    forgotten_id = remember("A deployment fact to forget")["id"]
    forget(forgotten_id)
    forgotten = update_memory(forgotten_id, importance=0.1)
    assert forgotten["ok"] is False
    assert forgotten["status"] == "conflict"

    active = update_memory(new_id, importance=0.5)
    assert active["ok"] is True


def test_remember_check_contradictions_auto_supersedes_the_old_fact():
    old_id = remember("Alice likes tea", entity="Alice")["id"]

    result = remember("Alice does not like tea", entity="Alice",
                      check_contradictions=True)

    assert result["ok"] is True
    assert result["contradictions"]["found"] is True
    new_id = result["id"]
    assert new_id != old_id
    assert result["warnings"]

    history = recall("Alice tea", include_superseded=True, limit=10)["results"]
    old = next(r for r in history if r["id"] == old_id)
    assert old["superseded_by"] == new_id


def test_contradictions_report_finds_conflicting_active_facts():
    first = remember("Alice likes tea", entity="Alice")["id"]
    second = remember("Alice does not like tea", entity="Alice")["id"]

    report = contradictions_report(entity="Alice")

    assert report["ok"] is True
    assert report["count"] == 1
    pair = report["results"][0]
    assert {pair["a"]["id"], pair["b"]["id"]} == {first, second}
    assert pair["confidence"] == 0.95


def test_contradictions_report_without_entity_scans_the_whole_scope():
    report = contradictions_report()

    assert report["ok"] is True
    assert report["count"] == 0
    assert any("nothing to compare" in w for w in report["warnings"])
