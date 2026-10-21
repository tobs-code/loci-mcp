from mcp_server.server import (
    contradictions_report,
    forget,
    recall,
    remember,
)


def test_forget_soft_hides_the_fact_but_keeps_its_history():
    memory_id = remember("The secret token is abc")

    assert forget(memory_id) == f"Soft-forgotten {memory_id} (1 memories)"
    assert recall("secret token") == []

    history = recall("secret token", include_superseded=True)
    assert all(r["id"] != memory_id for r in history)
    assert forget("missing-id") == "NOT FOUND: missing-id"


def test_forget_hard_deletes_the_fact():
    memory_id = remember("A fact that will be purged")

    assert forget(memory_id, soft=False) == f"Deleted {memory_id} (1 memories)"
    assert recall("purged", include_superseded=True) == []


def test_forget_entity_only_touches_the_selected_scope():
    global_id = remember("Bob likes tea", entity="Bob")
    project_id = remember("Bob works on alpha", entity="Bob", project="alpha")

    result = forget(entity_name="Bob", project="alpha")

    assert result.startswith("Soft-forgotten Bob")
    project_hits = recall("Bob", project="alpha", limit=10)
    assert any(r["id"] == global_id for r in project_hits)
    assert all(r["id"] != project_id for r in project_hits)

    history = recall("Bob", project="alpha", include_superseded=True, limit=10)
    assert any(r["id"] == global_id for r in history)
    assert all(r["id"] != project_id for r in history)


def test_supersede_hides_the_old_fact_from_normal_recall():
    old_id = remember("The server runs on port 8000")
    new_id = remember("The server runs on port 9000", supersede=old_id)

    active = recall("server port", limit=10)
    active_ids = {r["id"] for r in active}
    assert new_id in active_ids
    assert old_id not in active_ids

    history = recall("server port", include_superseded=True, limit=10)
    old = next(r for r in history if r["id"] == old_id)
    assert old["superseded_by"] == new_id


def test_remember_check_contradictions_auto_supersedes_the_old_fact():
    old_id = remember("Alice likes tea", entity="Alice")

    result = remember("Alice does not like tea", entity="Alice",
                      check_contradictions=True)

    assert "CONTRADICTIONS_CHECKED" in result
    assert '"found": true' in result
    new_id = result.split()[0]
    assert new_id != old_id

    history = recall("Alice tea", include_superseded=True, limit=10)
    old = next(r for r in history if r["id"] == old_id)
    assert old["superseded_by"] == new_id


def test_contradictions_report_finds_conflicting_active_facts():
    first = remember("Alice likes tea", entity="Alice")
    second = remember("Alice does not like tea", entity="Alice")

    report = contradictions_report(entity="Alice")

    assert len(report) == 1
    pair = report[0]
    assert {pair["a"]["id"], pair["b"]["id"]} == {first, second}
    assert pair["confidence"] == 0.95


def test_contradictions_report_requires_an_entity():
    report = contradictions_report()

    assert report == [{"error": "entity parameter is required. "
                                "contradictions_report works on one entity at a time."}]
