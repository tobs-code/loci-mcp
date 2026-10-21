import json

from mcp_server.server import (
    get_entity,
    init_context,
    list_entities,
    recall,
    remember,
    remember_batch,
    update_memory,
)


def test_remember_stores_a_fact_that_recall_finds():
    memory_id = remember("The sky is blue", entity="Sky", mem_type="fact")["id"]

    assert len(memory_id) == 16

    results = recall("sky blue", limit=5)["results"]

    assert any(r["id"] == memory_id for r in results)
    hit = next(r for r in results if r["id"] == memory_id)
    assert hit["content"] == "The sky is blue"
    assert hit["entity"] == "Sky"
    assert hit["access_count"] >= 1


def test_remember_rejects_duplicates_and_invalid_input():
    memory_id = remember("A uniquely stored fact")["id"]

    duplicate = remember("A uniquely stored fact")
    assert duplicate["status"] == "duplicate"
    assert duplicate["id"] == memory_id

    invalid_type = remember("x", mem_type="bogus")
    assert invalid_type["ok"] is False
    assert "Invalid mem_type" in invalid_type["error"]

    empty = remember("   ")
    assert empty["ok"] is False
    assert empty["error"] == "content must not be empty"

    bad_importance = remember("x", importance=2)
    assert bad_importance["ok"] is False
    assert "importance must be between" in bad_importance["error"]


def test_recall_sees_global_facts_inside_a_project_and_keeps_scopes_isolated():
    global_id = remember("Berlin is the capital of Germany")["id"]
    project_id = remember("Berlin has many museums", project="alpha")["id"]

    in_project = recall("Berlin", project="alpha", limit=10)["results"]
    in_global = recall("Berlin", limit=10)["results"]

    project_ids = {r["id"] for r in in_project}
    global_ids = {r["id"] for r in in_global}
    assert {global_id, project_id} <= project_ids
    assert global_id in global_ids
    assert project_id not in global_ids


def test_remember_can_write_global_from_inside_a_project():
    memory_id = remember("A global preference", project="alpha", scope="global")["id"]

    assert any(r["id"] == memory_id for r in recall("preference")["results"])


def test_remember_batch_reports_per_fact_status():
    result = remember_batch([
        {"content": "Batch fact one"},
        {"content": ""},
        {"content": "Batch note", "mem_type": "note"},
        {"content": "Batch fact one"},
    ])

    results = result["results"]
    assert [r["status"] for r in results] == ["stored", "error", "stored", "duplicate"]
    assert results[1]["error"] == "content is required"
    assert results[3]["id"] == results[0]["id"]


def test_update_memory_changes_content_importance_and_metadata():
    memory_id = remember("Original text")["id"]

    result = update_memory(
        memory_id,
        content="Updated text about mountains",
        importance=0.5,
        metadata={"source": "test"},
    )

    assert result["ok"] is True
    assert result["id"] == memory_id
    hits = recall("mountains", limit=5)["results"]
    hit = next(r for r in hits if r["id"] == memory_id)
    assert hit["content"] == "Updated text about mountains"
    assert hit["importance"] == 0.5
    assert hit["metadata"] == {"source": "test"}

    missing = update_memory("does-not-exist", importance=0.2)
    assert missing["ok"] is False
    assert missing["error"] == "NOT FOUND: does-not-exist"


def test_get_entity_returns_memories_even_without_a_graph_node():
    memory_id = remember("Carol prefers espresso", entity="Carol")["id"]

    result = get_entity("Carol")

    assert result["entity"] == "Carol"
    assert result["details"] == {
        "id": None,
        "name": "Carol",
        "type": "memory-only",
        "project": "global",
        "connections": [],
    }
    assert [m["id"] for m in result["memories"]] == [memory_id]


def test_list_entities_includes_memory_only_entities():
    remember("Eve reviews pull requests", entity="Eve", project="alpha")

    rows = list_entities(project="alpha")["results"]

    assert any(r["name"] == "Eve" and r["graph_node"] is False for r in rows)


def test_init_context_summarizes_the_selected_project():
    remember("Context fact for project", project="alpha")

    result = init_context(project="alpha")

    assert result["total_memories"] >= 1
    assert result["active_memories"] >= 1
    assert result["projects"] == ["alpha"]
    assert any("Context fact" in m["content"] for m in result["recent_memories"])
    assert result["inference_healthy"] is True
    assert isinstance(result["graph_entities_count"], int)
