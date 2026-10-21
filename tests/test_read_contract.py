"""Contract tests for the read tools (recall, get_entity, list_entities,
init_context): uniform envelope, slim default payload, field selection."""

from mcp_server.server import (
    get_entity,
    init_context,
    list_entities,
    recall,
    remember,
)


def _hit(result, memory_id):
    return next(r for r in result["results"] if r["id"] == memory_id)


def test_recall_answers_with_the_envelope_and_a_result_list():
    remember("Read contract: a yellow kayak in the harbor")

    result = recall("yellow kayak harbor", limit=5)

    assert result["ok"] is True
    assert result["status"] == "ok"
    assert result["error"] is None
    assert result["count"] == len(result["results"])
    assert result["count"] >= 1


def test_recall_hides_internal_fields_by_default():
    memory_id = remember("Read contract: slim rows have no content hash")["id"]

    result = recall("slim rows content hash", limit=5)

    hit = _hit(result, memory_id)
    assert "content_hash" not in hit
    assert "deleted_at" not in hit
    assert hit["content"]


def test_recall_can_return_only_the_requested_fields():
    memory_id = remember("Read contract: field selection wins")["id"]

    result = recall("field selection wins", limit=5, fields=["id", "content"])

    hit = _hit(result, memory_id)
    assert set(hit) == {"id", "content"}


def test_recall_can_return_the_original_project_name():
    memory_id = remember("Read contract: project names are preserved",
                         project="Team Alpha")["id"]

    result = recall("project names are preserved", project="Team Alpha",
                    limit=5, fields=["id", "project"])

    hit = _hit(result, memory_id)
    assert hit == {"id": memory_id, "project": "Team Alpha"}


def test_recall_warns_when_graph_boost_has_no_entity():
    remember("Read contract: graph boost without entity")

    result = recall("graph boost without entity", limit=5, graph_boost=True)

    assert result["ok"] is True
    assert any("entity" in w for w in result["warnings"])


def test_recall_can_return_only_superseded_rows():
    old_id = remember("Read contract: the queue runs on port 7000")["id"]
    new_id = remember("Read contract: the queue runs on port 7100",
                      supersede=old_id)["id"]

    result = recall("queue runs on port", limit=10, only_superseded=True)

    ids = {r["id"] for r in result["results"]}
    assert old_id in ids
    assert new_id not in ids


def test_get_entity_answers_with_the_envelope():
    memory_id = remember("Read contract: Nina reviews the dashboard",
                         entity="Nina")["id"]

    result = get_entity(entity="Nina")

    assert result["ok"] is True
    assert result["status"] == "ok"
    assert result["entity"] == "Nina"
    assert [m["id"] for m in result["memories"]] == [memory_id]


def test_get_entity_can_limit_the_memory_fields():
    memory_id = remember("Read contract: field selection for entities",
                         entity="Otto")["id"]

    result = get_entity(entity="Otto", fields=["id", "content"])

    memory = next(m for m in result["memories"] if m["id"] == memory_id)
    assert set(memory) == {"id", "content"}


def test_list_entities_answers_with_the_envelope():
    remember("Read contract: Uma writes the release notes", entity="Uma")

    result = list_entities()

    assert result["ok"] is True
    assert result["count"] == len(result["results"])
    assert any(r["name"] == "Uma" for r in result["results"])


def test_list_entities_rejects_non_positive_limits():
    result = list_entities(limit=0)
    assert result["ok"] is False
    assert "limit" in result["error"]


def test_init_context_slim_bootstrap_drops_internal_fields():
    remember("Read contract: bootstrap fact for alpha", project="alpha")

    result = init_context(project="alpha")

    assert result["ok"] is True
    assert result["status"] == "ok"
    memory = result["recent_memories"][0]
    assert set(memory) <= {"id", "content", "entity", "mem_type", "importance",
                            "updated_at", "scope"}


def test_init_context_removes_near_duplicates():
    remember("Read contract: the deploy pipeline runs on monday",
             entity="Deploy")
    remember("Read contract: the deploy pipeline runs on Mondays",
             entity="Deploy")

    result = init_context()

    pipeline = [m for m in result["recent_memories"]
                if "deploy pipeline" in m["content"].lower()]
    assert len(pipeline) == 1


def test_init_context_detail_full_keeps_the_internal_fields():
    remember("Read contract: full detail bootstrap", project="beta")

    result = init_context(project="beta", detail="full")

    memory = result["recent_memories"][0]
    assert "content_hash" in memory
    assert "access_count" in memory
