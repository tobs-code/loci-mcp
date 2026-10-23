"""Contract tests for the agent-usability fixes:

- an invalid call answers with the envelope instead of a raw validation error
- `min_similarity` has a documented default that is not noise
- a memory id that exists in several scopes is reported, not guessed
- the contradiction check covers every memory type, not only `fact`
- `list_memories` / `list_entities` can be paged
- `init_context` / `drop_project` manage the project scopes
"""

import asyncio

from mcp_server.app import app
from mcp_server.config import DEFAULT_MIN_SIMILARITY
from mcp_server.server import (
    drop_project,
    forget,
    import_memories,
    init_context,
    list_entities,
    list_memories,
    remember,
    update_memory,
)


def _call(tool: str, arguments: dict):
    """Call a tool through the MCP tool manager (as a client would)."""
    return asyncio.run(app._tool_manager.call_tool(tool, arguments))


def test_invalid_call_answers_with_the_envelope():
    result = _call("recall", {})

    assert result["ok"] is False
    assert result["status"] == "error"
    assert result["error"]
    assert "query" in result["error"]


def test_invalid_argument_type_answers_with_the_envelope():
    result = _call("remember", {"content": 42})

    assert result["ok"] is False
    assert result["status"] == "error"
    assert "content" in result["error"]


def test_min_similarity_default_is_documented_and_above_noise():
    assert DEFAULT_MIN_SIMILARITY == 0.30
    assert DEFAULT_MIN_SIMILARITY > 0.288

    schema = app._tool_manager.get_tool("recall").parameters
    description = schema["properties"]["min_similarity"]["description"]
    assert "0.30" in description


def test_ambiguous_memory_id_is_reported_instead_of_guessed():
    mem_id = remember("Ambiguity probe: the service writes to Postgres",
                      project="amb-a")["id"]
    import_memories(
        {"records": [{"id": mem_id,
                      "content": "Ambiguity probe: the service writes to Postgres"}]},
        project="amb-b",
    )

    result = update_memory(mem_id, content="Ambiguity probe: changed")

    assert result["ok"] is False
    assert "ambiguous" in result["error"]
    assert set(result["scopes"]) == {"amb-a", "amb-b"}

    scoped = update_memory(mem_id, content="Ambiguity probe: changed",
                            project="amb-b")
    assert scoped["ok"] is True
    assert scoped["project"] == "amb-b"


def test_forget_refuses_an_ambiguous_memory_id():
    mem_id = remember("Ambiguity probe 2: the queue is Redis",
                      project="amb-c")["id"]
    import_memories(
        {"records": [{"id": mem_id,
                      "content": "Ambiguity probe 2: the queue is Redis"}]},
        project="amb-d",
    )

    result = forget(memory_id=mem_id)

    assert result["ok"] is False
    assert "ambiguous" in result["error"]
    assert set(result["scopes"]) == {"amb-c", "amb-d"}


def test_contradiction_check_covers_every_memory_type():
    remember("Type probe: the service does not use Redis for sessions.",
             entity="TypeProbe", mem_type="decision", project="type-probe")

    result = remember("Type probe: the service uses Redis for sessions.",
                      entity="TypeProbe", mem_type="fact", project="type-probe",
                      check_contradictions=True)

    assert result["ok"] is True
    assert result["contradictions"]["found"] is True
    assert result["contradictions"]["superseded"]


def test_list_memories_pages_through_a_scope():
    for i in range(5):
        remember(f"Paging probe: memory number {i}", project="paging")

    first = list_memories(project="paging", limit=2)
    assert first["count"] == 2
    assert first["total"] >= 5
    assert first["has_more"] is True
    assert first["next_offset"] == 2

    second = list_memories(project="paging", limit=2, offset=first["next_offset"])
    assert second["count"] == 2

    ids_first = {r["id"] for r in first["results"]}
    ids_second = {r["id"] for r in second["results"]}
    assert not (ids_first & ids_second)


def test_list_entities_pages_through_a_scope():
    for i in range(4):
        remember(f"Entity paging probe {i}", entity=f"EntityPage{i}",
                 project="entity-paging")

    first = list_entities(project="entity-paging", limit=2)
    assert first["count"] == 2
    assert first["has_more"] is True

    second = list_entities(project="entity-paging", limit=2,
                           offset=first["next_offset"])
    assert second["count"] == 2
    names_first = {r["name"] for r in first["results"]}
    names_second = {r["name"] for r in second["results"]}
    assert not (names_first & names_second)


def test_init_context_reports_the_project_stats():
    remember("Scope probe: a fact in a listed project", project="listed-probe")

    result = init_context(project="*")

    assert result["ok"] is True
    projects = {row["project"]: row for row in result["project_stats"]}
    assert "listed-probe" in projects
    assert projects["listed-probe"]["memories"] >= 1
    assert projects["global"]["scope"] == "global"


def test_drop_project_removes_the_scope_completely():
    remember("Drop probe: a fact in a dropped project", entity="DropEntity",
             project="dropped-probe")

    preview = drop_project("dropped-probe", dry_run=True)
    assert preview["ok"] is True
    assert preview["counts"]["memories"] >= 1
    assert any(r["project"] == "dropped-probe"
               for r in init_context(project="*")["project_stats"])

    result = drop_project("dropped-probe")
    assert result["ok"] is True
    assert result["counts"]["memories"] >= 1

    assert not any(r["project"] == "dropped-probe"
                   for r in init_context(project="*")["project_stats"])
    assert list_memories(project="dropped-probe")["count"] == 0


def test_drop_project_refuses_the_global_scope():
    result = drop_project("global")

    assert result["ok"] is False
    assert "global" in result["error"]


def test_init_context_full_has_no_score_annotations():
    remember("Bootstrap probe: the batch job runs at night",
             project="bootstrap-probe")

    result = init_context(project="bootstrap-probe", detail="full")

    assert result["ok"] is True
    assert result["recent_memories"]
    for memory in result["recent_memories"]:
        assert not [key for key in memory if key.startswith("_")], \
            "the bootstrap is not a search result: no score annotations"
        assert "content_hash" not in memory
        assert "embedding" not in memory
        assert memory["project"] == "bootstrap-probe"
