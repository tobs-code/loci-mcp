"""Regression tests for the tenth round of fixes (audit follow-up).

Covers:
- explicit `project="global"` on export stays the shared scope (memories and
  graph), instead of widening to every scope;
- `update_memory`/`update_batch` refuse a content change that would duplicate
  another active memory (with `dedupe="off"` as the escape hatch);
- tolerant `metadata_filter` values (string vs stored number);
- blank entity names are rejected instead of creating invisible rows;
- alias conflicts (`entity` vs `name`, `entity` vs `entity_name`) are named;
- graph contract fixes: empty `relate_batch`, self-loop warning, zero-hop
  `find_paths`, and a stale entity hint that ignores forgotten rows;
- offset warnings when the requested page is beyond the result set;
- JSONL import of an export object is refused with a helpful error;
- the NLI parser recovers label/score from a non-JSON answer.
"""

from inference.llm import parse_nli_json
from mcp_server.server import (
    export_memories,
    find_connections,
    find_paths,
    forget,
    get_entity,
    import_memories,
    list_entities,
    list_memories,
    relate,
    relate_batch,
    remember,
    update_batch,
    update_memory,
)


# --------------------------------------------------------------------------
# Export: project="global" is the shared scope, not "every scope"
# --------------------------------------------------------------------------

def test_export_global_only_does_not_leak_project_rows():
    remember("global export probe", scope="global", dedupe="off")
    remember("project export probe", project="export-probe", dedupe="off")

    exported = export_memories(project="global")

    assert exported["ok"] is True
    contents = {r["content"] for r in exported["data"]["records"]}
    assert contents == {"global export probe"}
    assert all(r["project"] == "global" for r in exported["data"]["records"])


def test_export_global_only_filters_graph_relations():
    relate("Alpha", "Beta", "links", project="graph-export-probe")
    remember("global graph probe", scope="global", dedupe="off")

    exported = export_memories(project="global", include_graph=True)

    assert exported["data"]["graph"]["entities"] == []
    assert exported["data"]["graph"]["relations"] == []


def test_export_without_project_still_covers_every_scope():
    remember("global export all", scope="global", dedupe="off")
    remember("project export all", project="export-all-probe", dedupe="off")

    exported = export_memories()

    projects = {r["project"] for r in exported["data"]["records"]}
    assert {"global", "export-all-probe"} <= projects


# --------------------------------------------------------------------------
# Update dedupe
# --------------------------------------------------------------------------

def test_update_memory_refuses_a_duplicate_content_change():
    first = remember("update dedupe probe: alpha")["id"]
    second = remember("update dedupe probe: beta")["id"]

    result = update_memory(second, content="update dedupe probe: alpha")

    assert result["status"] == "duplicate"
    assert result["duplicate_of"] == first
    assert result["warnings"
                  ] and "update not applied" in result["warnings"][0]
    listed = list_memories(memory_id=second)
    assert listed["results"][0]["content"] == "update dedupe probe: beta"


def test_update_memory_allows_a_duplicate_with_dedupe_off():
    remember("update off probe: alpha")["id"]
    second = remember("update off probe: beta")["id"]

    result = update_memory(second, content="update off probe: alpha",
                           dedupe="off")

    assert result["status"] == "updated"


def test_update_memory_does_not_flag_its_own_content():
    memory_id = remember("update self probe: unchanged")["id"]

    result = update_memory(memory_id, content="update self probe: unchanged")

    assert result["status"] == "updated"


def test_update_batch_reports_duplicate_entries_as_partial():
    first = remember("batch dup probe: alpha")["id"]
    second = remember("batch dup probe: beta")["id"]

    result = update_batch([
        {"memory_id": second, "content": "batch dup probe: alpha"},
        {"memory_id": first, "content": "batch dup probe: gamma"},
    ])

    assert result["status"] == "partial"
    assert result["results"][0]["status"] == "duplicate"
    assert result["results"][0]["duplicate_of"] == first
    assert result["results"][1]["status"] == "updated"


# --------------------------------------------------------------------------
# Tolerant metadata filter
# --------------------------------------------------------------------------

def test_metadata_filter_matches_a_number_stored_as_int():
    memory_id = remember("metadata probe: port 8001",
                         metadata={"port": 8001})["id"]

    by_number = list_memories(metadata_filter={"port": 8001})
    by_string = list_memories(metadata_filter={"port": "8001"})

    assert memory_id in {r["id"] for r in by_number["results"]}
    assert memory_id in {r["id"] for r in by_string["results"]}


def test_metadata_filter_matches_a_boolean_stored_as_bool():
    memory_id = remember("metadata probe: flag", metadata={"prod": True})["id"]

    listed = list_memories(metadata_filter={"prod": True})

    assert memory_id in {r["id"] for r in listed["results"]}


def test_metadata_filter_still_matches_nested_objects():
    memory_id = remember("metadata probe: nested",
                         metadata={"owner": {"team": "core"}})["id"]

    listed = list_memories(metadata_filter={"owner": {"team": "core"}})

    assert memory_id in {r["id"] for r in listed["results"]}


# --------------------------------------------------------------------------
# Entity validation and alias conflicts
# --------------------------------------------------------------------------

def test_remember_rejects_a_blank_entity():
    result = remember("blank entity probe", entity="   ")

    assert result["ok"] is False
    assert "entity" in result["error"]


def test_list_memories_rejects_a_blank_entity():
    result = list_memories(entity=" ")

    assert result["ok"] is False
    assert "entity" in result["error"]


def test_get_entity_rejects_conflicting_aliases():
    remember("alias probe", entity="AliasOne")

    result = get_entity(entity="AliasOne", name="AliasTwo")

    assert result["ok"] is False
    assert "differ" in result["error"]


def test_find_connections_rejects_conflicting_aliases():
    relate("AliasTwo", "AliasThree", "links")

    result = find_connections(entity="AliasTwo", entity_name="AliasThree")

    assert result["ok"] is False
    assert "differ" in result["error"]


# --------------------------------------------------------------------------
# Graph contract
# --------------------------------------------------------------------------

def test_relate_batch_without_edges_is_an_error():
    result = relate_batch([])

    assert result["ok"] is False
    assert "empty" in result["error"]


def test_relate_reports_a_self_loop():
    result = relate("LoopNode", "LoopNode", "points_to")

    assert result["ok"] is True
    assert any("self-loop" in w for w in result["warnings"])


def test_find_paths_from_entity_to_itself_is_zero_hops():
    relate("ZeroA", "ZeroB", "links")

    result = find_paths("ZeroA", "ZeroA")

    assert result["ok"] is True
    assert result["count"] == 1
    assert result["results"][0]["hops"] == 0
    assert result["results"][0]["nodes"][0]["name"] == "ZeroA"


def test_entity_hint_ignores_forgotten_rows():
    memory = remember("stale hint probe", entity="StaleNode")
    forget(memory_id=memory["id"], soft=True, project="global")

    result = find_connections("StaleNode")

    assert result["ok"] is False
    assert "StaleNode" in result["error"]
    # A forgotten row is not an active memory-only entity: the hint must not
    # suggest `relate` for a node whose only memory was soft-forgotten.
    assert "memory-only" not in result["error"]


# --------------------------------------------------------------------------
# Offset warnings
# --------------------------------------------------------------------------

def test_list_memories_warns_when_offset_is_beyond_the_result():
    remember("offset probe", dedupe="off")

    result = list_memories(offset=50)

    assert result["count"] == 0
    assert any("beyond" in w for w in result["warnings"])


def test_list_entities_warns_when_offset_is_beyond_the_result():
    remember("offset entity probe", entity="OffsetEntity")

    result = list_entities(offset=50)

    assert result["count"] == 0
    assert any("beyond" in w for w in result["warnings"])


# --------------------------------------------------------------------------
# JSONL import contract
# --------------------------------------------------------------------------

def test_import_jsonl_rejects_an_export_object():
    exported = export_memories(format="json")
    result = import_memories(data=exported["data"], format="jsonl")

    assert result["ok"] is False
    assert "jsonl" in result["error"].lower()


def test_import_jsonl_accepts_a_jsonl_string():
    payload = export_memories(format="jsonl")["data"]

    result = import_memories(data=payload, format="jsonl")

    assert result["ok"] is True


# --------------------------------------------------------------------------
# NLI parser fallback
# --------------------------------------------------------------------------

def test_parse_nli_json_recovers_a_loose_answer():
    parsed = parse_nli_json("label: contradiction, score: 0.9")

    assert parsed["label"] == "contradiction"
    assert parsed["score"] == 0.9


def test_parse_nli_json_does_not_guess_a_missing_label():
    assert parse_nli_json("score: 0.9") is None
