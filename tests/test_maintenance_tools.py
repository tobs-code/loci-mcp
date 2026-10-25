import json

from mcp_server.server import (
    export_memories,
    find_connections,
    import_memories,
    recall,
    relate,
    remember,
    server_health,
    stats_resource,
)


def _export_fixture() -> dict:
    remember("Export me", project="alpha", entity="Alpha")
    relate("Alpha", "Beta", "KNOWS", project="alpha", weight=0.7)
    return export_memories()


def test_export_json_contains_records_and_graph():
    exported = _export_fixture()
    assert "records" not in exported
    payload = exported["data"]

    assert payload["type"] == "memory_export"
    assert payload["version"] == "0.4"
    assert payload["count"] == 1
    assert payload["records"][0]["content"] == "Export me"
    assert payload["records"][0]["project"] == "alpha"
    assert {e["name"] for e in payload["graph"]["entities"]} == {"Alpha", "Beta"}
    assert payload["graph"]["relations"][0]["relation_type"] == "KNOWS"


def test_export_jsonl_includes_graph_records():
    _export_fixture()
    exported = export_memories(format="jsonl")
    assert "records" not in exported

    lines = [json.loads(line) for line in exported["data"].splitlines()
             if line.strip()]
    memories = [r for r in lines if not r.get("record_type")]
    entities = [r for r in lines if r.get("record_type") == "entity"]
    relations = [r for r in lines if r.get("record_type") == "relation"]
    assert len(memories) == 1
    assert memories[0]["content"] == "Export me"
    assert "graph" not in memories[0]
    assert {e["name"] for e in entities} == {"Alpha", "Beta"}
    assert relations[0]["relation_type"] == "KNOWS"
    assert relations[0]["weight"] == 0.7
    assert exported["data_format"] == "jsonl"
    assert exported["graph_included"] is True


def test_import_restores_records_and_graph_into_a_new_project():
    payload = _export_fixture()["data"]

    result = import_memories(payload, project="restored")

    assert result["ok"] is True
    assert result["imported"] == 1
    assert result["skipped"] == 0
    assert result["graph_entities"] == 2
    assert result["graph_relations"] == 1
    hits = recall("Export me", project="restored")
    assert hits["results"][0]["content"] == "Export me"
    assert [r["entity"] for r in
            find_connections("Alpha", project="restored")["results"]] == ["Beta"]


def test_reimport_is_idempotent():
    payload = _export_fixture()["data"]
    import_memories(payload, project="restored")

    again = import_memories(payload, project="restored")

    assert again["imported"] == 0
    assert again["skipped"] == 1


def test_import_jsonl_round_trip():
    _export_fixture()
    data = export_memories(format="jsonl")["data"]

    result = import_memories(data, project="from-jsonl", format="jsonl")

    assert result["imported"] == 1
    assert result["skipped"] == 0
    hits = recall("Export me", project="from-jsonl")
    assert hits["results"][0]["content"] == "Export me"


def test_export_and_import_reject_unknown_formats():
    exported = export_memories(format="xml")
    assert exported["ok"] is False
    assert "unsupported format" in exported["error"]

    result = import_memories("{}", format="xml")
    assert result["ok"] is False
    assert result["imported"] == 0
    assert "unsupported format" in result["error"]


def test_import_rejects_invalid_json():
    result = import_memories("{not json")

    assert result["ok"] is False
    assert result["imported"] == 0
    assert "invalid json data" in result["error"]


def test_server_health_and_stats_resource():
    remember("Stats fact", project="alpha")

    health = server_health()
    assert health["ok"] is True
    assert health["postgres"] is True
    assert health["inference"] is True

    stats = json.loads(stats_resource())
    assert stats["total_memories"] == 1
    assert stats["active_memories"] == 1
    # The shared scope is a scope like every other one and is listed too.
    assert stats["projects"] == ["alpha", "global"]
    assert stats["inference_healthy"] is True
