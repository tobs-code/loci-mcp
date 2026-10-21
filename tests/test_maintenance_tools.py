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


def _export_fixture() -> str:
    remember("Export me", project="alpha", entity="Alpha")
    relate("Alpha", "Beta", "KNOWS", project="alpha", weight=0.7)
    return export_memories()


def test_export_json_contains_records_and_graph():
    payload = json.loads(_export_fixture())

    assert payload["type"] == "memory_export"
    assert payload["version"] == "0.2"
    assert payload["count"] == 1
    assert payload["records"][0]["content"] == "Export me"
    assert payload["records"][0]["project"] == "alpha"
    assert {e["name"] for e in payload["graph"]["entities"]} == {"Alpha", "Beta"}
    assert payload["graph"]["relations"][0]["relation_type"] == "KNOWS"


def test_export_jsonl_is_flat_and_has_no_graph():
    _export_fixture()
    data = export_memories(format="jsonl")

    lines = [line for line in data.splitlines() if line.strip()]
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["content"] == "Export me"
    assert "graph" not in record


def test_import_restores_records_and_graph_into_a_new_project():
    payload = _export_fixture()

    result = import_memories(payload, project="restored")

    assert result == {"imported": 1, "skipped": 0,
                      "graph_entities": 2, "graph_relations": 1}
    hits = recall("Export me", project="restored")
    assert hits[0]["content"] == "Export me"
    assert [r["entity"] for r in find_connections("Alpha", project="restored")] == ["Beta"]


def test_reimport_is_idempotent():
    payload = _export_fixture()
    import_memories(payload, project="restored")

    again = import_memories(payload, project="restored")

    assert again["imported"] == 0
    assert again["skipped"] == 1


def test_import_jsonl_round_trip():
    _export_fixture()
    data = export_memories(format="jsonl")

    result = import_memories(data, project="from-jsonl", format="jsonl")

    assert result == {"imported": 1, "skipped": 0}
    assert recall("Export me", project="from-jsonl")[0]["content"] == "Export me"


def test_export_and_import_reject_unknown_formats():
    assert "unsupported format" in export_memories(format="xml")
    result = import_memories("{}", format="xml")
    assert result["imported"] == 0
    assert "unsupported format" in result["error"]


def test_import_rejects_invalid_json():
    result = import_memories("{not json")

    assert result["imported"] == 0
    assert "invalid json data" in result["error"]


def test_server_health_and_stats_resource():
    remember("Stats fact", project="alpha")

    assert server_health() == {"postgres": True, "inference": True}

    stats = json.loads(stats_resource())
    assert stats["total_memories"] == 1
    assert stats["active_memories"] == 1
    assert stats["projects"] == ["alpha"]
    assert stats["inference_healthy"] is True
