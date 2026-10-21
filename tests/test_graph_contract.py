"""Contract tests for the graph and maintenance tools."""

import json

from mcp_server.server import (
    contradictions_report,
    export_memories,
    find_connections,
    find_paths,
    import_memories,
    recall,
    relate,
    relate_batch,
    server_health,
)


def test_relate_answers_with_the_envelope():
    result = relate("A", "B", "KNOWS", weight=0.5)

    assert result["ok"] is True
    assert result["status"] == "created"
    assert result["from_entity"] == "A"
    assert result["to_entity"] == "B"
    assert result["relation_type"] == "KNOWS"
    assert result["weight"] == 0.5
    assert result["context_memory"] is None


def test_relate_reports_an_update():
    relate("A", "B", "KNOWS", weight=0.5)

    result = relate("A", "B", "KNOWS", weight=0.9)

    assert result["status"] == "updated"
    assert result["weight"] == 0.9


def test_relate_reports_invalid_input_as_a_failed_envelope():
    result = relate("A", "B", "BAD TYPE")

    assert result["ok"] is False
    assert result["status"] == "error"
    assert "relation_type" in result["error"]


def test_relate_batch_answers_with_the_envelope():
    result = relate_batch([
        {"from_entity": "A", "to_entity": "B", "relation_type": "KNOWS"},
    ])

    assert result["ok"] is True
    assert result["count"] == 1
    assert result["results"][0]["status"] == "created"


def test_find_connections_answers_with_the_envelope():
    relate("A", "B", "KNOWS")

    result = find_connections("A")

    assert result["ok"] is True
    assert result["count"] == 1
    assert result["results"][0]["entity"] == "B"


def test_find_connections_reports_an_invalid_direction():
    result = find_connections("A", direction="sideways")

    assert result["ok"] is False
    assert "direction" in result["error"]


def test_find_paths_answers_with_the_envelope():
    relate("A", "B", "KNOWS")
    relate("B", "C", "KNOWS")

    result = find_paths("A", "C")

    assert result["ok"] is True
    assert result["count"] == 1
    assert [n["name"] for n in result["results"][0]["nodes"]] == ["A", "B", "C"]


def test_contradictions_report_answers_with_the_envelope():
    from mcp_server.server import remember

    first = remember("Peter likes tea", entity="Peter")["id"]
    second = remember("Peter does not like tea", entity="Peter")["id"]

    result = contradictions_report(entity="Peter")

    assert result["ok"] is True
    assert result["count"] == 1
    assert {result["results"][0]["a"]["id"],
            result["results"][0]["b"]["id"]} == {first, second}


def test_contradictions_report_without_entity_is_a_failed_envelope():
    result = contradictions_report()

    assert result["ok"] is False
    assert "entity" in result["error"]


def test_server_health_answers_with_the_envelope():
    result = server_health()

    assert result["ok"] is True
    assert result["postgres"] is True
    assert result["inference"] is True


def test_export_answers_with_the_envelope_and_round_trips_through_import():
    from mcp_server.server import remember

    remember("Export contract fact", project="alpha", entity="Alpha")

    exported = export_memories()

    assert exported["ok"] is True
    assert exported["format"] == "json"
    assert exported["count"] == 1
    assert exported["data"]["records"][0]["content"] == "Export contract fact"

    imported = import_memories(exported["data"], project="restored")

    assert imported["ok"] is True
    assert imported["imported"] == 1
    assert imported["skipped"] == 0
    hits = recall("Export contract fact", project="restored")
    assert hits["results"][0]["content"] == "Export contract fact"


def test_import_accepts_the_json_string_of_an_export():
    from mcp_server.server import remember

    remember("Export contract string fact", project="beta")
    payload = json.dumps(export_memories()["data"])

    result = import_memories(payload, project="from-string")

    assert result["ok"] is True
    assert result["imported"] == 1


def test_import_rejects_an_unknown_format():
    result = import_memories("{}", format="xml")

    assert result["ok"] is False
    assert "unsupported format" in result["error"]


def test_import_rejects_invalid_json():
    result = import_memories("{not json")

    assert result["ok"] is False
    assert "invalid json data" in result["error"]
