"""Round-40 feature: JSONL backups carry the graph (entities + relations)."""
import json

from mcp_server.server import (
    export_memories,
    find_connections,
    import_memories,
    recall,
    relate,
    remember,
)

PROJECT = "round40-jsonl-graph"


def _fixture():
    remember("Round40 billing reads the ledger", project=PROJECT,
             entity="BillingService")
    relate("BillingService", "Ledger", "READS", project=PROJECT, weight=0.6,
           context="BillingService reads the ledger for invoices")
    return export_memories(project=PROJECT, format="jsonl")


def _lines(result):
    return [json.loads(line) for line in result["data"].splitlines()
            if line.strip()]


def test_jsonl_export_carries_entities_relations_and_context():
    exported = _fixture()
    assert exported["ok"] is True
    assert exported["graph_included"] is True

    lines = _lines(exported)
    entities = [r for r in lines if r.get("record_type") == "entity"]
    relations = [r for r in lines if r.get("record_type") == "relation"]
    memories = [r for r in lines if not r.get("record_type")]

    assert {e["name"] for e in entities} == {"BillingService", "Ledger"}
    assert len(relations) == 1
    assert relations[0]["relation_type"] == "READS"
    assert relations[0]["weight"] == 0.6
    assert relations[0]["context"] == (
        "BillingService reads the ledger for invoices")
    assert memories and all("content" in m for m in memories)


def test_jsonl_round_trip_restores_the_graph_and_its_context():
    exported = _fixture()
    copy = "round40-jsonl-graph-copy"

    result = import_memories(exported["data"], project=copy, format="jsonl")

    assert result["ok"] is True
    assert result["skipped"] == 0
    assert result["graph_entities_created"] == 2
    assert result["graph_relations_created"] == 1
    connections = find_connections("BillingService", project=copy)["results"]
    assert [c["entity"] for c in connections] == ["Ledger"]
    assert connections[0]["context"] == (
        "BillingService reads the ledger for invoices")
    assert recall("BillingService reads the ledger", project=copy)["results"]


def test_reimporting_jsonl_is_idempotent_for_memories_and_graph():
    exported = _fixture()
    copy = "round40-jsonl-graph-again"
    import_memories(exported["data"], project=copy, format="jsonl")

    again = import_memories(exported["data"], project=copy, format="jsonl")

    assert again["ok"] is True
    assert again["imported"] == 0
    assert again["graph_entities_created"] == 0
    assert again["graph_relations_created"] == 0
    assert again["skipped_reasons"]["id_exists"] == exported["count"]


def test_unknown_record_type_is_reported_per_line():
    data = "\n".join([
        json.dumps({"content": "Round40 survives a bad graph line"}),
        json.dumps({"record_type": "widget", "name": "Nope"}),
        json.dumps({"record_type": "entity"}),
    ])

    result = import_memories(data, project="round40-invalid-line",
                             format="jsonl")

    assert result["ok"] is True
    assert result["imported"] == 1
    assert result["skipped"] == 2
    assert result["skipped_reasons"]["invalid"] == 2
    assert any("jsonl line 2" in e for e in result["errors"])
    assert any("jsonl line 3" in e for e in result["errors"])


def test_legacy_jsonl_without_record_type_still_imports():
    data = json.dumps({"content": "Round40 legacy flat backup"})

    result = import_memories(data, project="round40-legacy", format="jsonl")

    assert result["ok"] is True
    assert result["imported"] == 1
    assert result["skipped"] == 0
    assert "graph_entities" not in result


def test_include_graph_false_keeps_jsonl_memory_only():
    _fixture()
    exported = export_memories(project=PROJECT, format="jsonl",
                               include_graph=False)

    assert exported["ok"] is True
    assert exported["graph_included"] is False
    assert all(not line.get("record_type") for line in _lines(exported))
