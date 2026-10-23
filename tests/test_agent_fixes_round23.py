"""Regression tests for the fixes of the latest audit round.

Covers:
- `supersede_content` finds the shared (global) fact even when the new write
  names an entity;
- a supersede carries the replaced `mem_type` when the new fact omits it;
- `recall` finds an identifier via the exact-substring leg;
- included history is ranked below the current fact (`_history_penalty`);
- `update_memory` refuses an expired row unless it is revived explicitly;
- `relate` without `weight` keeps an existing edge's weight;
- `find_connections` returns the stored edge context;
- `export_memories(entity=...)` keeps the entity's relation-context sentences;
- `import_memories` accepts JSONL data as an already-parsed list.
"""

import json

from mcp_server.graph import find_connections, list_relations, relate
from mcp_server.server import (
    export_memories,
    import_memories,
    list_memories,
    recall,
    remember,
    update_batch,
    update_memory,
)


def test_supersede_content_reaches_the_global_fact_with_an_entity():
    old = remember("Round23 shared cache port is 6379", entity="Round23Cache")
    assert old["ok"] is True

    new = remember(
        "Round23 shared cache port is 6380",
        entity="Round23Cache",
        project="round23-supersede",
        supersede_content="Round23 shared cache port is 6379",
    )

    assert new["ok"] is True
    assert old["id"] in (new.get("cross_scope_superseded") or [])
    stored = list_memories(memory_id=old["id"], project="global",
                           include_superseded=True)["memory"]
    assert stored["superseded_by"] == new["id"]


def test_supersede_carries_the_replaced_mem_type():
    old = remember("Round23 deploy uses a blue-green switch",
                   entity="Round23Deploy", mem_type="decision",
                   project="round23-type")
    new = remember("Round23 deploy uses a canary switch",
                   entity="Round23Deploy", project="round23-type",
                   supersede=old["id"])

    assert new["ok"] is True
    stored = list_memories(memory_id=new["id"], project="round23-type")["memory"]
    assert stored["mem_type"] == "decision"


def test_recall_finds_an_identifier_through_the_exact_leg():
    remember("The pipeline token is ZEBRAFISH_TOKEN_42 for the staging job",
             project="round23-exact")

    result = recall("ZEBRAFISH_TOKEN_42", project="round23-exact", limit=5)

    assert result["results"]
    assert result["results"][0]["_exact_match"] == "substring"
    assert "ZEBRAFISH_TOKEN_42" in result["results"][0]["content"]


def test_included_history_is_penalised_below_the_current_fact():
    old = remember("Round23 queue uses RabbitMQ 3.12",
                   project="round23-history")
    new = remember("Round23 queue uses RabbitMQ 3.13",
                   project="round23-history", supersede=old["id"])

    result = recall("Round23 queue RabbitMQ", project="round23-history",
                    include_superseded=True, limit=5)

    by_id = {row["id"]: row for row in result["results"]}
    assert by_id[new["id"]]["_relevance_score"] >= \
        by_id[old["id"]]["_relevance_score"]
    assert by_id[old["id"]]["_history_penalty"] == 0.5
    assert "_history_penalty" not in by_id[new["id"]]


def test_expired_memory_must_be_revived_explicitly():
    created = remember("Round23 temporary lock expires quickly",
                       project="round23-ttl",
                       expires_at="2020-01-01T00:00:00Z")
    assert created["ok"] is True

    refused = update_memory(created["id"], content="Round23 lock is renewed",
                            project="round23-ttl")
    assert refused["ok"] is False
    assert refused["status"] == "conflict"
    assert "expired" in refused["error"]

    revived = update_memory(created["id"], clear_expires_at=True,
                            project="round23-ttl")
    assert revived["ok"] is True
    visible = list_memories(memory_id=created["id"], project="round23-ttl")
    assert visible["ok"] is True


def test_update_batch_refuses_an_expired_row_without_revive():
    created = remember("Round23 batch lock expires quickly",
                       project="round23-batch-ttl",
                       expires_at="2020-01-01T00:00:00Z")
    result = update_batch([{
        "memory_id": created["id"],
        "content": "Round23 batch lock is renewed",
    }], project="round23-batch-ttl")

    assert result["ok"] is False
    assert result["results"][0]["status"] == "conflict"
    assert "expired" in result["results"][0]["error"]


def test_relate_keeps_the_weight_when_only_context_is_added():
    relate("Round23A", "Round23B", "KNOWS", weight=0.4)
    updated = relate("Round23A", "Round23B", "KNOWS",
                     context="They worked on the same service")

    assert updated["weight"] == 0.4
    connections = find_connections("Round23A")
    edge = connections["results"][0]
    assert edge["weight"] == 0.4
    assert edge["context"] == "They worked on the same service"


def test_list_relations_browses_without_a_start_entity():
    relate("Round23ListA", "Round23ListB", "USES", weight=0.7,
           project="round23-list")
    relate("Round23ListB", "Round23ListC", "CALLS", weight=0.9,
           project="round23-list")

    all_edges = list_relations(project="round23-list")
    assert all_edges["ok"] is True
    assert all_edges["count"] == all_edges["total"] == 2
    assert all_edges["results"][0]["relation_type"] == "CALLS"
    assert all_edges["results"][0]["weight"] == 0.9

    filtered = list_relations(entity="Round23ListB", project="round23-list")
    assert filtered["total"] == 2
    assert list_relations(relation_type="USES",
                          project="round23-list")["total"] == 1


def test_export_of_one_entity_keeps_its_edge_context():
    relate("Round23ExportA", "Round23ExportB", "USES",
           context="Round23ExportA calls Round23ExportB per request",
           project="round23-export")

    exported = export_memories(project="round23-export",
                               entity="Round23ExportA")

    assert exported["ok"] is True
    contents = [record["content"] for record in exported["data"]["records"]]
    assert "Round23ExportA calls Round23ExportB per request" in contents


def test_import_accepts_a_parsed_jsonl_list():
    remember("Round23 jsonl record survives a parsed list import",
             project="round23-jsonl")
    exported = export_memories(project="round23-jsonl", format="jsonl")
    assert exported["ok"] is True
    records = [json.loads(line) for line in exported["data"].splitlines()
               if line.strip()]
    assert records

    imported = import_memories(
        data=records, format="jsonl",
        project="round23-jsonl-import", dry_run=True)

    assert imported["ok"] is True
    assert imported["would_create"] == len(records)
