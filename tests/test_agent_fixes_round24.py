"""Round 24: the fixes for the remaining test-report findings.

- `recall_batch` reports a bad query in its own entry instead of aborting.
- `recall` names its confidence and the best relevance.
- `list_memories(memory_ids=...)` separates hidden rows from missing ids.
- `forget(memory_ids=..., dry_run=True)` answers `would_*` and names scopes.
- `import_memories` accepts a list, keeps the rest after a broken record
  (savepoint), and reports a broken JSONL line without losing the file.
- `find_paths`/`list_relations` carry the edge's context sentence.
- A write's contradiction check replaces at most one memory by default.
"""
import json

from mcp_server.contradiction_scan import failure_kind
from mcp_server.server import (
    find_paths,
    forget,
    import_memories,
    list_memories,
    list_relations,
    recall,
    recall_batch,
    relate,
    remember,
)


def test_recall_reports_confidence_and_best_relevance():
    project = "round24-confidence"
    remember("Round24 deploys with helm charts", project=project)

    hit = recall("deploys with helm charts", project=project)
    assert hit["ok"] is True
    assert hit["confidence"] in ("high", "medium")
    assert hit["best_relevance"] is not None

    empty = recall("zzz-no-such-topic-anywhere", project=project)
    assert empty["count"] == 0
    assert empty["confidence"] == "none"
    assert empty["best_relevance"] is None


def test_recall_warns_about_low_confidence_hits():
    project = "round24-low-confidence"
    remember("Round24 alpha beta gamma", project=project)

    result = recall("unrelated delta epsilon zeta", project=project,
                    min_similarity=None)
    if result["count"]:
        assert result["confidence"] == "low"
        assert any("low confidence" in w for w in result["warnings"])
    else:
        assert result["confidence"] == "none"


def test_recall_batch_keeps_running_after_an_empty_query():
    result = recall_batch(["deploy", "   "], project="round24-batch")

    assert result["ok"] is True
    assert result["results"][1]["ok"] is False
    assert any("query 1 is empty" in w for w in result["warnings"])


def test_list_memories_bulk_lookup_separates_hidden_from_missing():
    project = "round24-hidden"
    memory = remember("Round24 hidden row", project=project)
    forget(memory["id"], project=project)

    result = list_memories(project=project,
                           memory_ids=[memory["id"], "missing-id"])

    assert result["missing"] == ["missing-id"]
    assert result["hidden"][0]["id"] == memory["id"]
    assert "include_deleted" in result["hidden"][0]["reveal_flags"]
    assert any("exist but are hidden" in w for w in result["warnings"])


def test_forget_bulk_dry_run_says_would_and_names_scopes():
    project = "round24-bulk"
    one = remember("Round24 bulk one", project=project)
    two = remember("Round24 bulk two", project=project)

    preview = forget(memory_ids=[one["id"], two["id"]], project=project,
                     dry_run=True)

    assert preview["ok"] is True
    assert preview["counts"]["memories"] == 2
    assert {entry["status"] for entry in preview["bulk_results"]} == {
        "would_soft_forget"}
    assert all(entry["scopes"] == [project]
               for entry in preview["bulk_results"])


def test_forget_bulk_reports_every_scope_of_an_id():
    project = "round24-cross-scope"
    shared = remember("Round24 cross scope fact", scope="global")
    imported = import_memories(
        {"records": [{"id": shared["id"],
                      "content": "Round24 cross scope fact"}]},
        project=project)
    assert imported["imported"] == 1

    result = forget(memory_ids=[shared["id"]])
    entry = result["bulk_results"][0]

    assert entry["count"] == 2
    assert set(entry["scopes"]) == {"global", project}


def test_import_accepts_a_record_list_and_survives_a_bad_embedding():
    project = "round24-import"
    payload = [
        {"id": "round24-bad", "content": "Round24 broken vector",
         "embedding": [0.1, 0.2, 0.3]},
        {"content": "Round24 good record after the broken one"},
    ]

    result = import_memories(data=payload, project=project)

    assert result["ok"] is True
    assert result["imported"] == 1
    assert result["invalid"] == 1
    assert any("round24-bad" in error for error in result["errors"])
    listed = list_memories(project=project)
    assert [row["content"] for row in listed["results"]] == [
        "Round24 good record after the broken one"]


def test_import_jsonl_reports_a_broken_line_and_keeps_the_rest():
    project = "round24-jsonl"
    data = "\n".join([
        json.dumps({"content": "Round24 jsonl one"}),
        "{not json",
        json.dumps({"content": "Round24 jsonl two"}),
    ])

    result = import_memories(data=data, format="jsonl", project=project)

    assert result["ok"] is True
    assert result["imported"] == 2
    assert result["skipped"] == 1
    assert result["skipped_reasons"]["invalid"] == 1
    assert any("jsonl line 2" in error for error in result["errors"])


def test_relations_and_paths_carry_the_context_sentence():
    project = "round24-graph"
    relate(from_entity="Round24A", to_entity="Round24B",
           relation_type="USES", context="A uses B for the tests",
           project=project)

    edges = list_relations(project=project)["results"]
    assert edges[0]["context"] == "A uses B for the tests"

    paths = find_paths("Round24A", "Round24B", project=project)["results"]
    relation = paths[0]["relations"][0]
    assert relation["relation_type"] == "USES"
    assert relation["context"] == "A uses B for the tests"


def test_contradiction_check_supersedes_at_most_one_memory_by_default():
    project = "round24-contradiction"
    remember("Round24 cache does not use redis for sessions",
             entity="Round24Cache", project=project)
    remember("Round24 cache does not use redis for sessions at all",
             entity="Round24Cache", project=project)

    result = remember("Round24 cache uses redis for sessions",
                      entity="Round24Cache", project=project,
                      check_contradictions=True)

    contradictions = result["contradictions"]
    assert len(contradictions["superseded"]) == 1
    assert contradictions["kept"]
    assert any("kept active" in w for w in result["warnings"])
    active = recall("Round24 cache redis sessions", project=project)
    assert active["count"] == 2


def test_contradiction_warn_mode_invalidates_nothing():
    project = "round24-warn"
    old = remember("Round24 relay is not enabled",
                   entity="Round24Relay", project=project)

    result = remember("Round24 relay is enabled",
                      entity="Round24Relay", project=project,
                      check_contradictions=True,
                      contradiction_mode="warn")

    assert result["contradictions"]["found"] is True
    assert result["contradictions"]["superseded"] == []
    assert result["contradictions"]["kept"]
    listed = list_memories(project=project, memory_id=old["id"])
    assert listed["ok"] is True


def test_nli_failure_kind_separates_budget_timeout_parse_and_outage():
    assert failure_kind("skipped: NLI time budget exhausted") == "budget"
    assert failure_kind("NLI unavailable: request timed out") == "timeout"
    assert failure_kind("unparseable JSON from the model") == "parse"
    assert failure_kind("NLI unavailable: connection refused") == "outage"
    assert failure_kind("inference down") == "error"
    assert failure_kind(None) == "unknown"
