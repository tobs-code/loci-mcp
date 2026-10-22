"""Regression tests for the eighth round of fixes.

Locks in the audit findings around cross-scope dedup, exact list totals, the
graph boost score, the `project="*"` wildcard and the forgotten sentinel in
exports.
"""

import pytest

from mcp_server.server import (
    export_memories,
    forget,
    get_entity,
    list_entities,
    list_memories,
    recall,
    relate,
    remember,
)


def test_superseded_row_is_not_hidden_by_an_active_copy():
    old_id = remember("Replay fact: port 9000", project="dedup-history")["id"]
    new_id = remember("Replay fact: port 9001", project="dedup-history",
                      supersede=old_id)["id"]
    copy_id = remember("Replay fact: port 9000", scope="global")["id"]

    listed = list_memories(project="dedup-history", include_superseded=True,
                           limit=20)
    ids = {r["id"] for r in listed["results"]}

    assert old_id in ids, "superseded history must not be hidden by a copy"
    assert new_id in ids
    assert copy_id in ids
    assert listed["count"] == listed["total"] == 3

    history = recall("Replay fact port 9000", project="dedup-history",
                     only_superseded=True, limit=10)
    assert old_id in {r["id"] for r in history["results"]}


def test_total_and_count_agree_when_a_global_copy_is_deduped():
    project_id = remember("Dup total: shared fact", project="dedup-total")["id"]
    remember("Dup total: shared fact", scope="global")

    listed = list_memories(project="dedup-total", limit=20)

    assert [r["id"] for r in listed["results"]] == [project_id]
    assert listed["count"] == listed["total"] == 1
    assert listed["has_more"] is False


def test_dedupe_off_copies_inside_one_scope_survive():
    first = remember("Dedupe off: the same fact twice", project="dedup-off",
                     dedupe="off")["id"]
    second = remember("Dedupe off: the same fact twice", project="dedup-off",
                      dedupe="off")["id"]

    listed = list_memories(project="dedup-off", limit=20)
    recalled = recall("the same fact twice", project="dedup-off", limit=20)

    assert {r["id"] for r in listed["results"]} == {first, second}
    assert listed["count"] == listed["total"] == 2
    assert {r["id"] for r in recalled["results"]} == {first, second}


def test_graph_boost_without_decay_does_not_shrink_relevance():
    remember("Boost contract: the cache expires at noon", entity="Cache",
             project="boost-contract")
    relate("Cache", "Database", "READS_FROM", project="boost-contract")

    result = recall("cache expires at noon", project="boost-contract",
                    graph_boost=True, graph_entity="Cache", limit=10)
    row = next(r for r in result["results"] if "cache expires" in r["content"])

    assert "_graph_boost" not in row
    assert row["_final_score"] == pytest.approx(row["_relevance_score"])


def test_project_star_lists_every_scope():
    remember("Wildcard: one project fact", entity="WildProject",
             project="wild-a")
    remember("Wildcard: one global fact", entity="WildGlobal", scope="global")

    memories = list_memories(project="*", limit=20)
    entities = list_entities(project="*", limit=20)
    project_only = list_entities(project="*", include_global=False, limit=20)

    assert memories["ok"] is True
    assert memories["total"] == 2
    assert not any("does not exist" in w for w in memories["warnings"])
    assert entities["ok"] is True
    assert not any("does not exist" in w for w in entities["warnings"])
    assert {r["project"] for r in project_only["results"]} == {"wild-a"}


def test_get_entity_rejects_the_star_wildcard_explicitly():
    remember("Wildcard: entity fact", entity="WildcardEntity",
             project="wild-b")

    result = get_entity("WildcardEntity", project="*")

    assert result["ok"] is False
    assert "not supported" in result["error"]


def test_same_id_in_two_scopes_are_separate_recall_rows():
    from mcp_server.server import import_memories

    mem_id = remember("Collision: alpha setting is on", project="coll-a")["id"]
    import_memories(
        {"records": [{"id": mem_id,
                      "content": "Collision: beta setting is off"}]},
        project="coll-b",
    )

    result = recall("Collision setting", project="*", limit=10,
                    min_similarity=None)
    rows = [r for r in result["results"] if r["id"] == mem_id]

    assert {r["content"] for r in rows} == {
        "Collision: alpha setting is on",
        "Collision: beta setting is off",
    }
    assert {r["project"] for r in rows} == {"coll-a", "coll-b"}


def test_nli_retries_once_on_an_unparseable_response(monkeypatch):
    import mcp_server.inference_client as ic

    calls: list[int] = []

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self.payload

    class FakeSession:
        def post(self, *args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                return FakeResponse({
                    "error": "response was not valid NLI JSON: bad format",
                })
            return FakeResponse({"contradiction_scores": [0.91]})

    monkeypatch.setattr(ic, "_get_session", lambda: FakeSession())

    scores, partial, failed = ic.nli_detailed("A", ["B"])

    assert scores == [0.91]
    assert partial is None
    assert failed == []
    assert len(calls) == 2


def test_export_does_not_leak_the_forgotten_sentinel():
    mem_id = remember("Export sentinel: temporary fact",
                      project="export-sentinel")["id"]
    forget(memory_id=mem_id, project="export-sentinel")

    payload = export_memories(format="json", project="export-sentinel",
                              include_superseded=True)
    record = next(r for r in payload["data"]["records"] if r["id"] == mem_id)

    assert record["deleted_at"] is not None
    assert record["superseded_by"] is None
