"""Round-35 fixes: honest update duplicates and visible cross-entity NLI.

1. `update_memory` refuses a content change that would duplicate an active
   row with `ok: false` + `applied: false`; `update_batch` flags the entry
   the same way. Previously it answered `ok: true`, so an agent could
   believe the edit was applied.
2. A contradiction supersede that invalidates ANOTHER entity's fact is
   named: the pair carries `cross_entity: true`, the payload sets
   `contradictions.cross_entity`, and a warning says whose fact was retired.
3. A near-duplicate warning must not claim "different subject" when NLI
   already confirmed a contradiction for that pair.
"""
from mcp_server.server import (
    get_entity,
    list_memories,
    remember,
    update_batch,
    update_memory,
)
from mcp_server.write_pipeline import _near_duplicate_warning


def test_update_duplicate_is_reported_as_not_applied():
    first = remember("round35 update probe: alpha")["id"]
    second = remember("round35 update probe: beta")["id"]

    result = update_memory(second, content="round35 update probe: alpha")

    assert result["ok"] is False
    assert result["status"] == "duplicate"
    assert result["applied"] is False
    assert result["duplicate_of"] == first
    assert result["memory_id"] == second
    stored = list_memories(memory_id=second)["results"][0]
    assert stored["content"] == "round35 update probe: beta"


def test_update_batch_duplicate_entry_is_not_applied():
    first = remember("round35 batch probe: alpha")["id"]
    second = remember("round35 batch probe: beta")["id"]

    result = update_batch([
        {"memory_id": second, "content": "round35 batch probe: alpha"},
        {"memory_id": first, "content": "round35 batch probe: gamma"},
    ])

    duplicate, updated = result["results"]
    assert duplicate["ok"] is False
    assert duplicate["applied"] is False
    assert duplicate["duplicate_of"] == first
    assert updated["status"] == "updated"


def test_cross_entity_contradiction_supersede_is_named():
    old = remember("Bob deploys to production every morning", entity="Bob")

    result = remember(
        "Bob does not deploy to production every morning",
        entity="Alice",
        check_contradictions=True,
    )

    payload = result["contradictions"]
    assert payload["found"] is True
    assert payload["cross_entity"] is True
    replaced = payload["superseded"]
    assert replaced
    assert replaced[0]["existing_id"] == old["id"]
    assert replaced[0]["existing_entity"] == "Bob"
    assert replaced[0]["cross_entity"] is True
    assert any("cross-entity supersede" in w and "Bob" in w
               for w in result["warnings"])


def test_near_duplicate_warning_respects_a_confirmed_contradiction():
    near_dup = {"id": "old-id", "entity": "production-db",
                "similarity": 0.913, "content": "The API listens on port 5432"}
    warning = _near_duplicate_warning(
        near_dup, entity="production-db", new_content="The API listens on 5433",
        contradiction_confidence=0.95)

    assert "share no subject term" not in warning
    assert "NLI confirms a contradiction" in warning
    assert "same subject" in warning


def test_unconfirmed_near_duplicate_keeps_the_topic_guard_wording():
    near_dup = {"id": "old-id", "entity": "production-db",
                "similarity": 0.80, "content": "Backups run at midnight"}
    warning = _near_duplicate_warning(
        near_dup, entity="production-db",
        new_content="The API listens on 5433")

    assert "different subject" in warning or "share no subject term" in warning


def test_get_entity_names_the_scope_deduped_copy():
    project = "round35-scope-dedupe"
    content = "Round35 shared fact: the worker pool has eight slots"
    remember(content, project=project, entity="Round35Worker")
    remember(content, scope="global", entity="Round35Worker")

    result = get_entity("Round35Worker", project=project)

    assert result["ok"] is True
    assert result["duplicates_hidden"] == 1, result
    assert any("hidden copy" in w for w in result["warnings"])
    # The page still describes exactly one active row per unique content.
    assert result["total"] == 1


def test_classify_many_judges_documents_concurrently_in_order(monkeypatch):
    import threading
    import time

    from inference import llm as inference_llm

    active = 0
    peak = 0
    lock = threading.Lock()

    def fake_classify(config, query, document, *, query_entity=None,
                      document_entity=None):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        if document == "boom":
            raise RuntimeError("provider hiccup")
        return {"doc1": 0.1, "doc2": 0.9, "boom": None, "doc3": 0.4}[document]

    monkeypatch.setattr(inference_llm, "classify", fake_classify)
    scores, failures, failed = inference_llm.classify_many(
        object(), "query", ["doc1", "doc2", "boom", "doc3"], max_workers=3)

    assert peak >= 2
    assert scores == [0.1, 0.9, 0.0, 0.4]
    assert failed == [2]
    assert failures == ["provider hiccup"]
