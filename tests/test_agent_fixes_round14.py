"""Regression tests for the fifteenth round of fixes.

Covers:
- `recall` names the strongest vector hit the similarity filter rejected
  (`best_dropped_by_similarity`) and hints at it in the empty-result warning;
- `remember(links=[...])` stores a fact and creates its graph edges in one
  call, also on the duplicate path;
- malformed `links` are an argument error before anything is stored;
- `classify` retries transient HTTP states with linear backoff and gives up
  immediately on client errors;
- the superseded counter counts rows that are both superseded and soft-
  forgotten (history and soft delete are independent markers).
"""

import pytest

from inference.llm import classify, load_config
from mcp_server.server import (
    find_connections,
    forget,
    init_context,
    recall,
    remember,
)


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


def _chat_payload(score=0.9, label="contradiction"):
    return {
        "choices": [{
            "message": {
                "content": f'{{"label": "{label}", "score": {score}}}',
            },
        }],
    }


def test_recall_names_best_dropped_vector_hit():
    stored = remember("The rollout coordinator pauses canary at step 7",
                      project="dropped-hint", entity="Rollout")
    # An unrelated question shares no full-text lexeme with the fact, so the
    # only candidate is the vector hit; it stays far below the default
    # threshold and the answer names it as the best dropped hit.
    result = recall("baltic sea weather forecast tonight",
                    project="dropped-hint")
    assert result["ok"] is True
    assert result["results"] == []
    best = result.get("best_dropped_by_similarity")
    assert best, "the strongest rejected vector hit must be reported"
    assert best["id"] == stored["id"]
    assert "scope" in best and "similarity" in best
    hint = " ".join(result["warnings"])
    assert "strongest rejected vector hit" in hint
    assert stored["id"] in hint
    assert "min_similarity=null" in hint


def test_best_dropped_hint_includes_content_preview():
    remember("The graph filler fact has a distinctive preview sentence",
             project="dropped-preview")
    result = recall("mountain hiking gear list",
                    project="dropped-preview")
    assert result["results"] == []
    hint = " ".join(result["warnings"])
    assert "preview sentence" in hint


def test_remember_links_store_fact_and_graph_edges_in_one_call():
    stored = remember("The connector registry ships with the gateway",
                      project="remember-links", entity="Gateway",
                      links=[{"to_entity": "Connector Registry",
                              "relation_type": "SHIPS_WITH"}])
    assert stored["ok"] is True
    assert stored["status"] == "stored"
    links = stored.get("links")
    assert links and links[0]["status"] in ("created", "updated")

    edges = find_connections("Gateway", project="remember-links")
    assert any(e["entity"] == "Connector Registry"
               and e["relation"] == "SHIPS_WITH"
               for e in edges["results"])


def test_remember_duplicate_still_applies_links():
    first = remember("The audit trail retention is 90 days",
                     project="remember-links-dup", entity="AuditTrail",
                     links=[{"to_entity": "Retention",
                             "relation_type": "USES"}])
    assert first["ok"] is True
    again = remember("The audit trail retention is 90 days",
                     project="remember-links-dup", entity="AuditTrail",
                     links=[{"to_entity": "Compliance",
                             "relation_type": "SATISFIES"}])
    assert again["ok"] is True
    assert again["status"] == "duplicate"
    links = again.get("links")
    assert links and links[0]["to_entity"] == "Compliance"

    edges = find_connections("AuditTrail", project="remember-links-dup")
    targets = {(e["entity"], e["relation"]) for e in edges["results"]}
    assert ("Retention", "USES") in targets
    assert ("Compliance", "SATISFIES") in targets


def test_remember_link_without_entity_needs_explicit_from():
    stored = remember("The orphan fact lives alone",
                      project="remember-links",
                      links=[{"to_entity": "Alone",
                              "relation_type": "NEEDS_FROM"}])
    assert stored["ok"] is False
    assert "links[0]" in stored["error"]
    assert "from_entity" in stored["error"]


def test_remember_malformed_links_are_an_argument_error():
    stored = remember("The fact with a bad link", project="remember-links",
                      entity="BadLinks",
                      links=[{"to_entity": "X", "relation_type": "bad type!"}])
    assert stored["ok"] is False
    assert "links[0]" in stored["error"]
    # Nothing must have been stored on a failed argument validation.
    assert stored["id"] is None

    not_a_list = remember("Another fact", project="remember-links",
                          links="gateway")
    assert not_a_list["ok"] is False
    assert "links must be a list" in not_a_list["error"]

    bad_weight = remember("The fact with a bad weight",
                          project="remember-links", entity="Weights",
                          links=[{"to_entity": "X", "relation_type": "WEIGHS",
                                  "weight": 3}])
    assert bad_weight["ok"] is False
    assert "weight" in bad_weight["error"]


def test_explicit_from_entity_overrides_the_default_one():
    stored = remember("The mapper fact points elsewhere",
                      project="remember-links", entity="DefaultSource",
                      links=[{"from_entity": "ExplicitSource",
                              "to_entity": "Target",
                              "relation_type": "POINTS_TO"}])
    assert stored["ok"] is True
    edges = find_connections("ExplicitSource", project="remember-links")
    assert any(e["entity"] == "Target" and e["relation"] == "POINTS_TO"
               for e in edges["results"])


def test_classify_retries_transient_http_with_backoff(monkeypatch):
    sleeps = []

    def fake_sleep(seconds):
        sleeps.append(seconds)

    state = {"called": 0}

    def fake_post(url, headers=None, json=None, timeout=None):
        # One transient provider hiccup, then a clean answer.
        if state["called"] == 0:
            state["called"] = 1
            return FakeResponse(status_code=503, text="overloaded")
        return FakeResponse(payload=_chat_payload(score=0.6))

    monkeypatch.setattr("inference.llm.requests.post", fake_post)
    monkeypatch.setattr("inference.llm.time.sleep", fake_sleep)
    config = load_config({"LLM_PROVIDER": "openai", "LLM_API_KEY": "sk-test"})

    assert classify(config, "A", "B") == 0.6
    assert len(sleeps) == 1
    assert sleeps[0] > 0


def test_classify_gives_up_immediately_on_client_error(monkeypatch):
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(True)
        return FakeResponse(status_code=401, text="unauthorized")

    monkeypatch.setattr("inference.llm.requests.post", fake_post)
    monkeypatch.setattr("inference.llm.time.sleep", lambda s: None)
    config = load_config({"LLM_PROVIDER": "openai", "LLM_API_KEY": "wrong"})

    with pytest.raises(RuntimeError, match="LLM request failed"):
        classify(config, "A", "B")
    # 4xx never recovers: no repeated attempts, straight through the
    # format fallback so all three formats fail fast.
    assert len(calls) == 3


def test_superseded_counter_counts_forgotten_rows_too():
    first = remember("The counter old value is 10", project="counter-fix")
    remember("The counter new value is 20", project="counter-fix",
             supersede=first["id"])
    # Soft-delete the superseded row: it is BOTH superseded and forgotten.
    forget(first["id"], project="counter-fix", soft=True)

    boot = init_context(project="counter-fix")
    stats = [s for s in boot["project_stats"] if s["project"] == "counter-fix"]
    assert stats, "the scope must be covered by the bootstrap"
    scope = stats[0]
    # Both markers are independent: the row carries superseded_by AND
    # deleted_at, and is counted in BOTH counters (not silently zero).
    assert scope["superseded_memories"] >= 1
    assert scope["forgotten_memories"] >= 1
    assert scope["memories"] == 1  # only the active replacement
