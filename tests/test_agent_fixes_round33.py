"""Regression tests for the round-33 fixes.

1. `remember_batch` accepts the documented per-fact contradiction options
   (`contradiction_mode`, `contradiction_min_confidence`,
   `max_contradiction_supersedes`) instead of rejecting them as unknown keys.
2. A per-fact `contradiction_mode="warn"` overrides the batch default and
   leaves the contradicted memory active.
"""

from mcp_server.server import list_memories, remember, remember_batch


def test_remember_batch_accepts_per_fact_contradiction_options():
    project = "round33-batch-options"
    result = remember_batch([{
        "content": "Round33 batch option fact",
        "contradiction_mode": "warn",
        "contradiction_min_confidence": 0.9,
        "max_contradiction_supersedes": 2,
    }], project=project)

    assert result["ok"] is True
    assert result["succeeded"] == 1
    assert result["results"][0]["status"] == "stored"


def test_remember_batch_per_fact_warn_mode_overrides_batch_default():
    project = "round33-batch-warn"
    old = remember("Round33 relay is not enabled",
                   entity="Round33Relay", project=project)

    result = remember_batch([{
        "content": "Round33 relay is enabled",
        "entity": "Round33Relay",
        "contradiction_mode": "warn",
    }], project=project, check_contradictions=True,
        contradiction_mode="supersede")

    contradictions = result["results"][0]["contradictions"]
    assert contradictions["found"] is True
    assert contradictions["superseded"] == []
    assert contradictions["kept"]

    listed = list_memories(project=project, memory_id=old["id"])
    assert listed["ok"] is True
