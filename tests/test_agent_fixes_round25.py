"""Regression tests for the round-25 live-audit fixes.

The write-side `check_contradictions` used the thin `nli()` wrapper, which
drops `partial_error` and `failed_indices`: a candidate document the provider
could not judge silently read as a 0.0 "neutral" score, so the write reported
`contradictions.found=False` while a contradicting pair stayed active.

Round-26 updates this: `_check_contradictions` now tolerates a partial NLI
failure. The candidates that WERE judged still produce contradictions, the
unjudged ones are reported as `contradictions.unjudged` (with the provider's
reason), and only a failure that leaves nothing judable raises into the
"contradiction check skipped" path.
"""

import pytest

from mcp_server.tools.memory import remember, remember_batch
from mcp_server.tools.read import get_entity


def _partial_nli(scores: list[float], partial: str | None,
                 failed: list[int]):
    def call(query, documents, *args, **kwargs):
        return scores, partial, failed
    return call


def _full_nli(scores: list[float]):
    def call(query, documents, *args, **kwargs):
        return scores, None, []
    return call


def test_partial_nli_write_reports_unjudged_candidates(monkeypatch):
    """A partial failure that still judged some candidates keeps the verdict.

    The unjudged candidate is reported as unverified (`contradictions.
    unjudged`), not silently read as "neutral" - and not turned into a
    whole-call skip either, so the judged pairs still count.
    """
    remember(content="The staging redis runs a 6380 cache on host alpha.",
             entity="Redis", project="nli-partial-write")
    monkeypatch.setattr(
        "mcp_server.storage.nli_detailed",
        _partial_nli([0.1],
                     "NLI unavailable for 1 of 1 document(s): "
                     "provider hiccup mid-batch",
                     [0])
    )
    result = remember(
        content="The staging redis runs a 6380 cache on host beta.",
        entity="Redis",
        project="nli-partial-write",
        check_contradictions=True,
    )
    assert result["ok"] is True
    assert result["status"] == "stored"
    payload = result["contradictions"]
    assert payload["found"] is False
    unjudged = payload["unjudged"]
    assert len(unjudged) == 1, "the unjudged candidate is named"
    assert "provider hiccup mid-batch" in str(unjudged[0]["reason"])
    warnings = " ".join(result["warnings"])
    assert "contradiction check incomplete" in warnings
    assert "unverified" in warnings
    assert "NLI unavailable" in warnings
    # The judged pair was neutral: nothing was invalidated.
    assert not payload["superseded"]
    assert not payload["kept"]


def test_partial_nli_write_still_counts_a_judged_contradiction(monkeypatch):
    """A judged contradiction survives a partial failure in the same call."""
    remember(content="The staging redis runs a 6380 cache on host alpha.",
             entity="Redis", project="nli-partial-judged")
    remember(content="The staging redis also mirrors a 6381 cache on host gamma.",
             entity="Redis", project="nli-partial-judged")
    monkeypatch.setattr(
        "mcp_server.storage.nli_detailed",
        _partial_nli([0.95, 0.1],
                     "NLI unavailable for 1 of 2 document(s): "
                     "provider hiccup mid-batch",
                     [1])
    )
    result = remember(
        content="The staging redis runs no cache on host beta.",
        entity="Redis",
        project="nli-partial-judged",
        check_contradictions=True,
    )
    payload = result["contradictions"]
    assert payload["found"] is True
    assert payload["superseded"], "the judged pair was auto-superseded"
    assert payload["unjudged"], "the unjudged pair is still reported"


def test_partial_nli_batch_entry_reports_unjudged_candidates(monkeypatch):
    remember(content="The staging redis batches exports on host alpha.",
             entity="Redis", project="nli-partial-batch")
    monkeypatch.setattr(
        "mcp_server.storage.nli_detailed",
        _partial_nli([0.1],
                     "NLI unavailable for 1 of 1 document(s): timed out",
                     [0])
    )
    result = remember_batch(
        facts=[{
            "content": "The staging redis batches exports on host beta.",
            "entity": "Redis",
        }],
        project="nli-partial-batch",
        check_contradictions=True,
    )
    assert result["ok"] is True
    entry = result["results"][0]
    assert entry["status"] == "stored"
    unjudged = entry["contradictions"]["unjudged"]
    assert len(unjudged) == 1
    assert "timed out" in str(unjudged[0]["reason"])
    assert any("contradiction check incomplete" in w
               for w in entry.get("warnings", []))
    assert all("skipped" not in w for w in entry.get("warnings", []))


def test_clean_nli_payload_still_finds_the_conflict(monkeypatch):
    """The full-payload path keeps working: a real contradiction supersedes."""
    remember(content="The staging redis runs a 6380 cache on host alpha.",
             entity="Redis", project="nli-clean-write")
    monkeypatch.setattr("mcp_server.storage.nli_detailed",
                        _full_nli([0.95, 0.0]))
    result = remember(
        content="The staging redis runs no cache on host alpha.",
        entity="Redis",
        project="nli-clean-write",
        check_contradictions=True,
    )
    assert result["ok"] is True
    payload = result["contradictions"]
    assert payload["found"] is True
    assert payload["superseded"], "auto-supersede replaced the old fact"
    assert "contradiction check skipped" not in " ".join(result["warnings"])


def test_partial_failure_keeps_both_memories_active(monkeypatch):
    """After a partial failure both facts stay readable - nothing vanished."""
    remember(content="The relay answers on port 5100.", entity="Relay",
             project="nli-partial-active")
    monkeypatch.setattr(
        "mcp_server.storage.nli_detailed",
        _partial_nli([], "NLI unavailable for 1 of 1 document(s): outage", [0])
    )
    result = remember(content="The relay answers on port 5201.", entity="Relay",
                      project="nli-partial-active", check_contradictions=True)
    assert result["status"] == "stored"
    entity = get_entity(entity="Relay", project="nli-partial-active")
    contents = " ".join(m["content"] for m in entity["memories"])
    assert "5100" in contents
    assert "5201" in contents
