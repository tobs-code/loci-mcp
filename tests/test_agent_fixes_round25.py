"""Regression tests for the round-25 live-audit fixes.

The write-side `check_contradictions` used the thin `nli()` wrapper, which
drops `partial_error` and `failed_indices`: a candidate document the provider
could not judge silently read as a 0.0 "neutral" score, so the write reported
`contradictions.found=False` while a contradicting pair stayed active. The
report path always surfaced this as `nli_failures`. Now `_check_contradictions`
uses `nli_detailed` and raises on a partial payload, so the write surfaces
"contradiction check skipped: NLI unavailable: ..." and keeps the fact - the
documented outage behaviour, extended to partial outages.
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


def test_partial_nli_write_surfaces_the_skip(monkeypatch):
    """A partial failure must not read the unjudged pair as "neutral".

    Live audit round-1 scenario: two teensy-differing facts (a port number)
    stayed BOTH active because the NLI call for the pair silently failed.
    """
    remember(content="The staging redis runs a 6380 cache on host alpha.",
             entity="Redis", project="nli-partial-write")
    monkeypatch.setattr(
        "mcp_server.storage.nli_detailed",
        _partial_nli(["0.9"],
                     "NLI unavailable for 1 of 2 document(s): "
                     "provider hiccup mid-batch",
                     [1])
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
    assert payload["skipped"].startswith("NLI unavailable:")
    assert "1 of 2 document(s)" in payload["skipped"]
    assert "provider hiccup mid-batch" in payload["skipped"]
    warnings = " ".join(result["warnings"])
    assert "contradiction check skipped" in warnings
    assert "NLI unavailable" in warnings
    # The fact itself is kept: the check failed, the write did not fail.
    assert not payload["superseded"]
    assert not payload["kept"]


def test_partial_nli_batch_entry_surfaces_the_skip(monkeypatch):
    remember(content="The staging redis batches exports on host alpha.",
             entity="Redis", project="nli-partial-batch")
    monkeypatch.setattr(
        "mcp_server.storage.nli_detailed",
        _partial_nli(["0.9"],
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
    assert entry["contradictions"]["skipped"].startswith("NLI unavailable:")
    assert "1 of 1 document(s)" in entry["contradictions"]["skipped"]
    assert any("contradiction check skipped" in w
               for w in entry.get("warnings", []))


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
