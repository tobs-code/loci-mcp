"""Regression tests for the twelfth round of fixes (audit follow-up).

Covers:
- `expires_at` TTL: an expired memory disappears from active reads and is
  not a duplicate candidate anymore, while exports still keep the row;
- the ISO 8601 validation of `expires_at`.

TTL behaviour is implemented centrally via `ACTIVE_ROW_SQL`, so a couple of
representative read paths (list, recall, dedupe) are sufficient.
"""

from datetime import datetime, timedelta, timezone

from mcp_server.server import (
    export_memories,
    list_memories,
    recall,
    remember,
    remember_batch,
)


FUTURE = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
PAST = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()


def test_remember_rejects_invalid_expires_at():
    result = remember("ttl probe", project="ttl-probe", expires_at="soon")
    assert result["ok"] is False
    assert "ISO 8601" in result["error"]


def test_unexpired_memory_still_active():
    remember("The billing service settles nightly",
             project="ttl-probe", expires_at=FUTURE)

    listed = list_memories(project="ttl-probe")
    assert listed["count"] == 1
    assert listed["results"][0]["expires_at"] is not None
    assert recall(query="billing service settles",
                  project="ttl-probe")["count"] == 1


def test_expired_memory_hidden_from_active_reads_but_exported():
    remember("The reporting cache is refreshed hourly",
             project="ttl-probe", expires_at=PAST)
    remember("The reporting cache is refreshed every two hours",
             project="ttl-probe", expires_at=FUTURE)

    listed = list_memories(project="ttl-probe")
    active = [r["content"] for r in listed["results"]]
    assert active == ["The reporting cache is refreshed every two hours"]

    searched = recall(query="reporting cache refresh interval",
                      project="ttl-probe")
    assert searched["count"] == 1
    assert "hourly" not in searched["results"][0]["content"]

    # An expired row is not a duplicate candidate: re-storing exact content
    # is a fresh write, not a duplicate answer.
    again = remember("The reporting cache is refreshed hourly",
                     project="ttl-probe")
    assert again["status"] == "stored"
    assert again["id"] is not None

    # Export keeps expired rows (a backup never loses rows silently).
    exported = export_memories(project="ttl-probe")
    contents = [r["content"] for r in exported["data"]["records"]]
    assert "The reporting cache is refreshed hourly" in contents


def test_batch_expires_at_per_fact():
    result = remember_batch(
        project="ttl-probe",
        facts=[{"content": "Feature flags rotate weekly", "expires_at": PAST},
               {"content": "Feature flags rotate monthly", "expires_at": FUTURE}])

    assert result["ok"] is True
    assert result["status"] == "ok"
    listed = list_memories(project="ttl-probe",
                           contains="Feature flags")
    active = [r["content"] for r in listed["results"]]
    assert active == ["Feature flags rotate monthly"]
