"""Regression tests for the NLI time budget of `contradictions_report`.

A slow or rate-limited provider used to make the MCP call outlive the
transport timeout: the caller got neither the pairs judged so far nor a
machine-readable failure list. The scan now has a deadline, bounded
concurrency and reports unchecked pairs in `nli_failures`.
"""

import time

from mcp_server.inference_client import nli_detailed
from mcp_server.server import contradictions_report, remember


def _slow_nli(*args, **kwargs):
    time.sleep(1.0)
    documents = kwargs.get("documents") or (args[1] if len(args) > 1 else [])
    return [0.0] * len(documents), None, []


def test_contradictions_report_stops_at_the_deadline_and_names_unchecked_pairs(
        monkeypatch):
    entity = "BudgetProbe"
    ids = [
        remember(f"{entity} service listens on port {port}", entity=entity,
                 project="budget-probe", dedupe="off")["id"]
        for port in (8000, 8100, 8200, 8300)
    ]

    monkeypatch.setattr(
        "mcp_server.tools.maintenance.nli_detailed", _slow_nli)

    started = time.monotonic()
    report = contradictions_report(entity=entity, project="budget-probe",
                                   budget_seconds=0.5)
    elapsed = time.monotonic() - started

    assert report["ok"] is True
    assert report["budget_exhausted"] is True
    assert report["count"] == 0
    assert elapsed < 10, "the scan must not wait for every slow call"
    failures = report["nli_failures"]
    assert failures
    assert all(entry["reason"] == "skipped: NLI time budget exhausted"
               for entry in failures)
    unchecked = {entry["fact_id"] for entry in failures}
    assert unchecked <= set(ids)
    assert any("time budget" in warning for warning in report["warnings"])


def test_nli_detailed_honours_an_expired_deadline():
    try:
        nli_detailed("A", ["B"], deadline=time.monotonic() - 1)
    except TimeoutError as exc:
        assert "budget" in str(exc).lower()
    else:  # pragma: no cover - the call must fail without a request
        raise AssertionError("an expired deadline must abort before the HTTP call")
