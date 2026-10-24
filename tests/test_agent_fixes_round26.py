"""Regression tests for the round-26 live-audit fixes.

1. `remember_batch` honours per-fact `links`: the edges are created with the
   fact, reported per entry, a malformed link is a per-fact prepared failure
   (not a crash), and a fact without links carries no `links` payload.
2. `get_entity` on a miss names the other scopes that hold the entity, so a
   cross-scope lookup no longer needs a second discovery round trip.
3. The contradiction budget is the wall-clock contract of the whole call:
   candidate loading eats into it, and a load that consumed the budget
   returns early with `budget_exhausted` before a single NLI call.
4. Failed write-tool calls answer `id: null` (the envelope contract), never
   the caller's memory id.
"""

import time

from mcp_server.contradiction_scan import (
    CandidateGroup,
    ContradictionScan,
    LoadedCandidates,
    Neighbour,
    ScanRequest,
    SKIPPED_BUDGET_REASON,
)
from mcp_server.graph import find_connections
from mcp_server.tools.memory import forget, remember, remember_batch, update_memory
from mcp_server.tools.read import get_entity


def test_batch_fact_creates_its_links():
    result = remember_batch(
        project="batch-links-26",
        facts=[
            {"content": "The API service stores its sessions in the Redis "
                        "cache.",
             "entity": "API",
             "links": [{"to_entity": "Redis", "relation_type": "USES",
                        "context": "session storage"}]},
            {"content": "The API service keeps its relational data in "
                        "Postgres.", "entity": "API"},
        ],
    )
    assert result["ok"] is True, result.get("error")
    first, second = result["results"][0], result["results"][1]
    assert first["status"] == "stored", first
    edges = first["links"]
    assert len(edges) == 1
    edge = edges[0]
    assert edge["from_entity"] == "API"
    assert edge["to_entity"] == "Redis"
    assert edge["relation_type"] == "USES"
    assert edge["status"] == "created"
    # A fact without links reports none - the payload stays honest.
    assert "links" not in second
    # The edge really exists in the project graph.
    conns = find_connections(entity="API", project="batch-links-26",
                             relation_type="USES")
    assert conns["ok"] is True
    assert any(c["to_entity"] == "Redis" and c["relation"] == "USES"
               for c in conns["results"])


def test_batch_malformed_link_is_a_per_fact_failure():
    result = remember_batch(
        project="batch-links-26-bad",
        facts=[
            {"content": "Fact one carries a valid edge.", "entity": "QA",
             "links": [{"to_entity": "Targets", "relation_type": "HITS"}]},
            {"content": "Fact two carries a malformed edge.", "entity": "QA",
             "links": [{"to_entity": "Nowhere"}]},
        ],
    )
    assert result["ok"] is True
    good, bad = result["results"][0], result["results"][1]
    assert good["status"] == "stored"
    assert bad["status"] == "error"
    assert "relation_type" in bad["error"]
    assert bad["id"] is None
    assert result["succeeded"] == 1 and result["failed"] == 1


def test_get_entity_names_other_scopes_on_a_miss():
    remember(content="The build agent signs releases with key beta.",
             entity="BuildAgent", project="hint-scope-26")
    result = get_entity(entity="BuildAgent", project="hint-scope-26-other")
    assert result["ok"] is False
    assert result["status"] == "not_found"
    assert set(result["scopes_with_entity"]) == {"hint-scope-26"}
    assert result["error"] is not None
    assert "hint-scope-26" in result["error"]
    assert any("hint-scope-26" in w for w in result["warnings"])


class _SlowLoadScan(ContradictionScan):
    """A scan whose loading leg alone eats the whole budget."""

    def __init__(self):
        super().__init__(lambda *a, **k: _pytest_never_called())

    def _load(self, request):
        time.sleep(0.05)
        fact = {"id": "f1", "content": "seed content", "entity": None,
                "updated_at": "2026-01-01"}
        group = CandidateGroup(
            fact, [Neighbour(id="n1", content="neighbour content",
                             entity=None, distance=0.1)])
        return LoadedCandidates(groups=[group],
                                timestamps={"f1": "2026-01-01"})


def _pytest_never_called():
    raise AssertionError("no NLI call may follow an exhausted budget")


def test_budget_covers_loading_and_judging():
    # 1.0s budget, 1.5s reserve: after 0.5s(incl) of loading there is no
    # judge capacity left, so the scan must stop before any NLI call.
    report = _SlowLoadScan().scan(ScanRequest(db=object(),
                                              budget_seconds=1.0))
    assert report.budget_exhausted is True
    assert report.pairs == []
    assert report.failures == [{
        "fact_id": "f1",
        "neighbor_ids": ["n1"],
        "reason": SKIPPED_BUDGET_REASON,
        "kind": "budget",
    }]
    assert any("NLI time budget" in w for w in report.warnings)


def test_failed_writes_answer_id_null():
    result = forget(memory_id="00000000-nosuch-id")
    assert result["ok"] is False
    assert result["status"] == "not_found"
    assert result["id"] is None
    result = update_memory(memory_id="00000000-nosuch-id", content="x")
    assert result["ok"] is False
    assert result["status"] == "not_found"
    assert result["id"] is None
