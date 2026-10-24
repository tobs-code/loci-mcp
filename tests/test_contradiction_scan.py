"""Unit tests for the extracted contradiction scan.

The scan's scheduling and resolution rules are pure enough to test without a
database: `judge_candidates` needs only an NLI callable and a clock,
`report_from_judgement` turns the scores into the report, and
`plan_resolution`/`apply_resolution` decide and apply a resolve.
"""

from mcp_server.contradiction_scan import (
    ALREADY_SUPERSEDED_REASON,
    SKIPPED_BUDGET_REASON,
    CandidateGroup,
    ContradictionScan,
    Judgement,
    Neighbour,
    ScanReport,
    apply_resolution,
    judge_candidates,
    plan_resolution,
    report_from_judgement,
)


def _group(fact_id: str, *neighbours: Neighbour,
           entity: str | None = "Relay") -> CandidateGroup:
    return CandidateGroup(
        fact={
            "id": fact_id,
            "content": f"fact {fact_id}",
            "entity": entity,
            "embedding": [1.0, 0.0],
            "content_hash": f"hash-{fact_id}",
        },
        neighbours=list(neighbours),
    )


def _neighbour(mid: str, *, entity: str | None = "Relay",
               distance: float = 0.1) -> Neighbour:
    return Neighbour(id=mid, content=f"fact {mid}", entity=entity,
                     distance=distance)


class _SequenceClock:
    def __init__(self, values: list[float]):
        self._values = list(values)
        self._last = values[-1]

    def __call__(self) -> float:
        if self._values:
            self._last = self._values.pop(0)
        return self._last


def test_budget_exhaustion_names_every_unchecked_pair():
    groups = [
        _group("a", _neighbour("b"), _neighbour("c")),
        _group("d", _neighbour("e")),
    ]
    # First clock read computes the deadline, the second one already shows the
    # budget used up: nothing in flight can be judged.
    clock = _SequenceClock([0.0, 100.0])
    judgement = judge_candidates(
        groups, lambda *a, **k: ([0.0] * len(a[1]), None, []),
        budget=5.0, workers=2, clock=clock)

    report = report_from_judgement(
        groups, judgement, entity="Relay", limit=20)

    assert report.budget_exhausted is True
    assert {f["fact_id"] for f in report.failures} == {"a", "d"}
    unchecked = [nid for f in report.failures for nid in f["neighbor_ids"]]
    assert sorted(unchecked) == ["b", "c", "e"]
    assert all(f["reason"] == SKIPPED_BUDGET_REASON for f in report.failures)
    assert any("NLI time budget" in w and SKIPPED_BUDGET_REASON in w
               for w in report.warnings)


def test_nli_outage_is_reported_as_an_incomplete_scan():
    groups = [_group("a", _neighbour("b"))]

    def boom(*args, **kwargs):
        raise RuntimeError("inference down")

    judgement = judge_candidates(
        groups, boom, budget=10.0, workers=1, clock=lambda: 0.0)
    report = report_from_judgement(
        groups, judgement, entity="Relay", limit=20)

    assert report.budget_exhausted is False
    assert report.failures == [
        {"fact_id": "a", "neighbor_ids": ["b"], "reason": "inference down"}]
    assert any(w.startswith("scan incomplete: NLI unavailable: inference down")
               for w in report.warnings)


def test_scores_below_the_threshold_are_not_pairs():
    groups = [_group("a", _neighbour("b"))]
    judgement = Judgement(
        scores_by_fact={"a": ([0.69], None, [])}, budget=10.0)

    report = report_from_judgement(
        groups, judgement, entity="Relay", limit=20)

    assert report.pairs == []
    assert report.scanned == 1
    assert any("no contradictions found: 1 similar pair(s)" in w
               for w in report.warnings)


def _pair(a: str, b: str, *, confidence: float = 0.95,
          cross_entity: bool = False,
          a_content: str = "relay is enabled",
          b_content: str = "relay is not enabled") -> dict:
    return {
        "a": {"id": a, "content": a_content, "entity": "Relay"},
        "b": {"id": b, "content": b_content, "entity": "Relay"},
        "cross_entity": cross_entity,
        "confidence": confidence,
    }


def test_resolution_skips_cross_entity_weak_and_unrelated_pairs():
    report = ScanReport(
        pairs=[
            _pair("x1", "x2", cross_entity=True),
            _pair("w1", "w2", confidence=0.5),
            _pair("u1", "u2", a_content="alpha beta",
                  b_content="gamma delta"),
            _pair("a", "b"),
        ],
        timestamps={"a": 1, "b": 2},
    )

    plan = plan_resolution(
        report, resolve="newest", min_confidence=0.8, dry_run=True)

    assert len(plan.decisions) == 1
    assert plan.decisions[0].older == "a"
    assert plan.decisions[0].newer == "b"
    assert any("cross-entity" in w for w in plan.warnings)
    assert any("shared content word" in w for w in plan.warnings)
    assert any("resolve_min_confidence=0.8" in w for w in plan.warnings)


def test_dry_run_writes_nothing_and_counts_what_a_real_run_would_do():
    report = ScanReport(pairs=[_pair("a", "b")],
                        timestamps={"a": 1, "b": 2})
    plan = plan_resolution(
        report, resolve="newest", min_confidence=0.8, dry_run=True)

    resolved, would_resolve = apply_resolution(plan, None, None)

    assert (resolved, would_resolve) == (0, 1)
    assert report.pairs[0]["resolution"] == {
        "superseded": "a", "kept": "b", "dry_run": True}


def test_real_resolution_updates_only_the_older_row():
    calls: list[tuple] = []

    class _Db:
        cur = None

        def execute(self, sql, params):
            calls.append((sql, params))
            return 1

    report = ScanReport(pairs=[_pair("a", "b")],
                        timestamps={"a": 1, "b": 2})
    plan = plan_resolution(
        report, resolve="newest", min_confidence=0.8, dry_run=False)

    resolved, would_resolve = apply_resolution(plan, _Db(), None)

    assert (resolved, would_resolve) == (1, 0)
    assert _updates_only_older(calls, older="a", newer="b")
    assert report.pairs[0]["resolution"] == {"superseded": "a", "kept": "b"}


def _updates_only_older(calls, *, older: str, newer: str) -> bool:
    if not calls:
        return False
    for sql, params in calls:
        if "superseded_by = %s" not in sql:
            return False
        if params != (newer, older):
            return False
    return True


def test_already_invalidated_older_row_is_skipped_twice_in_one_plan():
    # Three facts, two pairs: `a` is the older side of both. Only the first
    # decision may write, the second one must be reported as already handled.
    report = ScanReport(
        pairs=[_pair("a", "b"), _pair("a", "c")],
        timestamps={"a": 1, "b": 2, "c": 3},
    )
    plan = plan_resolution(
        report, resolve="newest", min_confidence=0.8, dry_run=True)

    assert plan.decisions[0].skip_reason is None
    assert plan.decisions[1].skip_reason == ALREADY_SUPERSEDED_REASON

    _resolved, would_resolve = apply_resolution(plan, None, None)

    assert would_resolve == 1
    assert report.pairs[1]["resolution"]["skipped"] == ALREADY_SUPERSEDED_REASON
    assert report.pairs[1]["resolution"]["dry_run"] is True


def test_touch_does_not_decide_newest():
    # `b` was created later but `a` was touched last: created_at decides.
    report = ScanReport(
        pairs=[_pair("a", "b")],
        timestamps={"a": 5, "b": 4},
    )
    plan = plan_resolution(
        report, resolve="newest", min_confidence=0.8, dry_run=True)

    assert plan.decisions[0].older == "b"
    assert plan.decisions[0].newer == "a"


def test_scan_exposes_the_plan_helper():
    scan = ContradictionScan(lambda *a, **k: ([0.0], None, []))
    report = ScanReport(pairs=[_pair("a", "b")], timestamps={"a": 1, "b": 2})

    plan = scan.plan_resolution(
        report, resolve="newest", min_confidence=0.8, dry_run=True)

    assert len(plan.decisions) == 1
