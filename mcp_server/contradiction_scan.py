"""The one owner of a contradiction scan.

`contradictions_report` used to mix five concerns in one ~600 line function:
candidate loading, per-scope neighbour fetch, NLI scheduling with a wall-clock
budget, score filtering and envelope rendering. The write-side
`_check_contradictions` re-implemented the "nearest active memories" query.
This module owns all of it:

* `nearest_active` is the single definition of "which memories are candidates"
  for both the report and a write's `check_contradictions`.
* `judge_candidates` schedules the NLI calls, honours the deadline and retries
  partial batches.
* `report_from_judgement` turns the scores into pairs, failures and warnings.
* `plan_resolution` decides which side of a pair survives; `apply_resolution`
  only executes a plan that was already decided.

The scan never opens a connection: the caller passes a `MemoryStore` (or a
cursor, for the write path) and owns the transaction. Everything except the
candidate query and the final UPDATE is pure and can be tested with a fake
inference and a fake clock.
"""

from __future__ import annotations

import logging
import re
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from mcp_server.duplicates import shared_content_word as _shares_subject
from mcp_server.lifecycle import ACTIVE_ROW_SQL
from mcp_server.scope_catalog import _retrieval_tables

if TYPE_CHECKING:
    from mcp_server.store import MemoryStore

logger = logging.getLogger(__name__)

# `partial_error` of the /nli endpoint:
# "NLI unavailable for <failed> of <total> document(s): <reason>"
_NLI_PARTIAL_RE = re.compile(
    r"NLI unavailable for (?P<failed>\d+) of (?P<total>\d+) document\(s\): "
    r"(?P<reason>.*)", re.DOTALL)

CONTRADICTION_SCORE_THRESHOLD = 0.7
SKIPPED_BUDGET_REASON = "skipped: NLI time budget exhausted"
ALREADY_SUPERSEDED_REASON = "already superseded by an earlier pair in this run"


class ScanLoadError(RuntimeError):
    """The candidate query failed; the caller reports a query error."""


# --------------------------------------------------------------------------
# Value objects
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ScanRequest:
    """Everything one scan needs; the adapter already validated it."""

    db: MemoryStore
    project: str | None = None
    entity: str | None = None
    limit: int = 20
    budget_seconds: float = 40.0
    neighbors_per_fact: int = 6
    workers: int = 6


@dataclass(frozen=True)
class Judge:
    """One judged pair (fact vs. neighbour) before it becomes a result dict."""

    left_id: str
    right_id: str
    score: float
    reason: str | None = None


@dataclass(frozen=True)
class Neighbour:
    """An active memory that is close enough to a seed to be compared."""

    id: str
    content: str
    entity: str | None
    distance: float | None
    created_at: Any = None
    updated_at: Any = None


@dataclass
class CandidateGroup:
    """One seed fact and the neighbours it will be judged against."""

    fact: dict
    neighbours: list[Neighbour]


@dataclass
class ScanReport:
    """The scan's answer, independent of the tool envelope."""

    pairs: list[dict] = field(default_factory=list)
    failures: list[dict] = field(default_factory=list)
    budget_exhausted: bool = False
    scanned: int = 0
    warnings: list[str] = field(default_factory=list)
    candidates: int = 0
    timestamps: dict[str, Any] = field(default_factory=dict)
    entity: str | None = None


@dataclass
class ResolutionDecision:
    """One pair the plan has decided; `skip_reason` means "do not write"."""

    pair: dict
    older: str
    newer: str
    skip_reason: str | None = None


@dataclass
class ResolutionPlan:
    """The decisions of one `resolve="newest"` run, before any write."""

    decisions: list[ResolutionDecision] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    dry_run: bool = False


@dataclass
class Judgement:
    """Raw NLI outcome per candidate group, before it becomes a report."""

    scores_by_fact: dict[str, tuple[list[float], str | None, list[int]]] = \
        field(default_factory=dict)
    errors: list[tuple[str, str]] = field(default_factory=list)
    error_neighbour_ids: dict[str, list[str]] = field(default_factory=dict)
    skipped: set[str] = field(default_factory=set)
    budget: float = 0.0


# --------------------------------------------------------------------------
# Candidate selection (shared with the write-side contradiction check)
# --------------------------------------------------------------------------


def nearest_active(
    cur,
    *,
    project: str | None,
    embedding: list[float] | None,
    limit: int,
    content_hash: str | None = None,
    exclude_id: str | None = None,
    exclude_content: str | None = None,
    where: str = "",
    where_params: tuple | list = (),
    seen: set[str] | None = None,
    per_table: bool = False,
) -> list[Neighbour]:
    """Active memories of the scope nearest to `embedding` (or newest first).

    This is the only place that decides which rows compete for a candidate
    seat. The report scans one seed at a time (`per_table=True`: every scope
    table contributes up to `limit` neighbours, and the caller dedupes pairs);
    a write's check uses one budget across the whole scope (the default).

    `where` is a bare SQL condition (joined with AND) such as
    ``"entity = %s"``. `seen` is shared across passes so the entity-priority
    query does not return a row twice.
    """
    if limit <= 0:
        return []
    if seen is None:
        seen = set()

    rows: list[Neighbour] = []
    for table in _retrieval_tables(cur, project):
        remaining = limit if per_table else limit - len(rows)
        if remaining <= 0:
            break

        conditions: list[str] = []
        params: list[Any] = []
        if embedding is not None:
            select = ("id, content, entity, created_at, updated_at, "
                      "(embedding <=> %s::vector) AS distance")
            params.append(embedding)
            conditions.append("embedding IS NOT NULL")
            order = "ORDER BY distance ASC"
        else:
            select = "id, content, entity, created_at, updated_at, NULL AS distance"
            order = "ORDER BY created_at DESC"
        if exclude_id is not None:
            conditions.append("id != %s")
            params.append(exclude_id)
        if content_hash:
            conditions.append("content_hash <> %s")
            params.append(content_hash)
        elif exclude_content is not None:
            conditions.append("content <> %s")
            params.append(exclude_content)
        if where:
            conditions.append(where)
            params.extend(where_params)
        conditions.append(ACTIVE_ROW_SQL)

        cur.execute(
            f"SELECT {select} FROM {table} "
            f"WHERE {' AND '.join(conditions)} {order} LIMIT %s",
            params + [remaining],
        )
        for mid, content, entity, created_at, updated_at, distance in \
                cur.fetchall():
            if mid in seen:
                continue
            seen.add(mid)
            rows.append(Neighbour(
                id=mid,
                content=content,
                entity=entity,
                distance=None if distance is None else float(distance),
                created_at=created_at,
                updated_at=updated_at,
            ))
            if not per_table and len(rows) >= limit:
                break
        if not per_table and len(rows) >= limit:
            break
    return rows


# --------------------------------------------------------------------------
# NLI scheduling
# --------------------------------------------------------------------------


def judge_candidates(
    groups: list[CandidateGroup],
    nli: Callable[..., Any],
    *,
    budget: float,
    workers: int,
    clock: Callable[[], float] = time.monotonic,
) -> Judgement:
    """Judge every candidate group inside a wall-clock budget.

    One NLI call per fact. A bounded pool overlaps the round trips; results
    are consumed in candidate order and the scan stops as soon as the deadline
    passes. A targeted second pass retries only the documents that failed in
    an otherwise successful batch.
    """
    judgement = Judgement(budget=budget)
    if not groups:
        return judgement

    deadline = clock() + budget
    # Cap one request below the whole budget: a single slow reply must not
    # consume it all, so the pool can still judge other facts before the
    # deadline.
    request_cap = max(1.0, min(15.0, budget))
    workers = min(workers, len(groups))
    in_flight: dict[Any, CandidateGroup] = {}
    next_index = 0

    def _score_fact(group: CandidateGroup):
        return nli(
            group.fact["content"],
            [n.content for n in group.neighbours],
            query_entity=group.fact.get("entity"),
            document_entities=[n.entity for n in group.neighbours],
            timeout=request_cap,
            deadline=deadline,
        )

    def _submit_until_full(pool: ThreadPoolExecutor) -> None:
        nonlocal next_index
        while (len(in_flight) < workers
               and next_index < len(groups)
               and clock() < deadline):
            group = groups[next_index]
            next_index += 1
            in_flight[pool.submit(_score_fact, group)] = group

    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        _submit_until_full(pool)
        while in_flight:
            remaining = deadline - clock()
            if remaining <= 0:
                break
            done, _ = wait(
                list(in_flight), timeout=remaining,
                return_when=FIRST_COMPLETED)
            if not done:
                break
            for future in done:
                group = in_flight.pop(future)
                try:
                    judgement.scores_by_fact[group.fact["id"]] = future.result()
                except TimeoutError:
                    judgement.skipped.add(group.fact["id"])
                except Exception as e:  # noqa: BLE001 - reported as failure
                    judgement.errors.append((group.fact["id"], str(e)))
                    judgement.error_neighbour_ids[group.fact["id"]] = [
                        n.id for n in group.neighbours]
            _submit_until_full(pool)

        # Deadline reached: do not wait for the requests still in flight (one
        # may hold the socket for up to `request_cap`). Abandoned groups are
        # recorded instead of silently disappearing.
        for future, group in list(in_flight.items()):
            future.cancel()
            judgement.skipped.add(group.fact["id"])
        in_flight.clear()
        for group in groups[next_index:]:
            judgement.skipped.add(group.fact["id"])
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    if judgement.skipped:
        logger.warning(
            "contradictions_report stopped after %.1fs: %d of %d "
            "candidate group(s) were not judged",
            budget, len(judgement.skipped), len(groups))

    # One targeted retry inside the same deadline: transient single-document
    # failures are recovered instead of reporting a permanently partial scan.
    for group in groups:
        fact_id = group.fact["id"]
        if fact_id in judgement.skipped:
            continue
        if clock() >= deadline:
            if fact_id not in judgement.scores_by_fact:
                judgement.skipped.add(fact_id)
            continue
        scored = judgement.scores_by_fact.get(fact_id)
        if scored is None:
            indices = list(range(len(group.neighbours)))
        else:
            _scores, _partial, indices = scored
        if not indices:
            continue
        try:
            retry_scores, retry_partial, retry_failed = nli(
                group.fact["content"],
                [group.neighbours[i].content for i in indices],
                query_entity=group.fact.get("entity"),
                document_entities=[group.neighbours[i].entity for i in indices],
                timeout=request_cap,
                deadline=deadline,
            )
        except TimeoutError:
            if fact_id not in judgement.scores_by_fact:
                judgement.skipped.add(fact_id)
            continue
        except Exception as e:  # noqa: BLE001 - the first-pass error wins
            logger.warning("contradictions_report NLI retry failed: %s", e)
            continue
        merged = ([0.0] * len(group.neighbours) if scored is None
                  else list(scored[0]))
        for position, index in enumerate(indices):
            merged[index] = retry_scores[position]
        still_failed = [indices[position]
                        for position in retry_failed
                        if 0 <= position < len(indices)]
        judgement.scores_by_fact[fact_id] = (
            merged, retry_partial, still_failed)

    # A fact that the retry healed is not a failure anymore.
    judgement.errors = [
        (fact_id, error) for fact_id, error in judgement.errors
        if fact_id not in judgement.scores_by_fact
        and fact_id not in judgement.skipped]
    return judgement


# --------------------------------------------------------------------------
# Report rendering
# --------------------------------------------------------------------------


def _aggregate_nli_partials(partials: list[str]) -> str | None:
    """Fold per-call partial-error messages into one summary line.

    Each /nli call reports its own failures ("1 of 16", "1 of 15", ...), so a
    scope-wide scan can collect several nearly identical warnings. The
    aggregated line keeps the failures per distinct reason countable; messages
    that do not match the endpoint format are kept verbatim (deduplicated).
    """
    failed_by_reason: dict[str, int] = {}
    total_docs = 0
    for partial in partials:
        match = _NLI_PARTIAL_RE.fullmatch(partial)
        if not match:
            return None
        reason = match.group("reason").strip()
        failed = int(match.group("failed"))
        total_docs += int(match.group("total"))
        failed_by_reason[reason] = failed_by_reason.get(reason, 0) + failed
    total_failed = sum(failed_by_reason.values())
    parts = [f"{count} document(s): {reason}"
             for reason, count in failed_by_reason.items()]
    return (f"NLI unavailable for {total_failed} of {total_docs} "
            f"document(s) in total: " + " | ".join(parts))


def _append_partial_warning(warnings: list[str], partials: list[str]) -> None:
    """Append one aggregated (or, as fallback, the raw deduped) warnings."""
    if not partials:
        return
    summary = _aggregate_nli_partials(partials)
    if summary is not None:
        warnings.append(
            f"scan partially incomplete: {summary}; the gaps are not a "
            f"clean contradiction scan")
    else:
        warnings.extend(
            f"scan partially incomplete: {p}"
            for p in dict.fromkeys(partials))


def _partial_reason(partial: str | None) -> str | None:
    """The reason inside a /nli partial_error, without the counts."""
    if not partial:
        return None
    match = _NLI_PARTIAL_RE.fullmatch(partial)
    return match.group("reason").strip() if match else partial


def failure_kind(reason: str | None) -> str:
    """Classify an NLI failure so a caller can act on it.

    "NLI unavailable" alone hides the difference between a provider outage
    (retry later), a parse error (the model answered garbage) and a budget
    stop (narrow the scan). The reason text stays unchanged for backwards
    compatibility; `kind` is the machine-readable addition.
    """
    text = (reason or "").strip().lower()
    if not text:
        return "unknown"
    if "budget" in text or "time budget" in text:
        return "budget"
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if "json" in text or "parse" in text or "unparseable" in text:
        return "parse"
    if any(token in text for token in (
            "unavailable", "connection", "connect", "refused", "reset",
            "http", "429", "500", "502", "503", "504", "rate limit")):
        return "outage"
    return "error"


def _entities_differ(a: str | None, b: str | None) -> bool:
    """True when two memories are tagged with different entities.

    Both untagged counts as the same subject; one tagged and one untagged
    counts as different, because a recency-based resolution between them is
    as unsafe as between two differently tagged memories.
    """
    if a is None or b is None:
        return a is not b
    return a.strip().lower() != b.strip().lower()


def _contradiction_pair(fact: dict, neighbour: Neighbour,
                        judge: Judge) -> dict:
    return {
        "a": {"id": fact["id"], "content": fact["content"],
              "entity": fact.get("entity")},
        "b": {"id": neighbour.id, "content": neighbour.content,
              "entity": neighbour.entity},
        # Same entity (or both untagged) -> a recency-based resolution is
        # meaningful. Different entities -> two different subjects.
        "cross_entity": _entities_differ(fact.get("entity"), neighbour.entity),
        "confidence": round(judge.score, 3),
    }


def report_from_judgement(
    groups: list[CandidateGroup],
    judgement: Judgement,
    *,
    entity: str | None,
    limit: int,
    timestamps: dict[str, Any] | None = None,
) -> ScanReport:
    """Turn raw NLI scores into the report's pairs, failures and warnings."""
    pairs: list[dict] = []
    warnings: list[str] = []
    failures: list[dict] = []

    if judgement.errors:
        nli_failed = f"NLI unavailable: {judgement.errors[0][1]}"
        warnings.append(
            f"scan incomplete: {nli_failed} - the result is not a clean "
            f"contradiction scan; retry later")
        logger.warning("contradictions_report NLI failed: %s",
                       judgement.errors[0][1])
        for fact_id, error in judgement.errors:
            failures.append({
                "fact_id": fact_id,
                "neighbor_ids": judgement.error_neighbour_ids.get(fact_id, []),
                "reason": error,
                "kind": failure_kind(error)})

    if judgement.skipped:
        warnings.append(
            f"scan incomplete: the NLI time budget of {judgement.budget:.1f}s "
            f"was exhausted before {len(judgement.skipped)} of "
            f"{len(groups)} candidate fact group(s) could be judged; the "
            f"unchecked pairs are listed in `nli_failures` (reason "
            f"'{SKIPPED_BUDGET_REASON}'). Narrow the scan with entity=..., "
            f"lower limit or raise budget_seconds to check more")

    # A fact whose call failed or was skipped has no scores: its pairs were
    # not judged. Reported independently of `limit`, so a truncated
    # contradiction list never hides an unchecked pair.
    checked_pairs = 0
    for group in groups:
        scored = judgement.scores_by_fact.get(group.fact["id"])
        if scored is None:
            if group.fact["id"] in judgement.skipped:
                failures.append({
                    "fact_id": group.fact["id"],
                    "neighbor_ids": [n.id for n in group.neighbours],
                    "reason": SKIPPED_BUDGET_REASON,
                    "kind": "budget"})
            continue
        checked_pairs += len(group.neighbours) - len(scored[2])

    partials: list[str] = []
    for group in groups:
        if len(pairs) >= limit:
            break
        scored = judgement.scores_by_fact.get(group.fact["id"])
        if scored is None:
            continue
        scores, partial, failed_indices = scored
        if partial:
            # Collected here, aggregated into ONE warning after the loop: a
            # scope-wide check makes one /nli call per fact, and several
            # identical partial lines are harder to read than a summary.
            partials.append(partial)
        if failed_indices:
            reason = _partial_reason(partial) or "NLI unavailable"
            failures.append({
                "fact_id": group.fact["id"],
                "neighbor_ids": [group.neighbours[i].id
                                 for i in failed_indices
                                 if 0 <= i < len(group.neighbours)],
                "reason": reason,
                "kind": failure_kind(reason)})
        for neighbour, score in zip(group.neighbours, scores):
            if len(pairs) >= limit:
                break
            if score > CONTRADICTION_SCORE_THRESHOLD:
                pairs.append(_contradiction_pair(
                    group.fact, neighbour,
                    Judge(group.fact["id"], neighbour.id, score)))
    _append_partial_warning(warnings, partials)

    # An empty result must say *why* it is empty: no similar pair, or no
    # contradiction among the pairs that were checked.
    if not pairs and not warnings:
        scope_desc = (f"active memories of {entity!r}"
                      if entity else "active memories of the scope")
        warnings.append(
            f"no contradictions found: {checked_pairs} similar pair(s) of "
            f"{scope_desc} were checked with NLI")

    return ScanReport(
        pairs=pairs,
        failures=failures,
        budget_exhausted=bool(judgement.skipped),
        scanned=checked_pairs,
        warnings=warnings,
        candidates=len(groups),
        timestamps=dict(timestamps or {}),
        entity=entity,
    )


# --------------------------------------------------------------------------
# Resolution planning and application
# --------------------------------------------------------------------------


def plan_resolution(
    report: ScanReport,
    *,
    resolve: str,
    min_confidence: float,
    dry_run: bool,
) -> ResolutionPlan:
    """Decide which side of every resolvable pair survives.

    The decision uses the creation time, never `updated_at`: a `touch` must
    not promote an older fact to "newest". Cross-entity pairs and pairs
    without a shared content word are reported but never acted on.
    """
    if resolve != "newest" or not report.pairs:
        return ResolutionPlan(dry_run=dry_run)

    resolvable: list[dict] = []
    skipped_cross = 0
    skipped_subject = 0
    skipped_weak = 0
    for pair in report.pairs:
        if pair.get("cross_entity"):
            skipped_cross += 1
            continue
        if pair.get("confidence", 0.0) < min_confidence:
            skipped_weak += 1
            continue
        if not _shares_subject(pair["a"]["content"], pair["b"]["content"]):
            # A noisy NLI hit between unrelated statements must never
            # invalidate a good memory (a version fact was once superseded as
            # a "contradiction" of a deploy note).
            skipped_subject += 1
            continue
        resolvable.append(pair)

    warnings: list[str] = []
    if skipped_cross:
        warnings.append(
            f"resolve skipped for {skipped_cross} cross-entity pair(s): "
            f"the two facts belong to different entities, so recency does "
            f"not decide which one survives - inspect them manually or "
            f"re-run with an entity filter")
    if skipped_subject:
        warnings.append(
            f"resolve skipped for {skipped_subject} pair(s) without a "
            f"shared content word: the NLI score alone is not enough "
            f"evidence that both facts describe the same thing; inspect "
            f"them manually")
    if skipped_weak:
        warnings.append(
            f"resolve skipped for {skipped_weak} pair(s) below "
            f"resolve_min_confidence={min_confidence}: they are reported but "
            f"not acted on")

    decisions: list[ResolutionDecision] = []
    invalidated: set[str] = set()
    for pair in resolvable:
        a_id, b_id = pair["a"]["id"], pair["b"]["id"]
        a_ts = report.timestamps.get(a_id)
        b_ts = report.timestamps.get(b_id)
        keep_a = a_ts is not None and (b_ts is None or a_ts >= b_ts)
        older, newer = (b_id, a_id) if keep_a else (a_id, b_id)
        skip_reason = None
        if older in invalidated:
            skip_reason = ALREADY_SUPERSEDED_REASON
        else:
            invalidated.add(older)
        decisions.append(ResolutionDecision(pair, older, newer, skip_reason))

    return ResolutionPlan(decisions=decisions, warnings=warnings,
                          dry_run=dry_run)


def apply_resolution(
    plan: ResolutionPlan,
    db: MemoryStore | None,
    project: str | None,
) -> tuple[int, int]:
    """Execute a decided plan; return (resolved, would_resolve).

    A dry run touches no database at all. A real run updates only the older
    row of each decision and reports a decision whose row was already
    invalidated as skipped.
    """
    resolved = 0
    would_resolve = 0
    if not plan.decisions:
        return resolved, would_resolve
    if plan.dry_run:
        for decision in plan.decisions:
            decision.pair["resolution"] = (
                _skipped_resolution(decision, dry_run=True)
                if decision.skip_reason else {
                    "superseded": decision.older,
                    "kept": decision.newer,
                    "dry_run": True,
                })
            if not decision.skip_reason:
                would_resolve += 1
        return resolved, would_resolve

    tables = _retrieval_tables(db.cur, project)
    for decision in plan.decisions:
        if decision.skip_reason:
            decision.pair["resolution"] = _skipped_resolution(decision)
            continue
        changed = 0
        for table in tables:
            changed += db.execute(
                f"UPDATE {table} SET superseded_by = %s, "
                "updated_at = NOW() WHERE id = %s "
                "AND superseded_by IS NULL AND deleted_at IS NULL",
                (decision.newer, decision.older))
        if changed:
            resolved += 1
            decision.pair["resolution"] = {
                "superseded": decision.older,
                "kept": decision.newer,
            }
        else:
            decision.pair["resolution"] = _skipped_resolution(decision)
    return resolved, would_resolve


def _skipped_resolution(decision: ResolutionDecision,
                        dry_run: bool = False) -> dict:
    resolution = {
        "superseded": decision.older,
        "kept": decision.newer,
        "skipped": decision.skip_reason or ALREADY_SUPERSEDED_REASON,
    }
    if dry_run:
        resolution["dry_run"] = True
    return resolution


# --------------------------------------------------------------------------
# The scan
# --------------------------------------------------------------------------


class ContradictionScan:
    """Load candidates, judge them, and plan a resolution."""

    def __init__(self, nli: Callable[..., Any],
                 clock: Callable[[], float] = time.monotonic):
        self._nli = nli
        self._clock = clock

    def scan(self, request: ScanRequest) -> ScanReport:
        # The budget is the wall-clock contract of the whole call, not only
        # of the NLI leg: loading candidates (a pgvector search per seed) sits
        # in front of the judge leg, so a budget that starts at the judge
        # call alone could push the total past the client's transport
        # timeout - the exact failure the budget exists to prevent.
        started = self._clock()
        loaded = self._load(request)
        # A small reserve covers result assembly and the resolution pass; the
        # judge leg gets the rest.
        remaining = request.budget_seconds - (self._clock() - started) - 1.5
        if loaded.warning is not None:
            return ScanReport(warnings=[loaded.warning],
                              candidates=0, entity=request.entity)
        if not loaded.groups:
            scope_desc = (f"active memories of {request.entity!r}"
                          if request.entity else "active memories of the scope")
            return ScanReport(
                warnings=[
                    f"no candidate pairs: no two {scope_desc} are similar "
                    f"enough to be worth comparing"],
                candidates=0, timestamps=loaded.timestamps,
                entity=request.entity)
        if remaining <= 0.5:
            remaining_for_judge = max(0.0, remaining)
            warnings = [
                f"scan incomplete: candidate loading consumed the NLI time "
                f"budget of {request.budget_seconds}s (remaining "
                f"{remaining_for_judge:.1f}s for judge calls); narrow the "
                f"scan with `entity`, lower `limit`, or raise "
                f"`budget_seconds`"
            ]
            failures = [{
                "fact_id": group.fact["id"],
                "neighbor_ids": [n.id for n in group.neighbours],
                "reason": SKIPPED_BUDGET_REASON,
                "kind": "budget",
            } for group in loaded.groups]
            return ScanReport(
                warnings=warnings,
                budget_exhausted=True,
                candidates=len(loaded.groups),
                failures=failures,
                timestamps=loaded.timestamps,
                entity=request.entity)

        try:
            judgement = judge_candidates(
                loaded.groups, self._nli,
                budget=remaining,
                workers=request.workers,
                clock=self._clock,
            )
            return report_from_judgement(
                loaded.groups, judgement,
                entity=request.entity, limit=request.limit,
                timestamps=loaded.timestamps)
        except Exception as e:  # noqa: BLE001 - an outage is surfaced loudly
            logger.warning("contradictions_report NLI failed: %s", e)
            return ScanReport(
                warnings=[
                    f"scan incomplete: NLI unavailable: {e} - the result is "
                    f"not a clean contradiction scan; retry later"],
                candidates=len(loaded.groups),
                timestamps=loaded.timestamps,
                entity=request.entity)

    @staticmethod
    def _unjudged_entry(group: CandidateGroup) -> dict:
        return {
            "fact_id": group.fact["id"],
            "neighbor_ids": [n.id for n in group.neighbours],
            "reason": SKIPPED_BUDGET_REASON,
            "kind": "budget",
        }

    def plan_resolution(self, report: ScanReport, *, resolve: str,
                        min_confidence: float,
                        dry_run: bool) -> ResolutionPlan:
        return plan_resolution(
            report, resolve=resolve, min_confidence=min_confidence,
            dry_run=dry_run)

    # -- loading ---------------------------------------------------------------

    def _load(self, request: ScanRequest) -> "LoadedCandidates":
        from mcp_server.search import SearchRequest, _search_memories

        db = request.db
        try:
            facts = _search_memories(db.cur, SearchRequest(
                query_embedding=None,
                query_text=None,
                entity=request.entity,
                limit=200,
                project=request.project,
                include_embedding=True,
                include_superseded=False,
            ))
            if request.entity is None and len(facts) < 2:
                # Only a scope-wide scan can bail early: with an entity a
                # single seed still competes against untagged neighbours of
                # the scope, and a tagged/untagged conflict must stay visible.
                return LoadedCandidates(warning=(
                    "nothing to compare: the scope has "
                    f"{len(facts)} active memory/memories - a contradiction "
                    f"needs at least two"))

            tables = _retrieval_tables(db.cur, request.project)
            if request.entity is not None and len(facts) < 2:
                # The seeds alone are not comparable, but the scope may still
                # hold untagged neighbours: only bail when the scope has no
                # other row to compare against at all.
                scope_active = 0
                for table in tables:
                    db.cur.execute(
                        f"SELECT COUNT(*) FROM {table} "
                        "WHERE embedding IS NOT NULL AND " + ACTIVE_ROW_SQL)
                    scope_active += db.cur.fetchone()[0]
                if scope_active < 2:
                    return LoadedCandidates(warning=(
                        "nothing to compare: the scope has "
                        f"{scope_active} active memory/memories - a "
                        f"contradiction needs at least two"))

            return self._load_candidates(request, facts)
        except Exception as e:
            raise ScanLoadError(str(e)) from e

    def _load_candidates(self, request: ScanRequest,
                         facts: list[dict]) -> "LoadedCandidates":
        db = request.db
        candidates: dict[str, CandidateGroup] = {}
        checked: set[tuple[str, str]] = set()
        fact_ids = {f["id"] for f in facts}
        timestamps: dict[str, Any] = {}
        for fact in facts:
            timestamps.setdefault(fact["id"],
                                  fact.get("created_at") or fact.get("updated_at"))

        # The self-join only sees the entity when one was given: a
        # scope-wide scan must be able to report pairs across entities.
        # With an entity, untagged neighbours stay in the candidate pool:
        # a fact stored without an entity can conflict with the entity's
        # facts, and an entity-scoped scan that hid such a pair would make
        # the report look clean while the conflict is still active.
        where = ("(entity = %s OR entity IS NULL)" if request.entity else "")
        where_params = (request.entity,) if request.entity else ()

        for fact in facts:
            if not fact.get("embedding"):
                continue
            if len(candidates) >= request.limit:
                break
            found: list[Neighbour] = []
            for row in db.nearest(
                    project=request.project,
                    embedding=fact["embedding"],
                    limit=request.neighbors_per_fact,
                    content_hash=fact.get("content_hash"),
                    exclude_content=(None if fact.get("content_hash")
                                     else fact["content"]),
                    exclude_id=fact["id"],
                    where=where,
                    where_params=where_params,
                    per_table=True):
                # Seeds are the entity's memories; an untagged neighbour is
                # not in `fact_ids` but is a valid counterpart.
                if row.id not in fact_ids and row.entity is not None:
                    continue
                pair = tuple(sorted([fact["id"], row.id]))
                if pair in checked:
                    continue
                checked.add(pair)
                found.append(row)
                timestamps.setdefault(
                    row.id, row.created_at or row.updated_at)
            if found:
                # Judge the most similar pairs first: a budget cut must hit
                # the least suspicious groups, not the near-identical ones
                # that are most likely to be a real contradiction.
                found.sort(key=lambda item: item.distance)
                candidates[fact["id"]] = CandidateGroup(fact, found)

        groups = list(candidates.values())
        groups.sort(key=lambda group: group.neighbours[0].distance)
        return LoadedCandidates(groups=groups, timestamps=timestamps)


@dataclass
class LoadedCandidates:
    groups: list[CandidateGroup] = field(default_factory=list)
    timestamps: dict[str, Any] = field(default_factory=dict)
    warning: str | None = None
