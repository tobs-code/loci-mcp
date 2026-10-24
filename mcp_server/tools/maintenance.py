import json
import logging
import os
import time
from typing import Annotated, Any, Literal

from pydantic import Field

from mcp_server.config import (
    RERANK_ENABLED,
    RERANK_MODEL,
    SENTINEL_FORGOTTEN,
    _env_float,
    _env_int,
)
from mcp_server.contradiction_scan import (
    ContradictionScan,
    ScanLoadError,
    ScanRequest,
    apply_resolution,
)
from mcp_server.store import MemoryStore
from mcp_server import store_maintenance
from mcp_server.store_maintenance import (
    clear_supersede,
    expired_active_count,
    mirror_counts,
    prune_rows,
    supersede_links,
)
from mcp_server.duplicates import (  # noqa: F401 - re-exported for tests
    shared_content_word as _shares_subject,
)
from mcp_server.envelope import (
    STATUS_ERROR,
    STATUS_NOT_FOUND,
    STATUS_OK,
    fail,
    ok,
)
from mcp_server.inference_gateway import health_details, nli_detailed
from mcp_server.lifecycle import counts as lifecycle_counts
from mcp_server.scope_catalog import (
    _all_mem_tables,
    _mem_table,
    _norm_project,
    _project_names,
    _project_suffix,
    _scope_of_table,
    _table_exists,
)
from mcp_server.storage import clear_dangling_supersede
from mcp_server.version import STARTED_AT, VERSION, code_fingerprint

logger = logging.getLogger(__name__)


# A scope-wide scan performs one NLI call per candidate fact; without a
# deadline a slow or rate-limited provider makes the tool outlive the MCP
# transport timeout and the caller sees neither the pairs judged so far nor a
# machine-readable failure list. The budget keeps the call inside the
# transport window and turns the cut-off into `nli_failures` + a warning.
DEFAULT_SCAN_BUDGET_SECONDS = 40.0
DEFAULT_NEIGHBORS_PER_FACT = 6
DEFAULT_NLI_WORKERS = 6


def _scan_budget_seconds(budget_seconds: float | None) -> float:
    """The wall-clock budget for the NLI leg of a contradiction scan."""
    if budget_seconds is not None:
        return max(0.5, min(float(budget_seconds), 300.0))
    return _env_float(
        "CONTRADICTION_SCAN_BUDGET_SECONDS",
        DEFAULT_SCAN_BUDGET_SECONDS, 1.0, 300.0,
    )


def _neighbors_per_fact() -> int:
    return _env_int("CONTRADICTION_NEIGHBORS",
                    DEFAULT_NEIGHBORS_PER_FACT, 1, 20)


def _nli_workers() -> int:
    return _env_int("CONTRADICTION_NLI_WORKERS", DEFAULT_NLI_WORKERS, 1, 16)


def contradictions_report(entity: Annotated[
        str | None,
        Field(description="Entity to check. The scan covers this entity's "
                          "memories plus untagged (entity-less) memories of "
                          "the scope as their neighbours: a conflict between "
                          "a tagged and an untagged fact stays visible. "
                          "Without it every active memory in the scope is "
                          "cross-checked against its nearest neighbours "
                          "(pairs across entities included).")] = None,
    project: Annotated[
        str | None,
        Field(description="Project scope. Without it only global memories are "
                          "checked.")] = None,
    limit: Annotated[
        int,
        Field(description="Maximum number of candidate pairs to report "
                          "(default 20).")] = 20,
    budget_seconds: Annotated[
        float | None,
        Field(description="Wall-clock budget for the NLI leg (default: "
                          "CONTRADICTION_SCAN_BUDGET_SECONDS or 40s). When "
                          "it runs out, the scan stops early and the pairs "
                          "that were NOT judged are listed in `nli_failures` "
                          "with `skipped: NLI time budget exhausted` instead "
                          "of the call timing out.",
              ge=0.5, le=300.0)] = None,
    resolve: Annotated[
        Literal["none", "newest"],
        Field(description="'none' (default) only reports. 'newest' keeps the "
                          "most recently created memory of each SAME-ENTITY "
                          "pair active and marks the older one as superseded; "
                          "`touch` does not change that decision. "
                          "Cross-entity pairs are reported (cross_entity: "
                          "true) but never auto-resolved, because recency "
                          "says nothing about which fact is correct when the "
                          "entities differ. Pairs without a shared content "
                          "word and pairs below `resolve_min_confidence` are "
                          "also left untouched.")] = "none",
    dry_run: Annotated[
        bool,
        Field(description="With resolve='newest': report what would be "
                          "superseded (`would_resolve`, pair `resolution` "
                          "with dry_run=true) without changing anything. "
                          "Ignored for resolve='none'.")] = False,
    resolve_min_confidence: Annotated[
        float,
        Field(description="Minimum NLI confidence for an auto-resolve "
                          "(default 0.80). Higher than the reporting "
                          "threshold on purpose: a questionable pair may be "
                          "reported but must not invalidate a memory.",
              ge=0.0, le=1.0)] = 0.8,
) -> dict[str, Any]:
    """Find potentially conflicting facts stored for the same entity.

    Uses vector similarity to find candidate pairs, then NLI to confirm.
    Returns pairs with confidence scores in [0, 1] (higher = more likely contradictory).
    Without `entity` the scan is scope-wide: all active memories compete for
    the candidate slots, so conflicts between differently tagged entities are
    reported too. With `entity` the scan covers that entity's memories plus
    untagged neighbours of the scope, so a tagged/untagged conflict stays
    visible (cross-entity pairs are reported, never auto-resolved). With
    `resolve="newest"` the older memory of every confirmed
    pair of the SAME entity is superseded by the newer one (the report names
    the ids); "newest" is the creation time, so a `touch` never changes the
    winner. Cross-entity pairs stay untouched. An empty result says in
    `warnings` why it is empty: nothing to compare, no similar pair, or no
    contradiction among the checked pairs.
    When the NLI provider fails for some pairs, `nli_failures` names exactly
    which fact/neighbour ids were NOT judged (with the reason), so a partial
    scan is never mistaken for a clean one. The NLI leg has a wall-clock
    budget (`budget_seconds`, default 40s): when it is exhausted the scan
    stops early, returns the pairs judged so far, sets `budget_exhausted`
    and records the unchecked pairs in `nli_failures` with reason
    `skipped: NLI time budget exhausted`. Narrow the scan with `entity`,
    lower `limit` or raise the budget to check more.
    """
    logger.info("contradictions_report entity=%s limit=%s", entity, limit)
    started_at = time.monotonic()
    try:
        project = _norm_project(project)
    except ValueError as e:
        return fail(STATUS_ERROR, str(e))
    if entity is not None:
        entity = str(entity).strip() or None
    if not entity:
        logger.info("contradictions_report scanning the whole scope")
    inference = health_details()
    if inference is not None and inference.get("llm_configured") is False:
        return ok(STATUS_OK, count=0, results=[], warnings=[
            "contradiction check skipped: NLI provider is not configured"
        ], nli_configured=False)

    db = MemoryStore()
    scan = ContradictionScan(nli_detailed)
    try:
        report = scan.scan(ScanRequest(
            db=db,
            project=project,
            entity=entity,
            limit=limit,
            budget_seconds=_scan_budget_seconds(budget_seconds),
            neighbors_per_fact=_neighbors_per_fact(),
            workers=_nli_workers(),
        ))
    except ScanLoadError:
        logger.exception("contradictions_report scan failed")
        db.rollback()
        db.close()
        return fail(STATUS_ERROR, "query failed")

    contradictions = report.pairs
    warnings = list(report.warnings)
    resolved = 0
    would_resolve = 0
    plan = scan.plan_resolution(
        report, resolve=resolve, min_confidence=resolve_min_confidence,
        dry_run=dry_run)
    warnings.extend(plan.warnings)
    if plan.decisions:
        try:
            resolved, would_resolve = apply_resolution(plan, db, project)
            db.commit()
        except Exception as e:
            db.rollback()
            logger.exception("contradictions_report resolve failed")
            warnings.append(f"resolve failed: {e}")
        finally:
            db.close()
    else:
        db.commit()
        db.close()

    result: dict[str, Any] = {"count": len(contradictions),
                              "results": contradictions,
                              "resolved": resolved,
                              "checked_pairs": report.scanned,
                              "budget_exhausted": report.budget_exhausted,
                              "elapsed_seconds": round(
                                  time.monotonic() - started_at, 2),
                              "warnings": warnings}
    if resolve == "newest" and dry_run:
        result["would_resolve"] = would_resolve
        result["dry_run"] = True
    if report.failures:
        # Which pairs were NOT judged, per fact. A stale inference container
        # that does not send `failed_indices` leaves this empty; the
        # aggregated warning still names the incomplete scan then.
        # The per-neighbour list is intentionally complete (an agent may need
        # to retry exactly those ids), but the compact summary lets a caller
        # see the scale without reading every entry.
        result["nli_failures"] = report.failures
        result["nli_failures_count"] = len(report.failures)
        summary: dict[str, int] = {}
        kinds: dict[str, int] = {}
        for failure in report.failures:
            reason = str(failure.get("reason") or "unknown")
            summary[reason] = summary.get(reason, 0) + 1
            kind = str(failure.get("kind") or "unknown")
            kinds[kind] = kinds.get(kind, 0) + 1
        result["nli_failure_summary"] = summary
        # A provider outage, a parse error and an exhausted budget need
        # different reactions; the kind counts make that visible without
        # reading every failure entry.
        result["nli_failure_kinds"] = kinds
    return ok(STATUS_OK, **result)


def verify_integrity(
    project: Annotated[
        str | None,
        Field(description="Scope to check. Without it every scope is checked; "
                          "'global' checks only the shared scope.")] = None,
    fix: Annotated[
        bool,
        Field(description="Clear dangling/self-referencing supersede links. "
                          "Without it the tool only reports.")] = False,
    limit: Annotated[
        int,
        Field(description="Maximum number of problem rows to list (default 20).",
              ge=1, le=500)] = 20,
) -> dict[str, Any]:
    """Check stored memories for links that make rows unreachable.

    The main failure this catches: a memory is marked `superseded_by` an id
    that no longer exists (the replacement was hard-deleted, a project was
    dropped, or a partial import restored only one side). Such a row is hidden
    from every active read forever, although nothing supersedes it any more.
    `fix=true` clears those links so the row becomes readable history again;
    it never deletes content and never restores a valid supersede.

    Also reports self-references (`superseded_by = id`) and supersede chains
    (A replaced by B, B replaced by C), which are legal but often point at an
    unintended second write. Broken links and chains are listed in `results`;
    `chains` counts them even when `limit` truncates the list. Expired-but-active
    rows are counted as context, not as an error.

    With `fix=true` the counts and `results` describe the state AFTER the
    repair (`fixed` carries what was cleared), so a call reads as an answer
    to "is the scope clean now?".
    """
    logger.info("verify_integrity project=%s fix=%s", project, fix)
    try:
        name = _norm_project(project)
    except ValueError as e:
        return fail(STATUS_ERROR, str(e))
    only_global = str(project or "").strip().lower() == "global"
    db = MemoryStore()
    try:
        cur = db.cur
        all_tables = _all_mem_tables(cur)
        if only_global:
            tables = ["memories"]
        elif name:
            tables = [f"mem_{_project_suffix(name)}"]
        else:
            tables = all_tables
        tables = [t for t in tables if t in all_tables]
        if not tables:
            return fail(STATUS_ERROR, f"unknown project {name!r}")

        names = _project_names(cur)

        def _scan():
            # A supersede link may point into another scope (cross-scope
            # supersede), so validity is checked against every scope, not
            # only the selected one. The scan returns the full state; the
            # fix branch re-runs it so the report describes the state AFTER
            # the fix, not a stale pre-fix picture.
            all_ids = store_maintenance.all_ids(db, all_tables)

            details: list[dict[str, Any]] = []
            dangling: list[tuple[str, str, str]] = []
            self_refs: list[tuple[str, str, str]] = []
            links: dict[str, str] = {}
            link_rows: dict[str, tuple[str, str | None, str]] = {}
            total = 0
            for table in tables:
                scope = ("global" if table == "memories"
                         else names.get(table[len("mem_"):],
                                        table[len("mem_"):]))
                for mid, sup, entity, content in supersede_links(db, table):
                    total += 1
                    links[mid] = sup
                    link_rows[mid] = (scope, entity, content)
                    if sup == mid:
                        self_refs.append((table, mid, entity))
                    elif sup not in all_ids:
                        dangling.append((table, mid, entity))
                    if len(details) < limit and (sup == mid
                                                 or sup not in all_ids):
                        details.append({
                            "project": scope,
                            "id": mid,
                            "superseded_by": sup,
                            "entity": entity,
                            "content": content[:200],
                            "problem": ("self_reference" if sup == mid
                                        else "dangling_superseded_by"),
                        })
            return details, dangling, self_refs, links, link_rows, total

        def _chain_details(links, link_rows, details):
            # A chain (A -> B -> C) is legal after repeated supersede, but a
            # chain whose middle link is dangling is not: report the count
            # and list the rows so a caller can spot repeated rewrites of
            # one fact instead of only seeing a number.
            chain_links = [
                (mid, sup, links[sup]) for mid, sup in links.items()
                if sup in links and links.get(sup) != mid
            ]
            for mid, sup, chain_to in chain_links:
                if len(details) >= limit:
                    break
                scope, entity, content = link_rows[mid]
                details.append({
                    "project": scope,
                    "id": mid,
                    "superseded_by": sup,
                    "chain_to": chain_to,
                    "entity": entity,
                    "content": content[:200],
                    "problem": "supersede_chain",
                })
            return len(chain_links)

        details, dangling, self_refs, links, link_rows, total = _scan()
        expired_active = 0
        for table in tables:
            expired_active += expired_active_count(db, table)
        chains = _chain_details(links, link_rows, details)

        fixed = 0
        if fix and (dangling or self_refs):
            bad = [(table, mid) for table, mid, _ in dangling + self_refs]
            for table, mid in bad:
                clear_supersede(db, table, mid)
                fixed += 1
            db.commit()
            # Re-scan after the fix: the response keys
            # (`dangling_superseded_by`, `self_references`, `superseded_links`)
            # and `results` describe the state AFTER the repair, so a caller
            # can read the call as "is it clean now?" instead of a stale
            # pre-fix picture; `fixed` carries what was cleared. TTL rows are
            # unaffected by the fix, so `expired_active` keeps its count.
            details, dangling, self_refs, links, link_rows, total = _scan()
            chains = _chain_details(links, link_rows, details)
        else:
            db.commit()
    except Exception as e:
        db.rollback()
        return fail(STATUS_ERROR, str(e))
    finally:
        db.close()

    warnings: list[str] = []
    if fixed:
        # After a fix the scan reports the post-fix state (no dangling rows
        # left, so `dangling`/`self_refs` are empty); `fixed` says what the
        # repair cleared.
        warnings.append(
            f"cleared {fixed} broken supersede link(s); those memories "
            f"are readable history now")
    elif dangling or self_refs:
        warnings.append(
            f"{len(dangling) + len(self_refs)} broken supersede link(s) "
            f"hide rows from every read; call verify_integrity(fix=true) "
            f"to clear them")
    elif total and not chains:
        warnings.append("no integrity problems found")
    if chains:
        warnings.append(
            f"{chains} supersede chain(s) found: A replaced by B and B "
            f"replaced by C is legal, but verify the second replacement was "
            f"intended (see results)")
    if expired_active:
        warnings.append(
            f"{expired_active} active memory/memories have a passed TTL and "
            f"are hidden from reads (not an error; use include_expired=true "
            f"to inspect them)")
    issue_count = len(dangling) + len(self_refs) + chains
    if issue_count > len(details):
        warnings.append(
            f"results truncated at limit={limit}: {issue_count} problem "
            f"row(s) found, {len(details)} listed; raise limit to see more")
    return ok(
        STATUS_OK,
        checked_tables=tables,
        superseded_links=total,
        dangling_superseded_by=len(dangling),
        self_references=len(self_refs),
        chains=chains,
        expired_active=expired_active,
        fixed=fixed,
        results=details,
        warnings=warnings,
    )


def prune_memories(
    project: Annotated[
        str | None,
        Field(description="Scope to prune. Without it every scope is pruned; "
                          "'global' prunes only the shared scope.")] = None,
    older_than_days: Annotated[
        float,
        Field(description="Only remove inactive rows older than this many "
                          "days (default 30).",
              ge=0.0)] = 30.0,
    include_expired: Annotated[
        bool,
        Field(description="Remove rows whose TTL passed more than "
                          "`older_than_days` ago.")] = True,
    include_forgotten: Annotated[
        bool,
        Field(description="Remove soft-forgotten rows whose `deleted_at` is "
                          "older than `older_than_days`.")] = True,
    include_superseded: Annotated[
        bool,
        Field(description="Also remove superseded history older than "
                          "`older_than_days`. Off by default: history is "
                          "usually kept.")] = False,
    dry_run: Annotated[
        bool,
        Field(description="True (default) only reports what would be removed; "
                          "nothing is deleted.")] = True,
) -> dict[str, Any]:
    """Permanently remove inactive memories (expired/forgotten/superseded).

    Active memories are never touched: only rows that every read already
    hides are eligible. `dry_run=true` is the default, so an accidental call
    reports instead of deleting. Supersede links that point at a removed row
    are cleared, so surviving history stays readable.
    """
    logger.info("prune_memories project=%s older_than_days=%s dry_run=%s",
                project, older_than_days, dry_run)
    if not (include_expired or include_forgotten or include_superseded):
        return fail(STATUS_ERROR,
                    "nothing to prune: enable include_expired, "
                    "include_forgotten or include_superseded")
    try:
        project = _norm_project(project)
    except ValueError as e:
        return fail(STATUS_ERROR, str(e))
    cutoff_sql = "NOW() - (%s * INTERVAL '1 day')"
    conditions: list[str] = []
    params: list[Any] = []
    if include_expired:
        conditions.append(
            f"(expires_at IS NOT NULL AND expires_at <= {cutoff_sql})")
        params.append(older_than_days)
    if include_forgotten:
        conditions.append(
            f"(deleted_at IS NOT NULL AND deleted_at <= {cutoff_sql})")
        params.append(older_than_days)
    if include_superseded:
        conditions.append(
            f"(superseded_by IS NOT NULL AND superseded_by <> %s "
            f"AND deleted_at IS NULL "
            f"AND (expires_at IS NULL OR expires_at > NOW()) "
            f"AND updated_at <= {cutoff_sql})")
        params.extend([SENTINEL_FORGOTTEN, older_than_days])
    where = " OR ".join(conditions)

    db = MemoryStore()
    counts: dict[str, int] = {}
    total = 0
    warnings: list[str] = []
    try:
        cur = db.cur
        if project:
            tables = [_mem_table(project)]
            if not _table_exists(cur, tables[0]):
                db.rollback()
                return fail(STATUS_NOT_FOUND,
                            f"unknown project {project!r}: no memory table")
        else:
            tables = _all_mem_tables(cur)
        for table in tables:
            if not _table_exists(cur, table):
                continue
            removed = prune_rows(db, table, where, tuple(params),
                                 dry_run=dry_run)
            if removed:
                scope = _scope_of_table(cur, table)
                counts[scope] = counts.get(scope, 0) + removed
                total += removed
        if dry_run:
            db.rollback()
        else:
            clear_dangling_supersede(cur, tables)
            db.commit()
    except Exception as e:
        db.rollback()
        return fail(STATUS_ERROR, str(e))
    finally:
        db.close()

    scopes = list(counts)
    if len(scopes) > 1:
        warnings.append(
            f"prune covered several scopes ({', '.join(sorted(scopes))}); "
            f"pass project=<name> to restrict it")
    return ok(
        STATUS_OK,
        project=(project or None),
        counts={"memories": total, "by_scope": counts},
        dry_run=dry_run,
        older_than_days=older_than_days,
        removed_expired=include_expired,
        removed_forgotten=include_forgotten,
        removed_superseded=include_superseded,
        warnings=warnings,
        message=("Would remove" if dry_run else "Removed")
                + f" {total} inactive memor{'y' if total == 1 else 'ies'}",
    )


def server_health() -> dict[str, Any]:
    """Check if all services (Postgres, Inference) are healthy."""
    logger.info("server_health")
    pg_ok = False
    try:
        db = MemoryStore()
        db.ping()
        pg_ok = True
        db.commit()
        db.close()
    except Exception:
        pass
    inference_details = health_details()
    toolset = os.environ.get("MCP_TOOLSET", "full").strip().lower() or "full"
    if toolset not in ("core", "full"):
        toolset = "full"
    return ok(
        STATUS_OK,
        version=VERSION,
        code_fingerprint=code_fingerprint(),
        started_at=STARTED_AT,
        postgres=pg_ok,
        inference=inference_details is not None,
        nli_configured=(inference_details.get("llm_configured")
                        if inference_details is not None
                        and "llm_configured" in inference_details else None),
        rerank_enabled=RERANK_ENABLED,
        rerank_model=(inference_details.get("rerank_model")
                      if inference_details is not None
                      and "rerank_model" in inference_details else RERANK_MODEL),
        rerank_loaded=(inference_details.get("rerank_loaded")
                       if inference_details is not None else None),
        rerank_error=(inference_details.get("rerank_error")
                      if inference_details is not None else None),
        toolset=toolset,
    )


def stats_resource() -> str:
    """Memory server stats: counts, health."""
    db = MemoryStore()
    try:
        cur = db.cur
        tables = _all_mem_tables(cur)
        names = _project_names(cur)
        total = 0
        active = 0
        superseded = 0
        forgotten = 0
        for table in tables:
            table_counts = lifecycle_counts(cur, table)
            total += table_counts["total"]
            active += table_counts["active"]
            superseded += table_counts["superseded"]
            forgotten += table_counts["forgotten"]
        entities, relations = mirror_counts(db)
        db.commit()
    except Exception:
        return json.dumps({"error": "unavailable"})
    finally:
        db.close()

    return json.dumps({
        "version": VERSION,
        "code_fingerprint": code_fingerprint(),
        "started_at": STARTED_AT,
        "total_memories": total,
        "active_memories": active,
        "superseded_memories": superseded,
        "forgotten_memories": forgotten,
        # The shared scope is a scope like every other one, so it is listed
        # here too - the same list `init_context(project="*")` reports.
        "projects": sorted(
            [names.get(t[len("mem_"):], t[len("mem_"):]) for t in tables
             if t != "memories"] + (["global"] if "memories" in tables else [])
        ),
        "entities": entities,
        "relations": relations,
        "inference_healthy": _health_check(),
    }, indent=2)


def _health_check() -> bool:
    try:
        from mcp_server.inference_gateway import health as h
        return h()
    except Exception:
        return False
