import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Any, Literal

from pydantic import Field

from mcp_server.config import SENTINEL_FORGOTTEN
from mcp_server.db import get_conn
from mcp_server.envelope import STATUS_ERROR, STATUS_OK, fail, ok
from mcp_server.inference_client import health_details, nli_detailed
from mcp_server.scopes import (
    _all_mem_tables,
    _norm_project,
    _project_names,
    _project_suffix,
    _retrieval_tables,
)
from mcp_server.search import _search_memories
from mcp_server.storage import ACTIVE_ROW_SQL
from mcp_server.version import STARTED_AT, VERSION, code_fingerprint

logger = logging.getLogger(__name__)

# `partial_error` of the /nli endpoint:
# "NLI unavailable for <failed> of <total> document(s): <reason>"
_NLI_PARTIAL_RE = re.compile(
    r"NLI unavailable for (?P<failed>\d+) of (?P<total>\d+) document\(s\): "
    r"(?P<reason>.*)", re.DOTALL)


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


def _entities_differ(a: str | None, b: str | None) -> bool:
    """True when two memories are tagged with different entities.

    Both untagged counts as the same subject; one tagged and one untagged
    counts as different, because a recency-based resolution between them is
    as unsafe as between two differently tagged memories.
    """
    if a is None or b is None:
        return a is not b
    return a.strip().lower() != b.strip().lower()


def contradictions_report(entity: Annotated[
        str | None,
        Field(description="Entity to check. Without it every active memory in "
                          "the scope is cross-checked against its nearest "
                          "neighbours (pairs across entities included).")] = None,
    project: Annotated[
        str | None,
        Field(description="Project scope. Without it only global memories are "
                          "checked.")] = None,
    limit: Annotated[
        int,
        Field(description="Maximum number of candidate pairs to report "
                          "(default 20).")] = 20,
    resolve: Annotated[
        Literal["none", "newest"],
        Field(description="'none' (default) only reports. 'newest' keeps the "
                          "most recently updated memory of each SAME-ENTITY "
                          "pair active and marks the older one as superseded; "
                          "cross-entity pairs are reported (cross_entity: "
                          "true) but never auto-resolved, because recency "
                          "says nothing about which fact is correct when the "
                          "entities differ.")] = "none",
) -> dict[str, Any]:
    """Find potentially conflicting facts stored for the same entity.

    Uses vector similarity to find candidate pairs, then NLI to confirm.
    Returns pairs with confidence scores in [0, 1] (higher = more likely contradictory).
    Without `entity` the scan is scope-wide: all active memories compete for
    the candidate slots, so conflicts between differently tagged entities are
    reported too. With `resolve="newest"` the older memory of every confirmed
    pair of the SAME entity is superseded by the newer one (the report names
    the ids); cross-entity pairs stay untouched. An empty result says in
    `warnings` why it is empty: nothing to compare, no similar pair, or no
    contradiction among the checked pairs.
    When the NLI provider fails for some pairs, `nli_failures` names exactly
    which fact/neighbour ids were NOT judged (with the reason), so a partial
    scan is never mistaken for a clean one.
    """
    logger.info("contradictions_report entity=%s limit=%s", entity, limit)
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
    conn = get_conn()
    candidates: dict[str, tuple[dict, list[tuple[str, str]]]] = {}
    try:
        cur = conn.cursor()
        # Only active memories are seeds: already superseded pairs are resolved
        # and must not be reported again. Every type is a candidate - a
        # decision that contradicts a pattern is a conflict too.
        facts = _search_memories(cur, None, None, None, entity, 200, project,
                                  include_embedding=True, include_superseded=False)
        if len(facts) < 2:
            conn.commit()
            scope_desc = (f"entity {entity!r} has" if entity
                          else "the scope has")
            return ok(STATUS_OK, count=0, results=[], warnings=[
                f"nothing to compare: {scope_desc} "
                f"{len(facts)} active memory/memories - a contradiction needs "
                f"at least two"
            ])

        tables = _retrieval_tables(cur, project)
        neighbors_per_fact = 20
        # fact id -> list of (neighbour id, neighbour content)
        checked: set[tuple[str, str]] = set()
        fact_ids = {f["id"] for f in facts}
        # The self-join only sees the entity when one was given: a
        # scope-wide scan must be able to report pairs across entities.
        if entity:
            entity_sql = "AND entity = %s"
            entity_params: list[Any] = [entity]
        else:
            entity_sql = ""
            entity_params = []

        for f in facts:
            if not f.get("embedding"):
                continue
            if len(candidates) >= limit:
                break
            found: list[tuple[str, str]] = []
            for table in tables:
                # Identical content is not a contradiction, but the same fact
                # may live in the project and in the global scope.
                if f.get("content_hash"):
                    filter_sql = "AND content_hash <> %s"
                    filter_params: list[Any] = [f["content_hash"]]
                else:
                    filter_sql = "AND content <> %s"
                    filter_params = [f["content"]]
                cur.execute(
                    f"SELECT id, content, entity FROM {table} "
                    f"WHERE id != %s {entity_sql} "
                    f"{filter_sql} "
                    f"AND embedding IS NOT NULL "
                    f"AND " + ACTIVE_ROW_SQL + " "
                    f"ORDER BY embedding <=> %s::vector ASC LIMIT %s",
                    [f["id"]] + entity_params + filter_params
                    + [f["embedding"], neighbors_per_fact],
                )
                for nid, ncontent, nentity in cur.fetchall():
                    if nid not in fact_ids:
                        continue
                    pair = tuple(sorted([f["id"], nid]))
                    if pair not in checked:
                        checked.add(pair)
                        found.append((nid, ncontent, nentity))
            if found:
                candidates[f["id"]] = (f, found)

        conn.commit()
    except Exception:
        logger.exception("contradictions_report scan failed")
        conn.rollback()
        return fail(STATUS_ERROR, "query failed")
    finally:
        conn.close()

    contradictions: list[dict] = []
    warnings: list[str] = []
    # Machine-readable account of the pairs NLI did NOT judge: one entry per
    # fact whose call failed (all its neighbours) and one per fact whose call
    # failed only for some documents (partial answer). The aggregated warning
    # above stays the human-readable summary.
    nli_failures: list[dict] = []
    nli_failed: str | None = None
    checked_pairs = 0
    try:
        candidate_list = list(candidates.values())

        def _score_fact(item: tuple[dict, list[tuple[str, str]]]):
            fact, neighbours = item
            # One LLM call per fact instead of one per pair. The entity
            # labels let the NLI model judge whether both facts describe the
            # same subject - a differing port of two different services is
            # not a contradiction.
            return nli_detailed(
                fact["content"], [n[1] for n in neighbours],
                query_entity=fact.get("entity"),
                document_entities=[n[2] for n in neighbours],
            )

        # NLI is a network round trip per fact. A scope-wide scan of a few
        # dozen memories ran into the client timeout when the calls were
        # strictly sequential, so a small bounded pool overlaps them. The
        # results are still consumed in candidate order, which keeps the
        # report deterministic.
        scores_by_fact: dict[str, tuple[list[float], str | None, list[int]]] = {}
        nli_errors: list[tuple[str, str]] = []
        nli_error_facts: dict[str, list[str]] = {}
        partials: list[str] = []
        if candidate_list:
            workers = min(8, len(candidate_list))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_score_fact, item)
                           for item in candidate_list]
                for item, future in zip(candidate_list, futures):
                    try:
                        scores_by_fact[item[0]["id"]] = future.result()
                    except Exception as e:  # noqa: BLE001 - reported below
                        nli_errors.append((item[0]["id"], str(e)))
                        nli_error_facts[item[0]["id"]] = [
                            neighbour[0] for neighbour in item[1]]
        if nli_errors:
            nli_failed = f"NLI unavailable: {nli_errors[0][1]}"
            warnings.append(
                f"scan incomplete: {nli_failed} - the result is not a clean "
                f"contradiction scan; retry later")
            logger.warning("contradictions_report NLI failed: %s",
                           nli_errors[0][1])
            for fact_id, error in nli_errors:
                nli_failures.append({
                    "fact_id": fact_id,
                    "neighbor_ids": nli_error_facts.get(fact_id, []),
                    "reason": error})

        for f, neighbours in candidate_list:
            if len(contradictions) >= limit:
                break
            scored = scores_by_fact.get(f["id"])
            if scored is None:
                # This fact's NLI call failed; its pairs were not judged.
                continue
            scores, partial, failed_indices = scored
            if partial:
                # Collected here, aggregated into ONE warning after the
                # loop: a scope-wide check makes one /nli call per fact, and
                # three identical partial lines are harder to read than a
                # qualifying summary.
                partials.append(partial)
            if failed_indices:
                nli_failures.append({
                    "fact_id": f["id"],
                    "neighbor_ids": [neighbours[i][0] for i in failed_indices
                                     if 0 <= i < len(neighbours)],
                    "reason": _partial_reason(partial) or "NLI unavailable"})
            checked_pairs += len(neighbours)
            for (nid, ncontent, nentity), score in zip(neighbours, scores):
                if len(contradictions) >= limit:
                    break
                if score > 0.7:
                    contradictions.append({
                        "a": {"id": f["id"], "content": f["content"],
                              "entity": f.get("entity")},
                        "b": {"id": nid, "content": ncontent,
                              "entity": nentity},
                        # Same entity (or both untagged) -> a recency-based
                        # resolution is meaningful. Different entities ->
                        # two different subjects, resolve="newest" must not
                        # pick a winner (see the resolve block below).
                        "cross_entity": _entities_differ(
                            f.get("entity"), nentity),
                        "confidence": round(score, 3),
                    })
        _append_partial_warning(warnings, partials)
    except Exception as e:
        # An NLI outage must not crash the tool outside the envelope: the
        # pairs seen so far stay in the result, the rest is skipped loudly.
        nli_failed = f"NLI unavailable: {e}"
        warnings.append(
            f"scan incomplete: {nli_failed} - the result is not a clean "
            f"contradiction scan; retry later")
        logger.warning("contradictions_report NLI failed: %s", e)

    contradictions = contradictions[:limit]

    # An empty result must say *why* it is empty: nothing to compare, no
    # similar pair, or no contradiction among the pairs that were checked.
    if not contradictions and not warnings:
        scope_desc = (f"active memories of {entity!r}"
                      if entity else "active memories of the scope")
        if not candidates:
            warnings.append(
                f"no candidate pairs: no two {scope_desc} "
                f"are similar enough to be worth comparing")
        else:
            warnings.append(
                f"no contradictions found: {checked_pairs} similar pair(s) of "
                f"{scope_desc} were checked with NLI")

    resolved = 0
    if resolve == "newest" and contradictions:
        resolvable = [p for p in contradictions if not p.get("cross_entity")]
        skipped_cross = len(contradictions) - len(resolvable)
        if skipped_cross:
            warnings.append(
                f"resolve skipped for {skipped_cross} cross-entity pair(s): "
                f"the two facts belong to different entities, so recency does "
                f"not decide which one survives - inspect them manually or "
                f"re-run with an entity filter")
        if resolvable:
            conn = get_conn()
            try:
                cur = conn.cursor()
                tables = _retrieval_tables(cur, project)
                for pair in resolvable:
                    a_id, b_id = pair["a"]["id"], pair["b"]["id"]
                    timestamps = {}
                    for mem_id in (a_id, b_id):
                        for table in tables:
                            cur.execute(
                                f"SELECT updated_at FROM {table} WHERE id = %s",
                                (mem_id,),
                            )
                            row = cur.fetchone()
                            if row:
                                timestamps[mem_id] = row[0]
                                break
                    a_ts = timestamps.get(a_id)
                    b_ts = timestamps.get(b_id)
                    keep_a = a_ts is not None and (b_ts is None or a_ts >= b_ts)
                    older, newer = (b_id, a_id) if keep_a else (a_id, b_id)
                    changed = 0
                    for table in tables:
                        cur.execute(
                            f"UPDATE {table} SET superseded_by = %s, "
                            f"updated_at = NOW() WHERE id = %s "
                            f"AND superseded_by IS NULL AND deleted_at IS NULL",
                            (newer, older),
                        )
                        changed += cur.rowcount
                    if changed:
                        resolved += 1
                        pair["resolution"] = {"superseded": older, "kept": newer}
                conn.commit()
            except Exception as e:
                conn.rollback()
                logger.exception("contradictions_report resolve failed")
                warnings.append(f"resolve failed: {e}")
            finally:
                conn.close()

    result: dict[str, Any] = {"count": len(contradictions),
                              "results": contradictions,
                              "resolved": resolved,
                              "warnings": warnings}
    if nli_failures:
        # Which pairs were NOT judged, per fact. A stale inference container
        # that does not send `failed_indices` leaves this empty; the
        # aggregated warning still names the incomplete scan then.
        result["nli_failures"] = nli_failures
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
    unintended second write. Expired-but-active rows are counted as context,
    not as an error.
    """
    logger.info("verify_integrity project=%s fix=%s", project, fix)
    try:
        name = _norm_project(project)
    except ValueError as e:
        return fail(STATUS_ERROR, str(e))
    only_global = str(project or "").strip().lower() == "global"
    conn = get_conn()
    try:
        cur = conn.cursor()
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

        # A supersede link may point into another scope (cross-scope
        # supersede), so validity is checked against every scope, not only
        # the selected one.
        all_ids: set[str] = set()
        for table in all_tables:
            cur.execute(f"SELECT id FROM {table}")
            all_ids |= {row[0] for row in cur.fetchall()}

        names = _project_names(cur)
        details: list[dict[str, Any]] = []
        dangling: list[tuple[str, str, str]] = []
        self_refs: list[tuple[str, str, str]] = []
        links: dict[str, str] = {}
        expired_active = 0
        total = 0
        for table in tables:
            scope = ("global" if table == "memories"
                     else names.get(table[len("mem_"):], table[len("mem_"):]))
            cur.execute(
                f"SELECT COUNT(*) FROM {table} WHERE superseded_by IS NULL "
                f"AND deleted_at IS NULL AND expires_at IS NOT NULL "
                f"AND expires_at <= NOW()")
            expired_active += cur.fetchone()[0]
            cur.execute(
                f"SELECT id, superseded_by, entity, content FROM {table} "
                f"WHERE superseded_by IS NOT NULL "
                f"AND superseded_by <> %s", (SENTINEL_FORGOTTEN,))
            for mid, sup, entity, content in cur.fetchall():
                total += 1
                links[mid] = sup
                if sup == mid:
                    self_refs.append((table, mid, entity))
                elif sup not in all_ids:
                    dangling.append((table, mid, entity))
                if len(details) < limit and (sup == mid or sup not in all_ids):
                    details.append({
                        "project": scope,
                        "id": mid,
                        "superseded_by": sup,
                        "entity": entity,
                        "content": content[:200],
                        "problem": ("self_reference" if sup == mid
                                    else "dangling_superseded_by"),
                    })

        # A chain (A -> B -> C) is legal after repeated supersede, but a chain
        # whose middle link is dangling is not: report the count so a caller
        # can spot repeated rewrites of one fact.
        chains = sum(
            1 for mid, sup in links.items()
            if sup in links and links.get(sup) != mid)

        fixed = 0
        if fix and (dangling or self_refs):
            bad = [(table, mid) for table, mid, _ in dangling + self_refs]
            for table, mid in bad:
                cur.execute(
                    f"UPDATE {table} SET superseded_by = NULL WHERE id = %s",
                    (mid,))
                fixed += 1
            conn.commit()
        else:
            conn.commit()
    except Exception as e:
        conn.rollback()
        return fail(STATUS_ERROR, str(e))
    finally:
        conn.close()

    warnings: list[str] = []
    if dangling or self_refs:
        if fix:
            warnings.append(
                f"cleared {fixed} broken supersede link(s); those memories "
                f"are readable history now")
        else:
            warnings.append(
                f"{len(dangling) + len(self_refs)} broken supersede link(s) "
                f"hide rows from every read; call verify_integrity(fix=true) "
                f"to clear them")
    elif total:
        warnings.append("no integrity problems found")
    if expired_active:
        warnings.append(
            f"{expired_active} active memory/memories have a passed TTL and "
            f"are hidden from reads (not an error; use include_expired=true "
            f"to inspect them)")
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


def server_health() -> dict[str, Any]:
    """Check if all services (Postgres, Inference) are healthy."""
    logger.info("server_health")
    pg_ok = False
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT 1")
        pg_ok = True
        conn.commit()
        conn.close()
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
        toolset=toolset,
    )


def stats_resource() -> str:
    """Memory server stats: counts, health."""
    conn = get_conn()
    try:
        cur = conn.cursor()
        tables = _all_mem_tables(cur)
        names = _project_names(cur)
        total = 0
        active = 0
        superseded = 0
        forgotten = 0
        for table in tables:
            cur.execute(
                f"SELECT COUNT(*), "
                f"COUNT(*) FILTER (WHERE superseded_by IS NULL AND deleted_at IS NULL), "
                f"COUNT(*) FILTER (WHERE superseded_by IS NOT NULL AND deleted_at IS NULL), "
                f"COUNT(*) FILTER (WHERE deleted_at IS NOT NULL) "
                f"FROM {table}"
            )
            table_total, table_active, table_superseded, table_forgotten = \
                cur.fetchone()
            total += table_total
            active += table_active
            superseded += table_superseded
            forgotten += table_forgotten
        cur.execute("""
            SELECT
                (SELECT COUNT(*) FROM entities),
                (SELECT COUNT(*) FROM relations)
        """)
        row = cur.fetchone()
        conn.commit()
        entities, relations = row if row else (0, 0)
    except Exception:
        return json.dumps({"error": "unavailable"})
    finally:
        conn.close()

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
        from mcp_server.inference_client import health as h
        return h()
    except Exception:
        return False
