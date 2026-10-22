import json
import logging
from typing import Annotated, Any, Literal

from pydantic import Field

from mcp_server.db import get_conn
from mcp_server.envelope import STATUS_ERROR, STATUS_OK, fail, ok
from mcp_server.inference_client import health_details, nli
from mcp_server.scopes import (
    _all_mem_tables,
    _norm_project,
    _project_names,
    _retrieval_tables,
)
from mcp_server.search import _search_memories

logger = logging.getLogger(__name__)

def contradictions_report(entity: Annotated[
        str | None,
        Field(description="Entity to check. Required: the report works on one "
                          "entity at a time.")] = None,
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
                          "most recently updated memory of each pair active "
                          "and marks the older one as superseded.")] = "none",
) -> dict[str, Any]:
    """Find potentially conflicting facts stored for the same entity.

    Uses vector similarity to find candidate pairs, then NLI to confirm.
    Returns pairs with confidence scores in [0, 1] (higher = more likely contradictory).
    With `resolve="newest"` the older memory of every confirmed pair is
    superseded by the newer one (recency decides; the report names the ids).
    An empty result says in `warnings` why it is empty: nothing to compare,
    no similar pair, or no contradiction among the checked pairs.
    """
    logger.info("contradictions_report entity=%s limit=%s", entity, limit)
    project = _norm_project(project)
    if not entity:
        return fail(STATUS_ERROR, "entity parameter is required. "
                                   "contradictions_report works on one entity at a time.")
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
            return ok(STATUS_OK, count=0, results=[], warnings=[
                f"nothing to compare: entity {entity!r} has "
                f"{len(facts)} active memory/memories - a contradiction needs "
                f"at least two"
            ])

        tables = _retrieval_tables(cur, project)
        neighbors_per_fact = 20
        # fact id -> list of (neighbour id, neighbour content)
        checked: set[tuple[str, str]] = set()
        fact_ids = {f["id"] for f in facts}

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
                    f"SELECT id, content FROM {table} "
                    f"WHERE id != %s AND entity = %s "
                    f"{filter_sql} "
                    f"AND embedding IS NOT NULL "
                    f"AND superseded_by IS NULL AND deleted_at IS NULL "
                    f"ORDER BY embedding <=> %s::vector ASC LIMIT %s",
                    [f["id"], entity] + filter_params
                    + [f["embedding"], neighbors_per_fact],
                )
                for nid, ncontent in cur.fetchall():
                    if nid not in fact_ids:
                        continue
                    pair = tuple(sorted([f["id"], nid]))
                    if pair not in checked:
                        checked.add(pair)
                        found.append((nid, ncontent))
            if found:
                candidates[f["id"]] = (f, found)

        conn.commit()
    except Exception:
        conn.rollback()
        return fail(STATUS_ERROR, "query failed")
    finally:
        conn.close()

    contradictions: list[dict] = []
    warnings: list[str] = []
    nli_failed: str | None = None
    checked_pairs = 0
    try:
        for f, neighbours in candidates.values():
            if len(contradictions) >= limit:
                break
            # One LLM call per fact instead of one per pair.
            scores = nli(f["content"], [n[1] for n in neighbours])
            checked_pairs += len(neighbours)
            for (nid, ncontent), score in zip(neighbours, scores):
                if len(contradictions) >= limit:
                    break
                if score > 0.7:
                    contradictions.append({
                        "a": {"id": f["id"], "content": f["content"]},
                        "b": {"id": nid, "content": ncontent},
                        "confidence": round(score, 3),
                    })
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
        if not candidates:
            warnings.append(
                f"no candidate pairs: no two active memories of {entity!r} "
                f"are similar enough to be worth comparing")
        else:
            warnings.append(
                f"no contradictions found: {checked_pairs} similar pair(s) of "
                f"entity {entity!r} were checked with NLI")

    resolved = 0
    if resolve == "newest" and contradictions:
        conn = get_conn()
        try:
            cur = conn.cursor()
            tables = _retrieval_tables(cur, project)
            for pair in contradictions:
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

    return ok(STATUS_OK, count=len(contradictions), results=contradictions,
              resolved=resolved, warnings=warnings)


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
    return ok(
        STATUS_OK,
        postgres=pg_ok,
        inference=inference_details is not None,
        nli_configured=(inference_details.get("llm_configured")
                        if inference_details is not None
                        and "llm_configured" in inference_details else None),
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
