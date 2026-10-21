import json
import logging
from typing import Annotated, Any

from pydantic import Field

from mcp_server.db import get_conn
from mcp_server.envelope import STATUS_ERROR, STATUS_OK, fail, ok
from mcp_server.inference_client import health as _inference_health, health_details, nli
from mcp_server.scopes import (
    _all_mem_tables,
    _count_entities,
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
                          "(default 20).")] = 20) -> dict[str, Any]:
    """Find potentially conflicting facts stored for the same entity.

    Uses vector similarity to find candidate pairs, then NLI to confirm.
    Returns pairs with confidence scores in [0, 1] (higher = more likely contradictory).
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
    try:
        cur = conn.cursor()
        # Only active facts are seeds: already superseded pairs are resolved
        # and must not be reported again.
        facts = _search_memories(cur, None, None, "fact", entity, 200, project,
                                  include_embedding=True, include_superseded=False)
        if len(facts) < 2:
            conn.commit()
            return ok(STATUS_OK, count=0, results=[])

        tables = _retrieval_tables(cur, project)
        neighbors_per_fact = 20
        candidates: list[tuple[dict, str, str]] = []
        checked: set[tuple[str, str]] = set()
        fact_ids = {f["id"] for f in facts}

        for f in facts:
            if not f.get("embedding"):
                continue
            if len(candidates) >= limit * 2:
                break
            found: list[tuple[str, str]] = []
            for table in tables:
                cur.execute(
                    f"SELECT id, content FROM {table} "
                    f"WHERE id != %s AND entity = %s AND mem_type = 'fact' "
                    f"AND embedding IS NOT NULL "
                    f"AND superseded_by IS NULL AND deleted_at IS NULL "
                    f"ORDER BY embedding <=> %s::vector ASC LIMIT %s",
                    (f["id"], entity, f["embedding"], neighbors_per_fact),
                )
                for nid, ncontent in cur.fetchall():
                    if nid not in fact_ids:
                        continue
                    pair = tuple(sorted([f["id"], nid]))
                    if pair not in checked:
                        checked.add(pair)
                        found.append((nid, ncontent))
            candidates.extend((f, nid, ncontent) for nid, ncontent in found)

        conn.commit()
    except Exception:
        conn.rollback()
        return fail(STATUS_ERROR, "query failed")
    finally:
        conn.close()

    contradictions: list[dict] = []
    for f, nid, ncontent in candidates:
        if len(contradictions) >= limit:
            break
        scores = nli(f["content"], [ncontent])
        if len(scores) >= 1 and scores[0] > 0.7:
            contradictions.append({
                "a": {"id": f["id"], "content": f["content"]},
                "b": {"id": nid, "content": ncontent},
                "confidence": round(scores[0], 3),
            })

    contradictions = contradictions[:limit]
    return ok(STATUS_OK, count=len(contradictions), results=contradictions)


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
        for table in tables:
            cur.execute(
                f"SELECT COUNT(*), "
                f"COUNT(*) FILTER (WHERE superseded_by IS NULL AND deleted_at IS NULL) "
                f"FROM {table}"
            )
            table_total, table_active = cur.fetchone()
            total += table_total
            active += table_active
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
        "projects": sorted(names.get(t[len("mem_"):], t[len("mem_"):])
                           for t in tables if t != "memories"),
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
