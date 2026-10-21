import json
import logging
from typing import Any

from mcp_server.db import get_conn
from mcp_server.inference_client import embed
from mcp_server.scopes import (
    _ag_str,
    _all_mem_tables,
    _count_entities,
    _norm_project,
    _project_names,
    _retrieval_tables,
    _scope_key,
    _scope_label,
)
from mcp_server.scoring import _decay_norm, _now
from mcp_server.search import _search_memories, _search_memories_graph_boost
from mcp_server.storage import _log_access
from mcp_server.tools.maintenance import _health_check

logger = logging.getLogger(__name__)

def recall(
    query: str,
    mem_type: str | None = None,
    entity: str | None = None,
    limit: int = 10,
    project: str | None = None,
    hybrid: bool = True,
    include_decay: bool = False,
    min_decay_score: float | None = None,
    include_superseded: bool = False,
    graph_boost: bool = False,  # Neu: Graph-Enhanced Retrieval
    min_similarity: float | None = None,  # 0..1, filtert semantisch schwache Treffer
) -> list[dict]:
    """Search memories using hybrid vector + fulltext search (RRF fusion).

    Project searches also see global memories; global searches see only global ones.
    Set graph_boost=True to include related entities in search via graph traversal.
    Set min_similarity (0..1) to drop semantically weak matches.
    Set include_decay=True to blend in recency/frequency (max 30% of the score).
    min_decay_score (0..1) filters on the normalized decay score.
    """
    logger.info("recall query='%s' entity=%s hybrid=%s include_decay=%s graph_boost=%s min_similarity=%s",
                query[:80], entity, hybrid, include_decay, graph_boost, min_similarity)
    project = _norm_project(project)
    if not query or not query.strip():
        return [{"error": "query must not be empty"}]
    if min_similarity is not None and not 0.0 <= min_similarity <= 1.0:
        return [{"error": "min_similarity must be between 0.0 and 1.0"}]
    if min_decay_score is not None and not 0.0 <= min_decay_score <= 1.0:
        return [{"error": "min_decay_score must be between 0.0 and 1.0"}]
    if limit < 1:
        return [{"error": "limit must be >= 1"}]
    query_emb = embed([query], mode="query")[0]
    conn = get_conn()
    try:
        cur = conn.cursor()
        
        # Neue RRF-Funktion nutzen
        results = _search_memories(
            cur, query_emb, query if hybrid else None, 
            mem_type, entity, limit, project, 
            include_decay, include_superseded=include_superseded, 
            min_similarity=min_similarity,
            hybrid=hybrid
        )
        
        # Graph-enhanced retrieval when enabled
        if graph_boost and entity:
            results = _search_memories_graph_boost(
                results, entity, limit, project, include_decay
            )

        # Update access counters (decay frequency)
        hit_ids = [r.get("id") for r in results if isinstance(r, dict) and r.get("id")]
        _log_access(cur, hit_ids, project)
        for r in results:
            if isinstance(r, dict) and r.get("id") in hit_ids and "error" not in r:
                r["access_count"] = (r.get("access_count") or 0) + 1

        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.exception("recall failed")
        return [{"error": str(e)}]
    finally:
        conn.close()

    if min_decay_score is not None:
        filtered = []
        for r in results:
            if not isinstance(r, dict) or "error" in r:
                filtered.append(r)
                continue
            norm = r.get("_decay_norm")
            if norm is None:
                norm = _decay_norm(
                    r.get("created_at"), r.get("updated_at"),
                    r.get("access_count") or 0, r.get("importance") or 1.0,
                )
                r["_decay_norm"] = norm
            if norm >= min_decay_score:
                filtered.append(r)
        results = filtered
    return results[:limit]


def get_entity(name: str, project: str | None = None,
               include_superseded: bool = False) -> dict:
    """Get everything known about an entity, including all linked memories."""
    logger.info("get_entity name=%s", name)
    project = _norm_project(project)
    conn = get_conn()
    memories = []
    entity_data = None
    try:
        cur = conn.cursor()
        memories = _search_memories(cur, None, None, None, name, 100, project,
                                    include_superseded=include_superseded)
        scope = _scope_key(project)
        cur.execute(
            "SELECT id, name, type, project FROM entities WHERE name = %s AND project = %s",
            (name, scope),
        )
        row = cur.fetchone()
        if row:
            entity_data = {"id": row[0], "name": row[1], "type": row[2],
                           "project": row[3], "connections": []}
            # also query AGE for graph neighbors (same scope label)
            label = _scope_label(project)
            cur.execute("SET search_path TO ag_catalog, public")
            name_json = json.dumps(name)
            cur.execute(
                f"SELECT * FROM cypher('memory_graph', $$ "
                f"MATCH (a:{label} {{name: {name_json}}})-[r]-(b) "
                f"RETURN b.name, type(r), coalesce(r.weight, 1.0), startNode(r).name $$) AS "
                f"(neighbor agtype, rel agtype, w agtype, src agtype)"
            )
            entity_data["connections"] = [
                {"entity": _ag_str(r[0]), "relation": _ag_str(r[1]),
                 "weight": float(str(r[2])),
                 "direction": "outgoing" if _ag_str(r[3]) == name else "incoming"}
                for r in cur.fetchall()
            ]
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.exception("get_entity failed")
        return {"entity": name, "details": None, "memories": [], "error": str(e)}
    finally:
        conn.close()
    return {"entity": name, "details": entity_data, "memories": memories}


def list_entities(entity_type: str | None = None, limit: int = 50,
                  project: str | None = None) -> list[dict]:
    """Browse entities.

    Returns graph entities (created via `relate`) plus entity names that only
    exist on memories. With `project`, that project and global scope are
    listed; without it every scope is listed. Each row carries its `project`.
    """
    logger.info("list_entities entity_type=%s limit=%s project=%s",
                entity_type, limit, project)
    project = _norm_project(project)
    conn = get_conn()
    try:
        cur = conn.cursor()
        tables = (_retrieval_tables(cur, project) if project
                  else _all_mem_tables(cur))
        names = _project_names(cur)
        scope_of = {t: (names.get(t[len("mem_"):], t[len("mem_"):])
                        if t != "memories" else "global") for t in tables}

        conditions = ["(project = %s OR project = 'global')"] if project else ["TRUE"]
        params: list[Any] = [project] if project else []
        if entity_type:
            conditions.append("type = %s")
            params.append(entity_type)
        cur.execute(
            f"SELECT id, name, type, project FROM entities "
            f"WHERE {' AND '.join(conditions)} ORDER BY name LIMIT %s",
            params + [limit],
        )
        rows = [{"id": r[0], "name": r[1], "type": r[2], "project": r[3],
                 "graph_node": True} for r in cur.fetchall()]

        seen = {(r["name"], r["project"]) for r in rows}
        for table in tables:
            scope = scope_of[table]
            cur.execute(
                f"SELECT DISTINCT entity FROM {table} "
                f"WHERE entity IS NOT NULL AND superseded_by IS NULL AND deleted_at IS NULL"
            )
            for (ent,) in cur.fetchall():
                if (ent, scope) in seen:
                    continue
                seen.add((ent, scope))
                rows.append({"name": ent, "project": scope, "graph_node": False})
        conn.commit()
    except Exception as e:
        conn.rollback()
        return [{"error": str(e)}]
    finally:
        conn.close()
    return rows[:limit]


def init_context(project: str | None = None) -> dict:
    """Call at session start. Returns recent + important bootstrap info + health.

    Pass `project` to bootstrap for one project; global memories are always
    included. Without a project all project tables are summarized.
    """
    project = _norm_project(project)
    conn = get_conn()
    memories = []
    total = 0
    active = 0
    projects: list[str] = []
    try:
        cur = conn.cursor()
        tables = (_retrieval_tables(cur, project) if project
                  else _all_mem_tables(cur))
        memories = _search_memories(cur, None, None, None, None, 10, project,
                                    include_decay=True, tables=tables)
        names = _project_names(cur)
        projects = [names.get(t[len("mem_"):], t[len("mem_"):])
                    for t in tables if t != "memories"]
        for table in tables:
            cur.execute(
                f"SELECT COUNT(*), "
                f"COUNT(*) FILTER (WHERE superseded_by IS NULL AND deleted_at IS NULL) "
                f"FROM {table}"
            )
            table_total, table_active = cur.fetchone()
            total += table_total
            active += table_active
        conn.commit()
    except Exception:
        conn.rollback()
    finally:
        conn.close()
    return {
        "session_start": _now().isoformat(),
        "total_memories": total,
        "active_memories": active,
        "projects": sorted(projects),
        "recent_memories": memories,
        "graph_entities_count": _count_entities(),
        "inference_healthy": _health_check(),
    }
