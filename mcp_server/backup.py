import json
import logging
from datetime import datetime

from mcp_server.db import get_conn
from mcp_server.graph import _ensure_graph_entity, _relate_edges
from mcp_server.inference_client import embed
from mcp_server.scopes import (
    _ensure_project_table,
    _mem_table,
    _norm_project,
    _project_names,
    _project_tables,
)
from mcp_server.scoring import _content_hash, _gen_id
from mcp_server.storage import _deduplicate

logger = logging.getLogger(__name__)

def _export_memories_json(cur, project: str | None,
                          include_superseded: bool = False,
                          include_embeddings: bool = False) -> list[dict]:
    names = _project_names(cur)
    if project:
        tables = [(_mem_table(project), project)]
    else:
        tables = [("memories", "global")]
        tables += [(t, names.get(t[len("mem_"):], t[len("mem_"):]))
                   for t in _project_tables(cur)]
    emb_col = ", embedding" if include_embeddings else ""
    where = "" if include_superseded else "WHERE superseded_by IS NULL AND deleted_at IS NULL"
    out: list[dict] = []
    for table, scope_name in tables:
        cur.execute(
            f"SELECT id, content, mem_type, entity, importance, metadata, "
            f"access_count, superseded_by, deleted_at, created_at, updated_at{emb_col} "
            f"FROM {table} {where}"
        )
        colnames = [desc[0] for desc in cur.description]
        for row in cur.fetchall():
            rec = dict(zip(colnames, row))
            rec["project"] = scope_name
            if "embedding" in rec and isinstance(rec["embedding"], str):
                rec["embedding"] = json.loads(rec["embedding"])
            out.append(rec)
    out.sort(key=lambda r: r["created_at"] or _now())
    return out


def _export_graph(cur, project: str | None) -> dict:
    """Entities and relations for the backup (scoped when project is given)."""
    scope_filter = "WHERE project = %s" if project else ""
    params: tuple[Any, ...] = (project,) if project else ()
    cur.execute(
        f"SELECT name, type, project FROM entities {scope_filter} ORDER BY project, name",
        params,
    )
    entities = [{"name": r[0], "type": r[1], "project": r[2]} for r in cur.fetchall()]
    rel_filter = "WHERE e1.project = %s" if project else ""
    cur.execute(
        f"SELECT e1.name, e2.name, r.relation_type, r.weight, e1.project "
        f"FROM relations r "
        f"JOIN entities e1 ON e1.id = r.source_id "
        f"JOIN entities e2 ON e2.id = r.target_id {rel_filter} "
        f"ORDER BY e1.project, e1.name, e2.name",
        params,
    )
    relations = [
        {"from_entity": r[0], "to_entity": r[1], "relation_type": r[2],
         "weight": float(r[3] or 1.0),
         "project": "global" if r[4] == "global" else r[4]}
        for r in cur.fetchall()
    ]
    return {"entities": entities, "relations": relations}


def export_memories(
    format: str = "json",
    project: str | None = None,
    include_embeddings: bool = False,
    include_superseded: bool = False,
    include_graph: bool = True,
) -> str:
    """Export memories (and optionally the graph) as JSON or JSONL.

    Without `project` every scope is exported and each record carries its
    `project`. IDs, `superseded_by` and timestamps are preserved, so imports
    can restore the exact state. The graph is only included in JSON exports.
    """
    logger.info("export_memories format=%s project=%s graph=%s", format, project, include_graph)
    project = _norm_project(project)
    if format not in ("json", "jsonl"):
        return json.dumps({"error": f"unsupported format {format!r}; use 'json' or 'jsonl'"})
    conn = get_conn()
    try:
        cur = conn.cursor()
        rows = _export_memories_json(cur, project, include_superseded,
                                     include_embeddings)
        graph = _export_graph(cur, project) if (include_graph and format != "jsonl") else None
        conn.commit()
    except Exception as e:
        conn.rollback()
        return json.dumps({"error": str(e)})
    finally:
        conn.close()

    records = []
    for r in rows:
        record = {
            "id": r["id"],
            "content": r["content"],
            "type": r["mem_type"],
            "entity": r["entity"],
            "project": r["project"],
            "importance": float(r["importance"]) if r.get("importance") is not None else 1.0,
            "metadata": r.get("metadata") if isinstance(r.get("metadata"), dict) else json.loads(r.get("metadata") or "{}"),
            "created_at": r["created_at"].isoformat() if isinstance(r["created_at"], datetime) else str(r["created_at"]),
            "updated_at": r["updated_at"].isoformat() if isinstance(r["updated_at"], datetime) else str(r["updated_at"]),
            "access_count": r.get("access_count", 0),
            "superseded_by": r.get("superseded_by"),
        }
        if r.get("deleted_at") is not None:
            record["deleted_at"] = (r["deleted_at"].isoformat()
                                    if isinstance(r["deleted_at"], datetime)
                                    else str(r["deleted_at"]))
        if include_embeddings and r.get("embedding") is not None:
            record["embedding"] = r["embedding"]
        records.append(record)

    if format == "jsonl":
        return "\n".join(json.dumps(r) for r in records)
    payload = {"type": "memory_export", "version": "0.2",
               "records": records, "count": len(records)}
    if graph is not None:
        payload["graph"] = graph
    return json.dumps(payload, indent=2)


def _parse_dt(value) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def import_memories(
    data: str,
    project: str | None = None,
    format: str = "json",
    skip_duplicates: bool = True,
    fuzzy_threshold: float | None = None,
) -> dict:
    """Import memories (and graph, for JSON exports).

    Records keep their `id`, timestamps and `superseded_by`, so a re-import is
    idempotent and restores the exact state. Without `project`, each record is
    restored into the scope it was exported from; with `project`, everything
    is written to that project.
    """
    logger.info("import_memories format=%s skip_duplicates=%s fuzzy=%s",
                format, skip_duplicates, fuzzy_threshold)
    project = _norm_project(project)
    if format not in ("json", "jsonl"):
        return {"error": f"unsupported format {format!r}; use 'json' or 'jsonl'",
                "imported": 0, "skipped": 0}
    graph: dict | None = None
    try:
        if format == "jsonl":
            records = [json.loads(line) for line in data.strip().split("\n")
                       if line.strip()]
        else:
            parsed = json.loads(data)
            if isinstance(parsed, dict) and "records" in parsed:
                records = parsed["records"]
                graph = parsed.get("graph")
            elif isinstance(parsed, list):
                records = parsed
            else:
                return {"error": "unsupported format", "imported": 0, "skipped": 0}
    except (json.JSONDecodeError, AttributeError) as e:
        return {"error": f"invalid {format} data: {e}", "imported": 0, "skipped": 0}
    if not isinstance(records, list):
        return {"error": "'records' must be a list", "imported": 0, "skipped": 0}

    conn = get_conn()
    imported = 0
    skipped = 0
    errors: list[str] = []
    graph_entities = 0
    graph_relations = 0
    try:
        cur = conn.cursor()
        for rec in records:
            if not isinstance(rec, dict):
                errors.append("record is not an object")
                skipped += 1
                continue
            content = rec.get("content") or rec.get("text") or ""
            if not content:
                skipped += 1
                continue
            target = (project if project is not None
                      else _norm_project(rec.get("project")))
            table = _mem_table(target)
            if target:
                _ensure_project_table(cur, target)

            mid = rec.get("id") or _gen_id()
            cur.execute(f"SELECT 1 FROM {table} WHERE id = %s", (mid,))
            if cur.fetchone():
                skipped += 1
                continue

            emb = rec.get("embedding")
            if not emb:
                emb = embed([content])[0]
            if skip_duplicates and not rec.get("id"):
                if _deduplicate(cur, content, target, emb,
                                rec.get("fuzzy_threshold") or fuzzy_threshold):
                    skipped += 1
                    continue
            meta = rec.get("metadata", {})
            if isinstance(meta, str):
                meta = json.loads(meta)
            try:
                cur.execute(
                    f"INSERT INTO {table} (id, content, content_hash, embedding, "
                    f"mem_type, scope, entity, importance, metadata, access_count, "
                    f"superseded_by, deleted_at, created_at, updated_at) "
                    f"VALUES (%s, %s, %s, %s::vector, %s, %s, %s, %s, %s::jsonb, "
                    f"%s, %s, %s, COALESCE(%s, NOW()), COALESCE(%s, NOW()))",
                    (mid, content, _content_hash(content), emb,
                     rec.get("type") or rec.get("mem_type", "fact"),
                     "project" if target else "global",
                     rec.get("entity"), float(rec.get("importance", 1.0)),
                     json.dumps(meta), int(rec.get("access_count", 0) or 0),
                     rec.get("superseded_by"), _parse_dt(rec.get("deleted_at")),
                     _parse_dt(rec.get("created_at")),
                     _parse_dt(rec.get("updated_at"))),
                )
                imported += 1
            except Exception as exc:
                errors.append(f"row {imported + skipped}: {exc}")
                skipped += 1

        if graph:
            cur.execute("SET search_path TO ag_catalog, public")

            def _resolve(value):
                scope = project if project is not None else _norm_project(value)
                return _norm_project(scope)

            for ent in graph.get("entities", []):
                if not ent.get("name"):
                    continue
                _ensure_graph_entity(cur, ent["name"], _resolve(ent.get("project")),
                                     ent.get("type"))
                graph_entities += 1

            grouped: dict[Any, list[dict]] = {}
            for rel in graph.get("relations", []):
                if not rel.get("from_entity") or not rel.get("to_entity"):
                    continue
                grouped.setdefault(_resolve(rel.get("project")), []).append({
                    "from_entity": rel["from_entity"],
                    "to_entity": rel["to_entity"],
                    "relation_type": rel["relation_type"],
                    "weight": float(rel.get("weight", 1.0)),
                })
            for scope, edges in grouped.items():
                _relate_edges(cur, edges, scope)
                graph_relations += len(edges)

        conn.commit()
    except Exception as e:
        conn.rollback()
        return {"error": str(e), "imported": imported, "skipped": skipped}
    finally:
        conn.close()

    result = {"imported": imported, "skipped": skipped}
    if graph is not None:
        result["graph_entities"] = graph_entities
        result["graph_relations"] = graph_relations
    if errors:
        result["errors"] = errors[:5]
    return result
