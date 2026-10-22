import json
import logging
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import Field

from mcp_server.db import get_conn
from mcp_server.envelope import STATUS_ERROR, STATUS_OK, fail, ok
from mcp_server.graph import _ensure_graph_entity, _relate_edges
from mcp_server.inference_client import embed
from mcp_server.scopes import (
    _ensure_project_table,
    _mem_table,
    _norm_project,
    _project_names,
    _project_tables,
    _table_exists,
)
from mcp_server.scoring import _content_hash, _gen_id, _now
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


def _export_excluded_counts(cur, project: str | None) -> tuple[int, int]:
    """Count rows a default export leaves out: (superseded, forgotten)."""
    if project:
        tables = [_mem_table(project)]
    else:
        tables = ["memories"] + list(_project_tables(cur))
    superseded = 0
    forgotten = 0
    for table in tables:
        if not _table_exists(cur, table):
            continue
        cur.execute(
            f"SELECT COUNT(*) FILTER (WHERE deleted_at IS NULL "
            f"AND superseded_by IS NOT NULL), "
            f"COUNT(*) FILTER (WHERE deleted_at IS NOT NULL) FROM {table}"
        )
        row = cur.fetchone()
        superseded += int(row[0] or 0)
        forgotten += int(row[1] or 0)
    return superseded, forgotten


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
    format: Annotated[
        Literal["json", "jsonl"],
        Field(description="'json' (default) returns an object with records and "
                          "the graph; 'jsonl' returns one record per line "
                          "(no graph).")]= "json",
    project: Annotated[
        str | None,
        Field(description="Scope to export. Without it every scope is exported "
                          "and each record carries its project.")] = None,
    include_embeddings: Annotated[
        bool,
        Field(description="Include the vector of every memory (large).")] = False,
    include_superseded: Annotated[
        bool,
        Field(description="Include superseded (invalidated) memories.")] = False,
    include_graph: Annotated[
        bool,
        Field(description="Include entities and relations (JSON exports only).")] = True,
) -> dict[str, Any]:
    """Export memories (and optionally the graph) as JSON or JSONL.

    Without `project` every scope is exported and each record carries its
    `project`. IDs, `superseded_by` and timestamps are preserved, so imports
    can restore the exact state. The graph is only included in JSON exports.

    The payload is returned in "data" and can be passed to `import_memories`
    unchanged.
    """
    logger.info("export_memories format=%s project=%s graph=%s", format, project, include_graph)
    project = _norm_project(project)
    if format not in ("json", "jsonl"):
        return fail(STATUS_ERROR,
                    f"unsupported format {format!r}; use 'json' or 'jsonl'",
                    count=0, data=None)
    conn = get_conn()
    try:
        cur = conn.cursor()
        rows = _export_memories_json(cur, project, include_superseded,
                                     include_embeddings)
        graph = _export_graph(cur, project) if (include_graph and format != "jsonl") else None
        excluded = ((0, 0) if include_superseded
                    else _export_excluded_counts(cur, project))
        conn.commit()
    except Exception as e:
        conn.rollback()
        return fail(STATUS_ERROR, str(e))
    finally:
        conn.close()

    records = []
    for r in rows:
        record = {
            "id": r["id"],
            "content": r["content"],
            # "type" is the historical key, "mem_type" the name every other
            # tool uses. Both are written so old and new exports stay
            # readable; imports accept either.
            "type": r["mem_type"],
            "mem_type": r["mem_type"],
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

    warnings: list[str] = []
    superseded_count, forgotten_count = excluded
    if superseded_count or forgotten_count:
        parts = []
        if superseded_count:
            parts.append(f"{superseded_count} superseded")
        if forgotten_count:
            parts.append(f"{forgotten_count} forgotten")
        warnings.append(
            "export excludes " + " and ".join(parts) + " memories of this "
            "scope; pass include_superseded=true for a complete backup")
    if format == "jsonl":
        data = "\n".join(json.dumps(r) for r in records)
        return ok(STATUS_OK, format=format, count=len(records), data=data,
                  data_format="jsonl", graph_included=False,
                  warnings=warnings)
    payload = {"type": "memory_export", "version": "0.2",
               "records": records, "count": len(records)}
    if graph is not None:
        payload["graph"] = graph
    return ok(STATUS_OK, format=format, count=len(records), data=payload,
              data_format="json", graph_included=graph is not None,
              warnings=warnings)


def _parse_dt(value) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _as_list(value):
    """Return a list, even if the caller packed the array into an object.

    Some MCP clients serialize a free-form JSON array as `{"item": ...}` -
    the array itself for multi-element arrays, the single element for
    one-element arrays. The payload is then structurally valid JSON but no
    longer a list, which would make an otherwise correct import fail with a
    confusing message.
    """
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in ("item", "items"):
            if key in value:
                inner = value[key]
                if isinstance(inner, list):
                    return inner
                if isinstance(inner, dict):
                    return [inner]
        lists = [v for v in value.values() if isinstance(v, list)]
        if len(lists) == 1:
            return lists[0]
    return value


def import_memories(
    data: Annotated[
        str | dict,
        Field(description="Payload of `export_memories`, either as the object "
                          "or as its JSON string. JSONL payloads are one "
                          "record per line. Passing the payload as a JSON "
                          "string is the most reliable form: some clients "
                          "repack the arrays inside a free-form object.")],
    project: Annotated[
        str | None,
        Field(description="Target scope for all records. Without it every "
                          "record is restored into the scope it was exported "
                          "from.")] = None,
    format: Annotated[
        Literal["json", "jsonl"],
        Field(description="Format of `data` (default: json).")] = "json",
    skip_duplicates: Annotated[
        bool,
        Field(description="Skip records whose id already exists instead of "
                          "inserting them again.")] = True,
    fuzzy_threshold: Annotated[
        float | None,
        Field(description="Similarity threshold 0.0-1.0 for semantic dedup of "
                          "records without an id.")] = None,
) -> dict[str, Any]:
    """Import memories (and graph, for JSON exports).

    `data` accepts the payload of `export_memories` either as an object or as
    its JSON string.
    Records keep their `id`, timestamps and `superseded_by`, so a re-import is
    idempotent and restores the exact state. Without `project`, each record is
    restored into the scope it was exported from; with `project`, everything
    is written to that project.
    """
    logger.info("import_memories format=%s skip_duplicates=%s fuzzy=%s",
                format, skip_duplicates, fuzzy_threshold)
    project = _norm_project(project)
    if format not in ("json", "jsonl"):
        return fail(STATUS_ERROR,
                    f"unsupported format {format!r}; use 'json' or 'jsonl'",
                    imported=0, skipped=0)
    graph: dict | None = None
    try:
        if isinstance(data, dict):
            parsed = data
        elif format == "jsonl":
            parsed = [json.loads(line) for line in str(data).strip().split("\n")
                      if line.strip()]
        else:
            parsed = json.loads(data)
        if isinstance(parsed, dict) and "records" in parsed:
            records = _as_list(parsed["records"])
            graph = parsed.get("graph")
        elif isinstance(parsed, list):
            records = parsed
        else:
            return fail(
                STATUS_ERROR,
                "payload must be the object of `export_memories` "
                "(a dict with a 'records' list) or a list of record objects; "
                f"got a dict with keys {sorted(parsed.keys()) if isinstance(parsed, dict) else type(parsed).__name__}",
                imported=0, skipped=0)
    except (json.JSONDecodeError, AttributeError) as e:
        return fail(STATUS_ERROR, f"invalid {format} data: {e}",
                    imported=0, skipped=0)
    if not isinstance(records, list):
        return fail(STATUS_ERROR,
                    f"'records' must be a list of objects, got {type(records).__name__}",
                    imported=0, skipped=0)

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
                duplicate = _deduplicate(cur, content, target, emb,
                                         rec.get("fuzzy_threshold") or fuzzy_threshold)
                if duplicate:
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

        if graph is not None and not isinstance(graph, dict):
            graph = None
        if graph:
            cur.execute("SET search_path TO ag_catalog, public")

            def _resolve(value):
                scope = project if project is not None else _norm_project(value)
                return _norm_project(scope)

            for ent in _as_list(graph.get("entities", [])):
                if not isinstance(ent, dict) or not ent.get("name"):
                    continue
                _ensure_graph_entity(cur, ent["name"], _resolve(ent.get("project")),
                                     ent.get("type"))
                graph_entities += 1

            grouped: dict[Any, list[dict]] = {}
            for rel in _as_list(graph.get("relations", [])):
                if not isinstance(rel, dict):
                    continue
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
        return fail(STATUS_ERROR, str(e), imported=imported, skipped=skipped)
    finally:
        conn.close()

    result = {"imported": imported, "skipped": skipped}
    if graph is not None:
        result["graph_entities"] = graph_entities
        result["graph_relations"] = graph_relations
    if errors:
        result["errors"] = errors[:5]
    return ok(STATUS_OK, **result)
