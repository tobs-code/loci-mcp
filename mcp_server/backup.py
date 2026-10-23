import json
import logging
from datetime import datetime, timezone
from typing import Annotated, Any, Literal

from pydantic import Field

from mcp_server.config import SENTINEL_FORGOTTEN
from mcp_server.db import get_conn
from mcp_server.envelope import STATUS_ERROR, STATUS_OK, fail, ok
from mcp_server.graph import _ensure_graph_entity, _relate_edges
from mcp_server.inference_client import embed
from mcp_server.scopes import (
    _ensure_project_table,
    _is_global_scope,
    _mem_table,
    _norm_project,
    _project_names,
    _project_tables,
    _table_exists,
)
from mcp_server.scoring import _content_hash, _gen_id, _now
from mcp_server.search import _extra_conditions
from mcp_server.storage import _deduplicate, clear_dangling_supersede

logger = logging.getLogger(__name__)

def _export_memories_json(cur, project: str | None,
                          include_superseded: bool = False,
                          include_embeddings: bool = False,
                          only_global: bool = False,
                          entity: str | None = None,
                          mem_type: str | None = None,
                          updated_after: str | None = None,
                          metadata_filter: dict | None = None,
                          include_expired: bool = True) -> list[dict]:
    names = _project_names(cur)
    if project:
        tables = [(_mem_table(project), project)]
    elif only_global:
        # An explicit project="global" means the shared scope, exactly like it
        # does in every other tool. Without this branch it widened to "every
        # scope" and a "global only" backup silently contained project rows.
        tables = [("memories", "global")]
    else:
        tables = [("memories", "global")]
        tables += [(t, names.get(t[len("mem_"):], t[len("mem_"):]))
                   for t in _project_tables(cur)]
    emb_col = ", embedding" if include_embeddings else ""
    conditions: list[str] = []
    params: list[Any] = []
    if not include_superseded:
        conditions.append("superseded_by IS NULL AND deleted_at IS NULL")
    if not include_expired:
        # Symmetric with every read tool: an expired row is inactive. The
        # default keeps it (a backup must not lose the TTL/history), but a
        # caller who wants only active facts can now say so.
        conditions.append("(expires_at IS NULL OR expires_at > NOW())")
    if entity:
        conditions.append("entity = %s")
        params.append(entity)
    if mem_type:
        conditions.append("mem_type = %s")
        params.append(mem_type)
    extra_conditions, extra_params = _extra_conditions(updated_after,
                                                       metadata_filter)
    conditions.extend(extra_conditions)
    params.extend(extra_params)
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    out: list[dict] = []
    for table, scope_name in tables:
        cur.execute(
            f"SELECT id, content, mem_type, entity, importance, metadata, "
            f"access_count, superseded_by, deleted_at, expires_at, "
            f"created_at, updated_at{emb_col} "
            f"FROM {table} {where}",
            params,
        )
        colnames = [desc[0] for desc in cur.description]
        for row in cur.fetchall():
            rec = dict(zip(colnames, row))
            rec["project"] = scope_name
            # The forgotten marker is internal bookkeeping, not a supersede
            # link: exporting it would write the sentinel back on import and
            # make a forgotten row look superseded by a fake memory id.
            if (rec.get("deleted_at") is not None
                    and rec.get("superseded_by") == SENTINEL_FORGOTTEN):
                rec["superseded_by"] = None
            if "embedding" in rec and isinstance(rec["embedding"], str):
                rec["embedding"] = json.loads(rec["embedding"])
            out.append(rec)
    out.sort(key=lambda r: r["created_at"] or _now())
    return out


def _export_excluded_counts(cur, project: str | None,
                            only_global: bool = False) -> tuple[int, int]:
    """Count rows a default export leaves out: (superseded, forgotten)."""
    if project:
        tables = [_mem_table(project)]
    elif only_global:
        tables = ["memories"]
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


def _export_graph(cur, project: str | None,
                  only_global: bool = False,
                  entity: str | None = None) -> dict:
    """Entities and relations for the backup (scoped when project is given).

    `entity` narrows the graph to that node and its incident edges, matching
    the memory filter: exporting only one entity's memories but the whole
    graph made the backup say more than the caller asked for. The other
    endpoints of those edges are included as well, so the exported graph is
    complete and can be restored.
    """
    scope_name = "global" if only_global else project
    conditions: list[str] = []
    params: list[Any] = []
    if scope_name:
        conditions.append("project = %s")
        params.append(scope_name)
    if entity:
        conditions.append("name = %s")
        params.append(entity)
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    cur.execute(
        f"SELECT name, type, project FROM entities {where} "
        f"ORDER BY project, name",
        params,
    )
    entities = [{"name": r[0], "type": r[1], "project": r[2]} for r in cur.fetchall()]
    rel_conditions = (["e1.project = %s"] if scope_name else [])
    rel_params: list[Any] = [scope_name] if scope_name else []
    if entity:
        rel_conditions.append("(e1.name = %s OR e2.name = %s)")
        rel_params.extend([entity, entity])
    rel_filter = ("WHERE " + " AND ".join(rel_conditions)
                  if rel_conditions else "")
    cur.execute(
        f"SELECT e1.name, e2.name, r.relation_type, r.weight, e1.project "
        f"FROM relations r "
        f"JOIN entities e1 ON e1.id = r.source_id "
        f"JOIN entities e2 ON e2.id = r.target_id {rel_filter} "
        f"ORDER BY e1.project, e1.name, e2.name",
        rel_params,
    )
    relations = [
        {"from_entity": r[0], "to_entity": r[1], "relation_type": r[2],
         "weight": float(r[3] or 1.0),
         "project": "global" if r[4] == "global" else r[4]}
        for r in cur.fetchall()
    ]
    if entity:
        # An entity-filtered export keeps the incident edges - so it must keep
        # their other endpoints too. Without them the backup held an edge whose
        # target node was missing, and a restore could not recreate the graph.
        known = {(e["name"], e["project"]) for e in entities}
        endpoints = {
            pair
            for rel in relations
            for pair in ((rel["from_entity"], rel["project"]),
                         (rel["to_entity"], rel["project"]))
        }
        for name, scope in sorted(endpoints - known):
            cur.execute(
                "SELECT name, type, project FROM entities "
                "WHERE name = %s AND project = %s",
                (name, scope),
            )
            row = cur.fetchone()
            if row:
                entities.append({"name": row[0], "type": row[1],
                                 "project": row[2]})
        entities.sort(key=lambda e: (e["project"], e["name"]))
    return {"entities": entities, "relations": relations}


def export_memories(
    format: Annotated[
        Literal["json", "jsonl"],
        Field(description="'json' (default) returns an object with records and "
                          "the graph; 'jsonl' returns one record per line "
                          "(no graph).")]= "json",
    project: Annotated[
        str | None,
        Field(description="Scope to export. 'global' exports only the shared "
                          "scope; without it every scope is exported and each "
                          "record carries its project.")] = None,
    include_embeddings: Annotated[
        bool,
        Field(description="Include the vector of every memory (large).")] = False,
    include_superseded: Annotated[
        bool,
        Field(description="Include superseded (invalidated) memories.")] = False,
    include_expired: Annotated[
        bool,
        Field(description="Include memories whose TTL has passed (default "
                          "true: a backup keeps them with their expires_at). "
                          "Pass false to export active rows only.")] = True,
    include_graph: Annotated[
        bool,
        Field(description="Include entities and relations (JSON exports only).")] = True,
    entity: Annotated[
        str | None,
        Field(description="Only export memories of this entity.")] = None,
    mem_type: Annotated[
        str | None,
        Field(description="Only export memories of this type.")] = None,
    updated_after: Annotated[
        str | None,
        Field(description="Only export memories changed at or after this ISO "
                          "date, e.g. '2026-10-01'.")] = None,
    metadata_filter: Annotated[
        dict[str, Any] | None,
        Field(description="Only export memories whose metadata contains these "
                          "key/value pairs (same tolerant comparison as "
                          "`list_memories`).")] = None,
) -> dict[str, Any]:
    """Export memories (and optionally the graph) as JSON or JSONL.

    Without `project` every scope is exported and each record carries its
    `project`. IDs, `superseded_by` and timestamps are preserved, so imports
    can restore the exact state. The graph is only included in JSON exports.

    The payload is returned in "data" and can be passed to `import_memories`
    unchanged.
    """
    logger.info("export_memories format=%s project=%s graph=%s", format, project, include_graph)
    # `_norm_project("global")` returns None, which also means "no scope
    # given". Remember the explicit shared-scope request before normalizing,
    # so it does not widen to every scope.
    only_global = _is_global_scope(project)
    try:
        project = _norm_project(project)
    except ValueError as e:
        return fail(STATUS_ERROR, str(e))
    if format not in ("json", "jsonl"):
        return fail(STATUS_ERROR,
                    f"unsupported format {format!r}; use 'json' or 'jsonl'",
                    count=0, data=None)
    conn = get_conn()
    try:
        cur = conn.cursor()
        rows = _export_memories_json(cur, project, include_superseded,
                                     include_embeddings,
                                     only_global=only_global,
                                     entity=entity, mem_type=mem_type,
                                     updated_after=updated_after,
                                     metadata_filter=metadata_filter,
                                     include_expired=include_expired)
        graph = (_export_graph(cur, project, only_global=only_global,
                               entity=entity)
                 if (include_graph and format != "jsonl") else None)
        excluded = ((0, 0) if include_superseded
                    else _export_excluded_counts(cur, project, only_global))
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
            # A backup must preserve the TTL: dropping it made an expired
            # fact active again on import and silently removed a future TTL.
            "expires_at": (r["expires_at"].isoformat()
                           if isinstance(r.get("expires_at"), datetime)
                           else (str(r["expires_at"])
                                 if r.get("expires_at") is not None else None)),
        }
        if r.get("deleted_at") is not None:
            record["deleted_at"] = (r["deleted_at"].isoformat()
                                    if isinstance(r["deleted_at"], datetime)
                                    else str(r["deleted_at"]))
        if include_embeddings and r.get("embedding") is not None:
            record["embedding"] = r["embedding"]
        records.append(record)

    warnings: list[str] = []
    if format == "jsonl" and include_graph:
        warnings.append(
            "format='jsonl' has no graph: include_graph was ignored. Use "
            "format='json' to back up entities and relations")
    if graph is not None and (mem_type or updated_after or metadata_filter):
        warnings.append(
            "the memory filters (mem_type/updated_after/metadata_filter) do "
            "not apply to the graph: entities and relations were exported in "
            "full for the selected scope; pass include_graph=false for a "
            "memory-only backup")
    now = datetime.now(timezone.utc)
    expired_count = sum(
        1 for r in rows
        if isinstance(r.get("expires_at"), datetime)
        and r["expires_at"] <= now)
    if expired_count:
        warnings.append(
            f"export includes {expired_count} expired memories: they are "
            f"backed up with their expires_at and stay inactive after an "
            f"import")
    elif not include_expired:
        warnings.append(
            "include_expired=false: memories whose TTL has passed were left "
            "out of the export")
    superseded_count, forgotten_count = excluded
    if superseded_count or forgotten_count:
        parts = []
        if superseded_count:
            parts.append(f"{superseded_count} superseded")
        if forgotten_count:
            parts.append(f"{forgotten_count} forgotten")
        if only_global:
            where = " of the global scope"
        elif project:
            where = f" of project {project!r}"
        else:
            where = " across all scopes"
        warnings.append(
            "export excludes " + " and ".join(parts) + " memories"
            + where + "; pass include_superseded=true for a complete backup")
    if format == "jsonl":
        data = "\n".join(json.dumps(r) for r in records)
        return ok(STATUS_OK, format=format, count=len(records), data=data,
                  data_format="jsonl", graph_included=False,
                  include_expired=include_expired, warnings=warnings)
    payload = {"type": "memory_export", "version": "0.3",
               "records": records, "count": len(records)}
    if graph is not None:
        payload["graph"] = graph
    return ok(STATUS_OK, format=format, count=len(records), data=payload,
              data_format="json", graph_included=graph is not None,
              include_expired=include_expired, warnings=warnings)


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
        Field(description="Skip records whose CONTENT already exists. A "
                          "record with an id is always skipped when that id "
                          "exists, so a re-import stays idempotent; this flag "
                          "only controls the content check for records "
                          "WITHOUT an id.")] = True,
    fuzzy_threshold: Annotated[
        float | None,
        Field(description="Similarity threshold 0.0-1.0 for semantic dedup of "
                          "records without an id.")] = None,
    dry_run: Annotated[
        bool,
        Field(description="True validates the payload and reports what would "
                          "be imported without writing anything.")] = False,
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
    try:
        project = _norm_project(project)
    except ValueError as e:
        return fail(STATUS_ERROR, str(e))
    if format not in ("json", "jsonl"):
        return fail(STATUS_ERROR,
                    f"unsupported format {format!r}; use 'json' or 'jsonl'",
                    imported=0, skipped=0)
    graph: dict | None = None
    try:
        if format == "jsonl" and isinstance(data, dict):
            return fail(
                STATUS_ERROR,
                "format='jsonl' but data is an object; pass a JSONL string "
                "(one record per line), a list of records, or format='json'",
                imported=0, skipped=0)
        if isinstance(data, dict):
            parsed = data
        elif format == "jsonl":
            lines = [line for line in str(data).strip().split("\n")
                     if line.strip()]
            parsed = []
            for lineno, line in enumerate(lines, 1):
                try:
                    parsed.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    return fail(
                        STATUS_ERROR,
                        f"invalid jsonl data at line {lineno}: {exc}",
                        imported=0, skipped=0)
            if (len(parsed) == 1 and isinstance(parsed[0], dict)
                    and "records" in parsed[0]):
                # A single line that is a whole export object is not JSONL.
                # Importing it as a list used to yield a silent 0/0 "success".
                return fail(
                    STATUS_ERROR,
                    "format='jsonl' but data is a JSON export object (it has "
                    "'records'); pass format='json' or send one record per "
                    "line",
                    imported=0, skipped=0)
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
    # Why a record was skipped matters to the caller: an idempotent re-import
    # ("the id already exists") is not the same as a silently dropped record
    # ("no content") or a content duplicate that a flag can override.
    skipped_id_exists = 0
    skipped_duplicate_content = 0
    skipped_empty_content = 0
    errors: list[str] = []
    graph_entities = 0
    graph_entities_created = 0
    graph_relations = 0
    graph_relations_created = 0
    try:
        cur = conn.cursor()
        for rec in records:
            if not isinstance(rec, dict):
                errors.append("record is not an object")
                skipped += 1
                continue
            raw_content = rec.get("content")
            if raw_content is None:
                raw_content = rec.get("text")
            # Whitespace-only content must be refused exactly like `remember`
            # refuses it: the old check used truthiness, so "   " was imported
            # as an unsearchable row that no read tool could ever explain.
            content = (raw_content.strip()
                       if isinstance(raw_content, str) else "")
            if not content:
                skipped += 1
                skipped_empty_content += 1
                continue
            try:
                target = (project if project is not None
                          else _norm_project(rec.get("project")))
            except ValueError as exc:
                # A record with a broken project name must not abort the
                # whole import: it is reported and skipped, like bad metadata.
                errors.append(f"record {rec.get('id') or imported + skipped}: {exc}")
                skipped += 1
                continue
            expires = _parse_dt(rec.get("expires_at"))
            if rec.get("expires_at") not in (None, "") and expires is None:
                errors.append(
                    f"record {rec.get('id') or imported + skipped}: "
                    f"expires_at is not a valid ISO 8601 timestamp")
                skipped += 1
                continue
            table = _mem_table(target)
            if target:
                _ensure_project_table(cur, target)

            mid = rec.get("id") or _gen_id()
            cur.execute(f"SELECT 1 FROM {table} WHERE id = %s", (mid,))
            if cur.fetchone():
                skipped += 1
                skipped_id_exists += 1
                continue

            emb = rec.get("embedding")
            if not emb:
                emb = embed([content])[0]
            if skip_duplicates and not rec.get("id"):
                duplicate = _deduplicate(cur, content, target, emb,
                                         rec.get("fuzzy_threshold") or fuzzy_threshold)
                if duplicate:
                    skipped += 1
                    skipped_duplicate_content += 1
                    continue
            meta = rec.get("metadata", {})
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except json.JSONDecodeError as exc:
                    # A broken metadata value must not abort the whole import:
                    # the record is reported, the rest is restored.
                    errors.append(
                        f"record {mid!r}: metadata is not valid JSON ({exc})")
                    skipped += 1
                    continue
            try:
                cur.execute(
                    f"INSERT INTO {table} (id, content, content_hash, embedding, "
                    f"mem_type, scope, entity, importance, metadata, access_count, "
                    f"superseded_by, deleted_at, expires_at, created_at, updated_at) "
                    f"VALUES (%s, %s, %s, %s::vector, %s, %s, %s, %s, %s::jsonb, "
                    f"%s, %s, %s, %s, COALESCE(%s, NOW()), "
                    f"COALESCE(%s, COALESCE(%s, NOW())))",
                    (mid, content, _content_hash(content), emb,
                     rec.get("type") or rec.get("mem_type", "fact"),
                     "project" if target else "global",
                     rec.get("entity"), float(rec.get("importance", 1.0)),
                     json.dumps(meta), int(rec.get("access_count", 0) or 0),
                     rec.get("superseded_by"), _parse_dt(rec.get("deleted_at")),
                     expires, _parse_dt(rec.get("created_at")),
                     _parse_dt(rec.get("updated_at")),
                     _parse_dt(rec.get("created_at"))),
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
                try:
                    return _norm_project(scope)
                except ValueError:
                    # A blank project in a graph record stays a data problem
                    # of that record, not of the whole import.
                    return None

            for ent in _as_list(graph.get("entities", [])):
                if not isinstance(ent, dict) or not ent.get("name"):
                    continue
                if _ensure_graph_entity(
                    cur, ent["name"], _resolve(ent.get("project")),
                    ent.get("type"),
                ):
                    graph_entities_created += 1
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
                restored = _relate_edges(cur, edges, scope)
                graph_relations += len(restored)
                graph_relations_created += sum(
                    1 for edge in restored if edge.get("status") == "created")

        repaired_links = 0
        if dry_run:
            conn.rollback()
        else:
            # A record can carry a `superseded_by` id that was not part of the
            # import (or whose replacement was hard-deleted). Without repair
            # that row would stay invisible forever.
            repaired_links = clear_dangling_supersede(cur)
            conn.commit()
    except Exception as e:
        conn.rollback()
        return fail(STATUS_ERROR, str(e), imported=imported, skipped=skipped,
                    invalid=len(errors))
    finally:
        conn.close()

    result = {"imported": imported, "skipped": skipped,
              "invalid": len(errors)}
    if dry_run:
        result["dry_run"] = True
        # `imported` is the dry-run preview of what would be written; the
        # explicit key stops an agent from reading it as "already done".
        result["would_create"] = imported
    elif repaired_links:
        result["repaired_superseded_links"] = repaired_links
    if graph is not None:
        # `graph_entities`/`graph_relations` count the records the import
        # processed; the `*_created` keys count what did not exist before.
        # MERGE updates an existing node/edge instead of duplicating it, so
        # processing a record is not the same as creating one.
        result["graph_entities"] = graph_entities
        result["graph_entities_created"] = graph_entities_created
        result["graph_relations"] = graph_relations
        result["graph_relations_created"] = graph_relations_created
    if errors:
        result["errors"] = errors[:5]
    reasons: dict[str, int] = {}
    if skipped_id_exists:
        reasons["id_exists"] = skipped_id_exists
    if skipped_duplicate_content:
        reasons["duplicate_content"] = skipped_duplicate_content
    if skipped_empty_content:
        reasons["empty_content"] = skipped_empty_content
    if errors:
        # Invalid records were skipped too; without this key the reasons did
        # not add up to `skipped` and an import that lost rows looked clean.
        reasons["invalid"] = len(errors)
    if reasons:
        result["skipped_reasons"] = reasons
    # A non-empty `skipped` must never be silent: "already imported" (fine and
    # expected on a re-run) reads very differently from "dropped because it
    # had no content" (data loss the caller should investigate).
    skipped_parts: list[str] = []
    if skipped_id_exists:
        skipped_parts.append(
            f"{skipped_id_exists} record(s) already existed with the same id "
            f"(a re-import is idempotent; the existing rows were not changed)")
    if skipped_duplicate_content:
        skipped_parts.append(
            f"{skipped_duplicate_content} record(s) without an id matched an "
            f"existing memory's content (pass skip_duplicates=false to import "
            f"them anyway)")
    if skipped_empty_content:
        skipped_parts.append(
            f"{skipped_empty_content} record(s) had no content")
    if errors:
        skipped_parts.append(
            f"{len(errors)} record(s) were invalid (see errors)")
    warnings: list[str] = []
    if dry_run:
        warnings.append(
            "dry-run: nothing was written; the counts describe what an "
            "import would do")
    if not dry_run and repaired_links:
        warnings.append(
            f"{repaired_links} superseded link(s) pointed at a memory that "
            f"was not part of the import; they were cleared so those rows "
            f"stay readable as history")
    if skipped_parts:
        warnings.append("import skipped: " + "; ".join(skipped_parts))
    return ok(STATUS_OK, warnings=warnings, **result)
