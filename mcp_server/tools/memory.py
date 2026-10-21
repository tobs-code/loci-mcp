import json
import logging
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from mcp_server.config import (
    DEDUPE_MODES,
    MEMORY_TYPES,
    SENTINEL_FORGOTTEN,
    VALID_SCOPES,
)
from mcp_server.db import get_conn
from mcp_server.envelope import (
    STATUS_DELETED,
    STATUS_DUPLICATE,
    STATUS_ERROR,
    STATUS_NOT_FOUND,
    STATUS_PARTIAL,
    STATUS_SOFT_FORGOTTEN,
    STATUS_STORED,
    STATUS_UPDATED,
    fail,
    ok,
)
from mcp_server.inference_client import embed
from mcp_server.inference_client import nli_configured
from mcp_server.scopes import (
    _all_mem_tables,
    _ensure_project_table,
    _mem_table,
    _norm_project,
    _scope_key,
    _scope_label,
    _table_exists,
)
from mcp_server.scoring import _content_hash
from mcp_server.storage import (
    _check_contradictions,
    _deduplicate,
    _find_by_content,
    _find_last_for_entity,
    _store_memory,
    _supersede,
)

logger = logging.getLogger(__name__)


class BatchFact(BaseModel):
    """One fact of a batch. Only `content` is required."""

    model_config = ConfigDict(extra="allow")

    content: Any = Field(json_schema_extra={"type": "string"},
                          description="The fact to store (required, non-empty).")
    mem_type: Any = Field(default="fact", json_schema_extra={
        "type": "string",
        "enum": ["fact", "decision", "learning", "preference", "pattern",
                 "incident", "runbook", "context", "observation", "convention",
                 "command", "note"],
    }, description="Kind of memory (default: fact).")
    entity: Any = Field(default=None, json_schema_extra={"type": ["string", "null"]},
                         description="Entity this memory belongs to; links it in the graph.")
    importance: Any = Field(default=None, json_schema_extra={"type": ["number", "null"]},
                            description="Importance in 0.0-1.0 (default 1.0); scales decay weighting.")
    metadata: Any = Field(default=None, json_schema_extra={"type": ["object", "null"]},
                          description="Free-form JSON object stored with the memory.")
    scope: Any = Field(default=None, json_schema_extra={
        "type": ["string", "null"], "enum": ["global", "project", None],
    }, description="Where to store: 'project' (needs a project) or 'global'.")
    project: Any = Field(default=None, json_schema_extra={"type": ["string", "null"]},
                         description="Overrides the batch project for this fact.")
    supersede: Any = Field(default=None, json_schema_extra={"type": ["string", "null"]},
                           description="Memory id this fact replaces.")
    supersede_content: Any = Field(default=None, json_schema_extra={"type": ["string", "null"]},
                                   description="Exact content of the memory this fact replaces.")
    supersede_last: Any = Field(default=None, json_schema_extra={"type": ["boolean", "null"]},
                                description="Replace the newest memory of the same entity.")
    fuzzy_threshold: Any = Field(default=None, json_schema_extra={"type": ["number", "null"]},
                                 description="Similarity threshold 0.0-1.0 for semantic dedup.")
    dedupe: Any = Field(default=None, json_schema_extra={
        "type": ["string", "null"], "enum": ["exact", "semantic", "off", None],
    }, description="Duplicate detection: 'exact' (content hash), 'semantic' (vector similarity) or 'off'.")


def remember(
    content: Annotated[str, Field(description="The fact to remember (non-empty).")],
    mem_type: Annotated[
        Literal["fact", "decision", "learning", "preference", "pattern", "incident",
                "runbook", "context", "observation", "convention", "command", "note"],
        Field(description="Kind of memory (default: fact).")] = "fact",
    entity: Annotated[
        str | None,
        Field(description="Entity this memory belongs to; connects it in the graph "
                          "and enables graph_boost / find_connections.")] = None,
    importance: Annotated[
        float,
        Field(description="Importance in 0.0-1.0 (default 1.0). Scales the decay "
                          "weight, so low-importance memories fade faster.")] = 1.0,
    metadata: Annotated[
        dict[str, Any] | None,
        Field(description="Free-form JSON object stored with the memory.")] = None,
    scope: Annotated[
        Literal["global", "project"] | None,
        Field(description="Where to store the memory: 'project' (requires a "
                          "project) or 'global' (shared table). Defaults to "
                          "'project' when a project is given, else 'global'.")] = None,
    project: Annotated[
        str | None,
        Field(description="Project scope. 'global' is reserved and means the "
                          "shared scope.")] = None,
    supersede: Annotated[
        str | None,
        Field(description="Memory id that this fact replaces; the old memory is "
                          "marked as superseded.")] = None,
    check_contradictions: Annotated[
        bool,
        Field(description="Check the new fact against existing facts of the same "
                          "entity (NLI) and auto-supersede contradictions. Needs "
                          "an entity and a configured LLM provider.")] = False,
    fuzzy_threshold: Annotated[
        float | None,
        Field(description="Similarity threshold 0.0-1.0 for semantic dedup "
                          "(default 0.9). Only used by dedupe='semantic'.")] = None,
    dedupe: Annotated[
        Literal["exact", "semantic", "off"],
        Field(description="Duplicate detection: 'exact' (content hash, default), "
                          "'semantic' (also vector similarity) or 'off'.")] = "exact",
    supersede_content: Annotated[
        str | None,
        Field(description="Exact content of the memory this fact replaces.")] = None,
    supersede_last: Annotated[
        bool,
        Field(description="Replace the newest memory of the same entity.")] = False,
) -> dict[str, Any]:
    """Store a memory (fact, decision, learning, preference, ...).

    Automatically embedded for vector search. Attach to an entity for graph connectivity.
    If `check_contradictions` is true, returns warnings for conflicting existing facts.
    Use `supersede` to replace an existing memory by ID, `supersede_content` to
    replace the fact with the same wording, or `supersede_last` to replace the
    newest fact of the same entity.
    `dedupe` selects how duplicates are detected: "exact" (default, content
    hash), "semantic" (also vector similarity, see `fuzzy_threshold`) or "off".
    `scope` defaults to "project" when a project is given, else "global".
    scope="global" stores in the shared table even when a project is given.

    Returns the standard envelope: {"ok", "status", "id", "error", "warnings"}.
    """
    logger.info("remember entity=%s mem_type=%s importance=%s check_contradictions=%s fuzzy=%s scope=%s dedupe=%s",
                entity, mem_type, importance, check_contradictions, fuzzy_threshold, scope, dedupe)
    project = _norm_project(project)
    if mem_type not in MEMORY_TYPES:
        return fail(STATUS_ERROR,
                    f"Invalid mem_type. Choose from: {', '.join(sorted(MEMORY_TYPES))}")
    if scope is not None and scope not in VALID_SCOPES:
        return fail(STATUS_ERROR,
                    f"Invalid scope. Choose from: {', '.join(sorted(VALID_SCOPES))}")
    if dedupe not in DEDUPE_MODES:
        return fail(STATUS_ERROR,
                    f"Invalid dedupe. Choose from: {', '.join(sorted(DEDUPE_MODES))}")
    content = (content or "").strip()
    if not content:
        return fail(STATUS_ERROR, "content must not be empty")
    try:
        importance = float(importance)
    except (TypeError, ValueError):
        return fail(STATUS_ERROR, "importance must be a number between 0.0 and 1.0")
    if not 0.0 <= importance <= 1.0:
        return fail(STATUS_ERROR, "importance must be between 0.0 and 1.0")
    if fuzzy_threshold is not None and not 0.0 <= fuzzy_threshold <= 1.0:
        return fail(STATUS_ERROR, "fuzzy_threshold must be between 0.0 and 1.0")
    if metadata is not None and not isinstance(metadata, dict):
        return fail(STATUS_ERROR, "metadata must be an object")
    effective_scope = scope or ("project" if project else "global")
    if effective_scope == "project" and not project:
        return fail(STATUS_ERROR, "project required when scope='project'")
    store_project = project if effective_scope == "project" else None

    meta = metadata or {}
    emb = embed([content])[0]
    conn = get_conn()
    try:
        cur = conn.cursor()
        if store_project:
            _ensure_project_table(cur, store_project)

        existing = _deduplicate(cur, content, store_project, emb, fuzzy_threshold,
                                dedupe=dedupe)
        if existing:
            return ok(STATUS_DUPLICATE, id=existing,
                      warnings=["memory already exists; nothing stored"])

        contradictions = []
        nli_unavailable = check_contradictions and nli_configured() is False
        if check_contradictions and entity and not nli_unavailable:
            contradictions = _check_contradictions(
                cur, content, entity, store_project, emb
            )

        # Resolve the replaced memories before storing: "supersede_last" must
        # see the state before this fact exists.
        superseded: list[str] = []
        try:
            superseded = _supersede_targets(cur, store_project, entity,
                                            supersede, supersede_content,
                                            supersede_last)
        except ValueError as e:
            conn.rollback()
            return fail(STATUS_ERROR, str(e))

        mem_id = _store_memory(cur, content, emb, mem_type, entity,
                               importance, meta, store_project, None,
                               scope=effective_scope)

        for old_id in superseded:
            _supersede(cur, store_project, mem_id, old_id)

        for c in contradictions:
            eid = c["existing_id"]
            if eid not in superseded:
                _supersede(cur, store_project, mem_id, eid)
                superseded.append(eid)

        conn.commit()

        warnings: list[str] = []
        if check_contradictions and entity and contradictions:
            warnings.append(
                f"superseded {len(contradictions)} contradicting memory/facts"
            )
        elif nli_unavailable:
            warnings.append("contradiction check skipped: NLI provider is not configured")
        elif check_contradictions and not entity:
            warnings.append("contradiction check skipped: entity required")
        if superseded and not check_contradictions:
            warnings.append(f"superseded {len(superseded)} memory/facts")
        payload: dict[str, Any] = {"superseded": superseded}
        if check_contradictions:
            payload["contradictions"] = {"found": bool(contradictions),
                                         "superseded": contradictions}
        return ok(STATUS_STORED, id=mem_id, warnings=warnings, **payload)
    except Exception as e:
        conn.rollback()
        logger.exception("remember failed")
        return fail(STATUS_ERROR, str(e))
    finally:
        conn.close()


def _supersede_targets(cur, store_project: str | None, entity: str | None,
                       supersede: str | None, supersede_content: str | None,
                       supersede_last: bool) -> list[str]:
    """Resolve which existing memories a new fact replaces."""
    targets: list[str] = []
    if supersede:
        targets.append(supersede)
    if supersede_content:
        found = _find_by_content(cur, supersede_content, store_project, entity)
        if found:
            targets.append(found)
    if supersede_last:
        if not entity:
            raise ValueError("supersede_last requires an entity")
        found = _find_last_for_entity(cur, entity, store_project)
        if found:
            targets.append(found)
    # Keep order, drop duplicates.
    return list(dict.fromkeys(targets))


def remember_batch(
    facts: Annotated[
        list[BatchFact],
        Field(description="Facts to store. Each dict needs 'content'; invalid "
                          "entries are reported individually and the rest is "
                          "still stored.")],
    project: Annotated[
        str | None,
        Field(description="Default project scope for all facts (per-fact "
                          "project wins).")]= None,
    skip_duplicates: Annotated[
        bool,
        Field(description="Skip facts that already exist instead of storing "
                          "them again.")] = True,
    fuzzy_threshold: Annotated[
        float | None,
        Field(description="Similarity threshold 0.0-1.0 for semantic dedup "
                          "across all facts.")] = None,
    check_contradictions: Annotated[
        bool,
        Field(description="NLI-check every fact against existing facts of the "
                          "same entity and auto-supersede contradictions.")] = False,
    scope: Annotated[
        Literal["global", "project"] | None,
        Field(description="Default scope for all facts (per-fact scope wins).")]= None,
    dedupe: Annotated[
        Literal["exact", "semantic", "off"],
        Field(description="Duplicate detection: 'exact' (default), 'semantic' or "
                          "'off'.")] = "exact",
) -> dict[str, Any]:
    """Store multiple facts at once. Each dict needs 'content'.

    Optional keys: mem_type (one of the memory types), entity, importance,
    metadata, scope ("project" or "global"), project, supersede (memory id
    or content), supersede_last, fuzzy_threshold, and dedupe ("exact",
    "semantic", or "off").
    Set skip_duplicates=False to allow storing identical content multiple times.
    Set fuzzy_threshold (0.0-1.0) for semantic dedup across all facts.
    Per-fact 'fuzzy_threshold' overrides the top-level value.
    `dedupe` selects how duplicates are detected: "exact" (default, content
    hash), "semantic" (also vector similarity) or "off".
    `scope` sets the default scope for all facts; per-fact 'scope' wins.
    Set check_contradictions=True to NLI-check each fact against existing
    facts of the same entity and auto-supersede contradictions.
    Invalid facts are reported individually; the remaining facts are stored.
    """
    logger.info("remember_batch count=%s skip_duplicates=%s fuzzy=%s check_contradictions=%s scope=%s dedupe=%s",
                len(facts) if isinstance(facts, list) else -1, skip_duplicates,
                fuzzy_threshold, check_contradictions, scope, dedupe)
    project = _norm_project(project)
    if scope is not None and scope not in VALID_SCOPES:
        return fail(STATUS_ERROR,
                    f"invalid scope {scope!r}; choose from "
                    f"{', '.join(sorted(VALID_SCOPES))}")
    if dedupe not in DEDUPE_MODES:
        return fail(STATUS_ERROR,
                    f"invalid dedupe {dedupe!r}; choose from "
                    f"{', '.join(sorted(DEDUPE_MODES))}")
    if fuzzy_threshold is not None and not 0.0 <= fuzzy_threshold <= 1.0:
        return fail(STATUS_ERROR, "fuzzy_threshold must be between 0.0 and 1.0")
    if not isinstance(facts, list) or not facts:
        return fail(STATUS_ERROR, "facts must be a non-empty list")

    facts = [fact.model_dump(exclude_unset=True) if isinstance(fact, BatchFact) else fact
             for fact in facts]


    # Validate everything before embedding so one malformed fact cannot abort
    # the batch with a KeyError.
    prepared: list[tuple[dict | None, str | None]] = []
    for fact in facts:
        if not isinstance(fact, dict):
            prepared.append((None, "fact must be an object"))
            continue
        content = fact.get("content")
        content = content.strip() if isinstance(content, str) else ""
        if not content:
            prepared.append((None, "content is required"))
            continue
        mem_type = fact.get("mem_type", "fact")
        if mem_type not in MEMORY_TYPES:
            prepared.append((None, f"invalid mem_type {mem_type!r}"))
            continue
        try:
            importance = float(fact.get("importance", 1.0))
        except (TypeError, ValueError):
            prepared.append((None, "importance must be a number between 0.0 and 1.0"))
            continue
        if not 0.0 <= importance <= 1.0:
            prepared.append((None, "importance must be between 0.0 and 1.0"))
            continue
        fp = _norm_project(fact.get("project", project))
        fscope = fact.get("scope") or scope or ("project" if fp else "global")
        if fscope not in VALID_SCOPES:
            prepared.append((None, f"invalid scope {fscope!r}"))
            continue
        if fscope == "project" and not fp:
            prepared.append((None, "project required when scope='project'"))
            continue
        ft = fact.get("fuzzy_threshold", fuzzy_threshold)
        if ft is not None and not 0.0 <= ft <= 1.0:
            prepared.append((None, "fuzzy_threshold must be between 0.0 and 1.0"))
            continue
        meta = fact.get("metadata") or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except json.JSONDecodeError:
                prepared.append((None, "metadata must be valid JSON"))
                continue
        if not isinstance(meta, dict):
            prepared.append((None, "metadata must be an object"))
            continue
        prepared.append(({
            "content": content,
            "mem_type": mem_type,
            "importance": importance,
            "project": fp if fscope == "project" else None,
            "scope": fscope,
            "entity": fact.get("entity"),
            "fuzzy_threshold": ft,
            "dedupe": fact.get("dedupe"),
            "metadata": meta,
            "supersede": fact.get("supersede"),
            "supersede_content": fact.get("supersede_content"),
            "supersede_last": bool(fact.get("supersede_last")),
        }, None))

    valid = [(i, spec) for i, (spec, err) in enumerate(prepared)
             if spec is not None]
    embeddings: dict[int, list[float]] = {}
    if valid:
        try:
            vectors = embed([spec["content"] for _, spec in valid])
        except Exception as e:
            logger.exception("remember_batch embedding failed")
            return fail(STATUS_ERROR, f"embedding failed: {e}")
        embeddings = {i: vec for (i, _), vec in zip(valid, vectors)}

    conn = get_conn()
    results: list[dict] = []
    ensured: set[str] = set()
    try:
        cur = conn.cursor()
        for i, (spec, err) in enumerate(prepared):
            if spec is None:
                raw = facts[i].get("content", "") if isinstance(facts[i], dict) else ""
                results.append({"id": None, "content": raw, "status": "error",
                                "error": err})
                continue
            store_project = spec["project"]
            savepoint = f"fact_{i}"
            cur.execute(f"SAVEPOINT {savepoint}")
            try:
                if store_project and store_project not in ensured:
                    _ensure_project_table(cur, store_project)
                    ensured.add(store_project)

                if skip_duplicates:
                    fact_dedupe = spec.get("dedupe") or dedupe
                    existing = _deduplicate(cur, spec["content"], store_project,
                                            embeddings[i], spec["fuzzy_threshold"],
                                            dedupe=fact_dedupe)
                    if existing:
                        cur.execute(f"RELEASE SAVEPOINT {savepoint}")
                        results.append({"id": existing, "content": spec["content"],
                                        "status": "duplicate"})
                        continue

                contradictions = []
                if check_contradictions and spec["entity"]:
                    contradictions = _check_contradictions(
                        cur, spec["content"], spec["entity"], store_project,
                        embeddings[i],
                    )

                # Resolve replaced memories before this fact is stored.
                superseded: list[str] = _supersede_targets(
                    cur, store_project, spec["entity"], spec["supersede"],
                    spec["supersede_content"], spec["supersede_last"])

                mid = _store_memory(
                    cur, spec["content"], embeddings[i], spec["mem_type"],
                    spec["entity"], spec["importance"], spec["metadata"],
                    store_project, scope=spec["scope"],
                )

                for old_id in superseded:
                    _supersede(cur, store_project, mid, old_id)

                auto_superseded = []
                for c in contradictions:
                    if c["existing_id"] in superseded:
                        continue
                    _supersede(cur, store_project, mid, c["existing_id"])
                    auto_superseded.append(c)
                cur.execute(f"RELEASE SAVEPOINT {savepoint}")
            except Exception as exc:
                cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                results.append({"id": None, "content": spec["content"],
                                "status": "error", "error": str(exc)})
                continue

            entry = {"id": mid, "content": spec["content"], "status": "stored",
                     "superseded": superseded}
            if check_contradictions:
                entry["contradictions"] = {
                    "found": bool(auto_superseded),
                    "superseded": auto_superseded,
                }
                if not spec["entity"]:
                    entry["contradictions"]["skipped"] = "entity required"
            results.append(entry)
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.exception("remember_batch failed")
        return fail(STATUS_ERROR, str(e))
    finally:
        conn.close()
    failed = [r for r in results if r.get("status") == "error"]
    if failed and len(failed) == len(results):
        return fail(STATUS_ERROR, "no fact could be stored",
                    count=len(results), results=results)
    status = STATUS_PARTIAL if failed else "ok"
    warnings = [f"{len(failed)} fact(s) failed"] if failed else []
    return ok(status, count=len(results), results=results, warnings=warnings)


def update_memory(
    memory_id: Annotated[
        str,
        Field(description="Id of the memory to update.")],
    content: Annotated[
        str | None,
        Field(description="New content; re-embeds the memory.")] = None,
    importance: Annotated[
        float | None,
        Field(description="New importance in 0.0-1.0.")] = None,
    metadata: Annotated[
        dict | None,
        Field(description="Replacement metadata object.")] = None,
    project: Annotated[
        str | None,
        Field(description="Project scope. Without it the id is looked up in "
                          "every scope.")] = None,
    entity: Annotated[
        str | None,
        Field(description="New entity the memory belongs to.")] = None,
    mem_type: Annotated[
        str | None,
        Field(description="New memory type (fact, decision, learning, "
                          "preference, pattern, incident, runbook, context, "
                          "observation, convention, command, note).")] = None,
) -> dict[str, Any]:
    """Update an existing memory's content, importance, metadata, entity or type.

    Without `project`, the memory is looked up in every table.
    """
    logger.info("update_memory id=%s has_content=%s has_importance=%s has_metadata=%s",
                memory_id, content is not None, importance is not None, metadata is not None)
    project = _norm_project(project)
    sets: list[str] = []
    params: list[Any] = []

    if content is not None:
        content = (content or "").strip()
        if not content:
            return fail(STATUS_ERROR, "content must not be empty")
        emb = embed([content])[0]
        sets.append("content = %s")
        params.append(content)
        sets.append("content_hash = %s")
        params.append(_content_hash(content))
        sets.append("embedding = %s::vector")
        params.append(emb)
    if importance is not None:
        try:
            importance = float(importance)
        except (TypeError, ValueError):
            return fail(STATUS_ERROR, "importance must be a number between 0.0 and 1.0")
        if not 0.0 <= importance <= 1.0:
            return fail(STATUS_ERROR, "importance must be between 0.0 and 1.0")
        sets.append("importance = %s")
        params.append(importance)
    if metadata is not None:
        if not isinstance(metadata, dict):
            return fail(STATUS_ERROR, "metadata must be an object")
        sets.append("metadata = %s::jsonb")
        params.append(json.dumps(metadata))
    if entity is not None:
        entity = (entity or "").strip()
        if not entity:
            return fail(STATUS_ERROR, "entity must not be empty")
        sets.append("entity = %s")
        params.append(entity)
    if mem_type is not None:
        if mem_type not in MEMORY_TYPES:
            return fail(STATUS_ERROR,
                        f"Invalid mem_type. Choose from: "
                        f"{', '.join(sorted(MEMORY_TYPES))}")
        sets.append("mem_type = %s")
        params.append(mem_type)

    if not sets:
        return fail(STATUS_ERROR, "nothing to update", id=memory_id)
    # Updated memories must also age like new ones for decay scoring.
    sets.append("updated_at = NOW()")

    conn = get_conn()
    try:
        cur = conn.cursor()
        tables = ([_mem_table(project), "memories"] if project
                  else _all_mem_tables(cur))
        updated = 0
        for table in tables:
            cur.execute(
                f"UPDATE {table} SET {', '.join(sets)} "
                "WHERE id = %s AND superseded_by IS NULL AND deleted_at IS NULL",
                params + [memory_id],
            )
            if cur.rowcount:
                updated += cur.rowcount
                break
        conn.commit()
        if not updated:
            return fail(STATUS_NOT_FOUND, f"NOT FOUND: {memory_id}", id=memory_id)
    except Exception as e:
        conn.rollback()
        return fail(STATUS_ERROR, str(e), id=memory_id)
    finally:
        conn.close()
    return ok(STATUS_UPDATED, id=memory_id)


def forget(
    memory_id: Annotated[
        str | None,
        Field(description="Id of a single memory to remove.")] = None,
    entity_name: Annotated[
        str | None,
        Field(description="Entity whose memories (and graph node) are removed. "
                          "Alias: `entity`.")] = None,
    project: Annotated[
        str | None,
        Field(description="Scope to act in. Without it the id/entity is looked "
                          "up in every scope.")] = None,
    soft: Annotated[
        bool,
        Field(description="True (default) hides the memory; False deletes it "
                          "permanently.")] = True,
    entity: Annotated[
        str | None,
        Field(description="Entity whose memories (and graph node) are removed. "
                          "Alias: `entity_name`.")] = None,
) -> dict[str, Any]:
    """Remove a fact by memory_id or delete an entity and all its memories.

    The entity can be passed as `entity` or `entity_name`.
    When soft=True (default), hides the memory by setting deleted_at and a
    forgotten sentinel; it remains available only for database-level recovery.
    Without `project`, the id/entity is looked up in every table; deleting an
    entity also removes its memories from all scopes and its graph node.
    """
    entity_name = entity or entity_name
    logger.info("forget memory_id=%s entity_name=%s soft=%s", memory_id, entity_name, soft)
    project = _norm_project(project)
    if not memory_id and not entity_name:
        return fail(STATUS_ERROR, "memory_id or entity_name is required")
    conn = get_conn()
    counts = {"memories": 0, "entities": 0, "relations": 0}
    try:
        cur = conn.cursor()
        if memory_id:
            tables = _all_mem_tables(cur) if not project else [_mem_table(project)]
            if project and not _table_exists(cur, tables[0]):
                return fail(STATUS_NOT_FOUND, f"NOT FOUND: {memory_id}", id=memory_id)
            for table in tables:
                if soft:
                    cur.execute(
                        f"UPDATE {table} SET superseded_by = %s, deleted_at = NOW() "
                        f"WHERE id = %s AND deleted_at IS NULL",
                        (SENTINEL_FORGOTTEN, memory_id),
                    )
                else:
                    cur.execute(f"DELETE FROM {table} WHERE id = %s", (memory_id,))
                counts["memories"] += cur.rowcount
            if not counts["memories"]:
                conn.rollback()
                return fail(STATUS_NOT_FOUND, f"NOT FOUND: {memory_id}", id=memory_id)
        if entity_name:
            cur.execute("SET search_path TO ag_catalog, public")
            en_json = json.dumps(entity_name)
            scope = _scope_key(project)
            cur.execute(
                "SELECT id FROM entities WHERE name = %s AND project = %s",
                (entity_name, scope),
            )
            entity_ids = [r[0] for r in cur.fetchall()]
            if entity_ids:
                cur.execute(
                    "SELECT COUNT(*) FROM relations "
                    "WHERE source_id = ANY(%s) OR target_id = ANY(%s)",
                    (entity_ids, entity_ids),
                )
                relation_count = cur.fetchone()
                counts["relations"] = relation_count[0] if relation_count else 0
            if project:
                label = _scope_label(project)
                cur.execute(
                    f"SELECT * FROM cypher('memory_graph', $$ "
                    f"MATCH (n:{label} {{name: {en_json}}}) DETACH DELETE n $$) AS (r agtype)",
                )
                cur.execute(
                    "DELETE FROM entities WHERE name = %s AND project = %s",
                    (entity_name, scope),
                )
            else:
                cur.execute(
                    f"SELECT * FROM cypher('memory_graph', $$ "
                    f"MATCH (n {{name: {en_json}}}) DETACH DELETE n $$) AS (r agtype)",
                )
                cur.execute("DELETE FROM entities WHERE name = %s", (entity_name,))
            counts["entities"] = cur.rowcount
            # Only the selected scope's memories: forgetting an entity in one
            # project must not wipe the global facts of the same entity.
            tables = (_all_mem_tables(cur) if not project
                      else [_mem_table(project)])
            for tbl in dict.fromkeys(tables):
                if not _table_exists(cur, tbl):
                    continue
                if soft:
                    cur.execute(f"UPDATE {tbl} SET superseded_by = %s, deleted_at = NOW() "
                                f"WHERE entity = %s AND deleted_at IS NULL",
                               (SENTINEL_FORGOTTEN, entity_name))
                else:
                    cur.execute(f"DELETE FROM {tbl} WHERE entity = %s", (entity_name,))
                counts["memories"] += cur.rowcount
                relation_context_filter = (
                    "metadata->>'relation_context' = 'true' "
                    "AND (metadata->>'from' = %s OR metadata->>'to' = %s)"
                )
                if soft:
                    cur.execute(
                        f"UPDATE {tbl} SET superseded_by = %s, deleted_at = NOW() "
                        f"WHERE deleted_at IS NULL AND {relation_context_filter}",
                        (SENTINEL_FORGOTTEN, entity_name, entity_name),
                    )
                else:
                    cur.execute(
                        f"DELETE FROM {tbl} WHERE {relation_context_filter}",
                        (entity_name, entity_name),
                    )
                counts["memories"] += cur.rowcount
            if not (counts["memories"] or counts["entities"] or counts["relations"]):
                conn.rollback()
                return fail(STATUS_NOT_FOUND, f"NOT FOUND: {entity_name}")
        conn.commit()
    except Exception as e:
        conn.rollback()
        return fail(STATUS_ERROR, str(e))
    finally:
        conn.close()
    status = STATUS_SOFT_FORGOTTEN if soft else STATUS_DELETED
    target = memory_id or entity_name
    return ok(status, id=memory_id, entity=entity_name, counts=counts,
              message=f"{'Soft-forgotten' if soft else 'Deleted'} {target}")
