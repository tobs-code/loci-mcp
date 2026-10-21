import json
import logging

from mcp_server.config import MEMORY_TYPES, SENTINEL_FORGOTTEN, VALID_SCOPES
from mcp_server.db import get_conn
from mcp_server.inference_client import embed
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
    _store_memory,
    _supersede,
)

logger = logging.getLogger(__name__)

def remember(
    content: str,
    mem_type: str = "fact",
    entity: str | None = None,
    importance: float = 1.0,
    metadata: dict | None = None,
    scope: str | None = None,
    project: str | None = None,
    supersede: str | None = None,
    check_contradictions: bool = False,
    fuzzy_threshold: float | None = None,
) -> str:
    """Store a memory (fact, decision, learning, preference, ...).

    Automatically embedded for vector search. Attach to an entity for graph connectivity.
    If `check_contradictions` is true, returns warnings for conflicting existing facts.
    Use `supersede` to replace an existing memory by ID.
    Set fuzzy_threshold (0.0-1.0) to detect semantically similar content as duplicate.
    `scope` defaults to "project" when a project is given, else "global".
    scope="global" stores in the shared table even when a project is given.
    """
    logger.info("remember entity=%s mem_type=%s importance=%s check_contradictions=%s fuzzy=%s scope=%s",
                entity, mem_type, importance, check_contradictions, fuzzy_threshold, scope)
    project = _norm_project(project)
    if mem_type not in MEMORY_TYPES:
        return f"Invalid mem_type. Choose from: {', '.join(sorted(MEMORY_TYPES))}"
    if scope is not None and scope not in VALID_SCOPES:
        return f"Invalid scope. Choose from: {', '.join(sorted(VALID_SCOPES))}"
    content = (content or "").strip()
    if not content:
        return "ERROR: content must not be empty"
    try:
        importance = float(importance)
    except (TypeError, ValueError):
        return "ERROR: importance must be a number between 0.0 and 1.0"
    if not 0.0 <= importance <= 1.0:
        return "ERROR: importance must be between 0.0 and 1.0"
    if fuzzy_threshold is not None and not 0.0 <= fuzzy_threshold <= 1.0:
        return "ERROR: fuzzy_threshold must be between 0.0 and 1.0"
    if metadata is not None and not isinstance(metadata, dict):
        return "ERROR: metadata must be an object"
    effective_scope = scope or ("project" if project else "global")
    if effective_scope == "project" and not project:
        return "project required when scope='project'"
    store_project = project if effective_scope == "project" else None

    meta = metadata or {}
    emb = embed([content])[0]
    conn = get_conn()
    try:
        cur = conn.cursor()
        if store_project:
            _ensure_project_table(cur, store_project)

        existing = _deduplicate(cur, content, store_project, emb, fuzzy_threshold)
        if existing:
            return f"DUPLICATE: {existing}"

        contradictions = []
        if check_contradictions and entity:
            contradictions = _check_contradictions(
                cur, content, entity, store_project, emb
            )

        mem_id = _store_memory(cur, content, emb, mem_type, entity,
                               importance, meta, store_project, None,
                               scope=effective_scope)

        if supersede:
            _supersede(cur, store_project, mem_id, supersede)

        auto_superseded: list[str] = []
        if contradictions:
            for c in contradictions:
                eid = c["existing_id"]
                _supersede(cur, store_project, mem_id, eid)
                auto_superseded.append(eid)

        conn.commit()

        if check_contradictions:
            if auto_superseded:
                result = f"{mem_id} CONTRADICTIONS_CHECKED: {json.dumps({'found': True, 'superseded': contradictions})}"
            elif not entity:
                result = f'{mem_id} CONTRADICTIONS_CHECKED: {{"found": false, "skipped": "entity required"}}'
            else:
                result = f'{mem_id} CONTRADICTIONS_CHECKED: {{"found": false}}'
        else:
            result = mem_id
        return result
    except Exception as e:
        conn.rollback()
        logger.exception("remember failed")
        return f"ERROR: {e}"
    finally:
        conn.close()


def remember_batch(
    facts: list[dict],
    project: str | None = None,
    skip_duplicates: bool = True,
    fuzzy_threshold: float | None = None,
    check_contradictions: bool = False,
    scope: str | None = None,
) -> list[dict]:
    """Store multiple facts at once. Each dict needs 'content'.

    Optional keys: mem_type, entity, importance, metadata, scope, project,
    supersede (memory id), fuzzy_threshold.
    Set skip_duplicates=False to allow storing identical content multiple times.
    Set fuzzy_threshold (0.0-1.0) for semantic dedup across all facts.
    Per-fact 'fuzzy_threshold' overrides the top-level value.
    `scope` sets the default scope for all facts; per-fact 'scope' wins.
    Set check_contradictions=True to NLI-check each fact against existing
    facts of the same entity and auto-supersede contradictions.
    Invalid facts are reported individually; the remaining facts are stored.
    """
    logger.info("remember_batch count=%s skip_duplicates=%s fuzzy=%s check_contradictions=%s scope=%s",
                len(facts) if isinstance(facts, list) else -1, skip_duplicates,
                fuzzy_threshold, check_contradictions, scope)
    project = _norm_project(project)
    if scope is not None and scope not in VALID_SCOPES:
        return [{"error": f"invalid scope {scope!r}; choose from "
                          f"{', '.join(sorted(VALID_SCOPES))}"}]
    if fuzzy_threshold is not None and not 0.0 <= fuzzy_threshold <= 1.0:
        return [{"error": "fuzzy_threshold must be between 0.0 and 1.0"}]
    if not isinstance(facts, list) or not facts:
        return [{"error": "facts must be a non-empty list"}]

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
            "metadata": meta,
            "supersede": fact.get("supersede"),
        }, None))

    valid = [(i, spec) for i, (spec, err) in enumerate(prepared)
             if spec is not None]
    embeddings: dict[int, list[float]] = {}
    if valid:
        try:
            vectors = embed([spec["content"] for _, spec in valid])
        except Exception as e:
            logger.exception("remember_batch embedding failed")
            return [{"error": f"embedding failed: {e}"}]
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
                    existing = _deduplicate(cur, spec["content"], store_project,
                                            embeddings[i], spec["fuzzy_threshold"])
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

                mid = _store_memory(
                    cur, spec["content"], embeddings[i], spec["mem_type"],
                    spec["entity"], spec["importance"], spec["metadata"],
                    store_project, scope=spec["scope"],
                )

                if spec["supersede"]:
                    _supersede(cur, store_project, mid, spec["supersede"])

                auto_superseded = []
                for c in contradictions:
                    _supersede(cur, store_project, mid, c["existing_id"])
                    auto_superseded.append(c)
                cur.execute(f"RELEASE SAVEPOINT {savepoint}")
            except Exception as exc:
                cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                results.append({"id": None, "content": spec["content"],
                                "status": "error", "error": str(exc)})
                continue

            entry = {"id": mid, "content": spec["content"], "status": "stored"}
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
        return [{"error": str(e)}]
    finally:
        conn.close()
    return results


def update_memory(
    memory_id: str,
    content: str | None = None,
    importance: float | None = None,
    metadata: dict | None = None,
    project: str | None = None,
) -> str:
    """Update an existing memory's content, importance, or metadata.

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
            return "ERROR: content must not be empty"
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
            return "ERROR: importance must be a number between 0.0 and 1.0"
        if not 0.0 <= importance <= 1.0:
            return "ERROR: importance must be between 0.0 and 1.0"
        sets.append("importance = %s")
        params.append(importance)
    if metadata is not None:
        if not isinstance(metadata, dict):
            return "ERROR: metadata must be an object"
        sets.append("metadata = %s::jsonb")
        params.append(json.dumps(metadata))

    if not sets:
        return "Nothing to update"
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
                f"UPDATE {table} SET {', '.join(sets)} WHERE id = %s",
                params + [memory_id],
            )
            if cur.rowcount:
                updated += cur.rowcount
                break
        conn.commit()
        if not updated:
            return f"NOT FOUND: {memory_id}"
    except Exception as e:
        conn.rollback()
        return f"ERROR: {e}"
    finally:
        conn.close()
    return memory_id


def forget(
    memory_id: str | None = None,
    entity_name: str | None = None,
    project: str | None = None,
    soft: bool = True,
) -> str:
    """Remove a fact by memory_id or delete an entity and all its memories.

    When soft=True (default), marks as superseded instead of hard delete.
    Without `project`, the id/entity is looked up in every table; deleting an
    entity also removes its memories from all scopes and its graph node.
    """
    logger.info("forget memory_id=%s entity_name=%s soft=%s", memory_id, entity_name, soft)
    project = _norm_project(project)
    if not memory_id and not entity_name:
        return "ERROR: memory_id or entity_name is required"
    conn = get_conn()
    affected = 0
    try:
        cur = conn.cursor()
        if memory_id:
            tables = _all_mem_tables(cur) if not project else [_mem_table(project)]
            if project and not _table_exists(cur, tables[0]):
                return f"NOT FOUND: {memory_id}"
            for table in tables:
                if soft:
                    cur.execute(
                        f"UPDATE {table} SET superseded_by = %s, deleted_at = NOW() "
                        f"WHERE id = %s AND deleted_at IS NULL",
                        (SENTINEL_FORGOTTEN, memory_id),
                    )
                else:
                    cur.execute(f"DELETE FROM {table} WHERE id = %s", (memory_id,))
                affected += cur.rowcount
            if not affected:
                conn.rollback()
                return f"NOT FOUND: {memory_id}"
        if entity_name:
            cur.execute("SET search_path TO ag_catalog, public")
            en_json = json.dumps(entity_name)
            if project:
                label = _scope_label(project)
                cur.execute(
                    f"SELECT * FROM cypher('memory_graph', $$ "
                    f"MATCH (n:{label} {{name: {en_json}}}) DETACH DELETE n $$) AS (r agtype)",
                )
                cur.execute(
                    "DELETE FROM entities WHERE name = %s AND project = %s",
                    (entity_name, _scope_key(project)),
                )
                affected += cur.rowcount
            else:
                cur.execute(
                    f"SELECT * FROM cypher('memory_graph', $$ "
                    f"MATCH (n {{name: {en_json}}}) DETACH DELETE n $$) AS (r agtype)",
                )
                cur.execute("DELETE FROM entities WHERE name = %s", (entity_name,))
                affected += cur.rowcount
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
                affected += cur.rowcount
            if not affected:
                conn.rollback()
                return f"NOT FOUND: {entity_name}"
        conn.commit()
    except Exception as e:
        conn.rollback()
        return f"ERROR: {e}"
    finally:
        conn.close()
    action = "Soft-forgotten" if soft else "Deleted"
    target = memory_id or entity_name
    return f"{action} {target} ({affected} memories)"
