import json
import logging
from typing import Any

from mcp_server.config import (
    DEFAULT_DUPLICATE_THRESHOLD,
    NEAR_DUPLICATE_THRESHOLD,
    SENTINEL_FORGOTTEN,
)
from mcp_server.inference_client import nli
from mcp_server.scopes import _mem_table, _retrieval_tables
from mcp_server.scoring import _content_hash, _gen_id, _now

logger = logging.getLogger(__name__)

def _store_memory(
    cur,
    content: str,
    embedding_vec: list[float] | None,
    mem_type: str,
    entity: str | None,
    importance: float,
    metadata: dict,
    project: str | None,
    supersede: str | None = None,
    scope: str | None = None,
) -> str:
    mem_id = _gen_id()
    now = _now()
    table = _mem_table(project)
    c_hash = _content_hash(content)
    effective_scope = scope or ("project" if project else "global")
    cur.execute(
        f"""INSERT INTO {table}
            (id, content, content_hash, embedding, mem_type, scope, entity,
             importance, metadata, superseded_by, created_at, updated_at)
        VALUES (%s, %s, %s, %s::vector, %s, %s, %s, %s, %s, %s, %s, %s)""",
        (mem_id, content, c_hash, embedding_vec, mem_type, effective_scope, entity,
         importance, json.dumps(metadata), supersede, now, now),
    )
    return mem_id


def _supersede(cur, project: str | None, new_id: str, old_id: str) -> None:
    """Mark old_id as superseded by new_id in the scoped and the global table."""
    table = _mem_table(project)
    cur.execute(
        f"UPDATE {table} SET superseded_by = %s, updated_at = NOW() WHERE id = %s",
        (new_id, old_id),
    )
    if project:
        cur.execute(
            "UPDATE memories SET superseded_by = %s, updated_at = NOW() WHERE id = %s",
            (new_id, old_id),
        )


def _deduplicate(cur, content: str, project: str | None,
                 embedding_vec: list[float] | None = None,
                 fuzzy_threshold: float | None = None,
                 dedupe: str = "exact",
                 exclude_id: str | None = None) -> tuple[str, str, float | None] | None:
    """Return the id of an existing memory that says the same thing.

    `dedupe` selects the mode: "exact" (content hash), "semantic" (hash plus
    vector similarity) or "off" (no check at all).

    Returns a tuple (memory id, mode, similarity). The mode tells the caller
    *why* the memory was seen as a duplicate, and the similarity is the
    measured cosine distance of a semantic match (None for an exact one).
    """
    if dedupe == "off":
        return None

    c_hash = _content_hash(content)
    table = _mem_table(project)
    exclude_sql = " AND id <> %s" if exclude_id else ""
    exclude_params = [exclude_id] if exclude_id else []
    cur.execute(
        f"SELECT id FROM {table} WHERE content_hash = %s "
        f"AND superseded_by IS NULL AND deleted_at IS NULL{exclude_sql} LIMIT 1",
        [c_hash] + exclude_params,
    )
    row = cur.fetchone()
    if row:
        return row[0], "exact", None

    if embedding_vec is not None and dedupe == "semantic":
        threshold = fuzzy_threshold if fuzzy_threshold is not None else DEFAULT_DUPLICATE_THRESHOLD
        max_dist = 1.0 - threshold
        cur.execute(
            f"SELECT id, embedding <=> %s::vector AS dist "
            f"FROM {table} WHERE superseded_by IS NULL AND deleted_at IS NULL AND embedding IS NOT NULL"
            f"{exclude_sql} "
            f"ORDER BY dist ASC LIMIT 1",
            [embedding_vec] + exclude_params,
        )
        row = cur.fetchone()
        if row and float(row[1]) <= max_dist:
            return row[0], "semantic", round(1.0 - float(row[1]), 4)

    return None


def _find_near_duplicate(cur, content: str, project: str | None,
                         embedding_vec: list[float] | None = None,
                         exclude_id: str | None = None) -> dict | None:
    """The active memory that is closest to `content` above the near-dup floor.

    Runs only when `_deduplicate` found no match: the dedupe thresholds say
    "different fact", but a similarity of at least `NEAR_DUPLICATE_THRESHOLD`
    still reads like a reworded copy of an existing memory. Storing it anyway
    is allowed (the caller may want both versions) - this function only
    reports the match so the response can warn instead of forking silently.

    Returns {"id", "similarity"} or None. Exact content hashes never land
    here: `_deduplicate` already reports them before this check runs.
    """
    if embedding_vec is None:
        return None
    c_hash = _content_hash(content)
    table = _mem_table(project)
    exclude_sql = " AND id <> %s" if exclude_id else ""
    params: list[Any] = [embedding_vec, c_hash]
    if exclude_id:
        params.append(exclude_id)
    cur.execute(
        f"SELECT id, embedding <=> %s::vector AS dist FROM {table} "
        f"WHERE content_hash <> %s "
        f"AND superseded_by IS NULL AND deleted_at IS NULL "
        f"AND embedding IS NOT NULL{exclude_sql} "
        f"ORDER BY dist ASC LIMIT 1",
        params,
    )
    row = cur.fetchone()
    if not row:
        return None
    similarity = round(1.0 - float(row[1]), 4)
    if similarity >= NEAR_DUPLICATE_THRESHOLD:
        return {"id": row[0], "similarity": similarity}
    return None


def _scope_of_id(cur, memory_id: str, store_project: str | None) -> str | None:
    """Which scope holds `memory_id`: "project", "global" or None (nowhere).

    `supersede=<id>` and a contradiction candidate can name a memory of a
    different scope than the one the new fact is written to. A project write
    that invalidates a shared (global) fact hides that fact in EVERY scope,
    which the response has to say - this function tells the caller when that
    happened. Ids are unique per scope, so the store table is checked first.
    """
    if store_project:
        cur.execute(f"SELECT 1 FROM {_mem_table(store_project)} WHERE id = %s",
                    (memory_id,))
        if cur.fetchone():
            return "project"
    cur.execute("SELECT 1 FROM memories WHERE id = %s", (memory_id,))
    return "global" if cur.fetchone() else None


def _find_superseded_match(
    cur,
    content: str,
    project: str | None,
    embedding_vec: list[float] | None = None,
    fuzzy_threshold: float | None = None,
    dedupe: str = "exact",
) -> dict | None:
    """Look for a *superseded* memory that matches the new content.

    A fact can be re-added after it was replaced by a newer one. That is not
    necessarily wrong (an agent may intentionally revert a decision), so the
    caller still stores it - but silently re-activating outdated knowledge is
    exactly the kind of fork `dedupe` is supposed to prevent. This check
    reports the match so `remember` can warn without blocking the write.
    """
    if dedupe == "off":
        return None

    c_hash = _content_hash(content)
    table = _mem_table(project)
    cur.execute(
        f"SELECT id, superseded_by FROM {table} "
        f"WHERE content_hash = %s AND superseded_by IS NOT NULL "
        f"AND superseded_by <> %s AND deleted_at IS NULL "
        f"ORDER BY updated_at DESC LIMIT 1",
        (c_hash, SENTINEL_FORGOTTEN),
    )
    row = cur.fetchone()
    if row:
        return {"id": row[0], "superseded_by": row[1], "mode": "exact",
                "similarity": None}

    if embedding_vec is not None and dedupe == "semantic":
        threshold = (fuzzy_threshold if fuzzy_threshold is not None
                     else DEFAULT_DUPLICATE_THRESHOLD)
        max_dist = 1.0 - threshold
        cur.execute(
            f"SELECT id, superseded_by, embedding <=> %s::vector AS dist "
            f"FROM {table} WHERE superseded_by IS NOT NULL "
            f"AND superseded_by <> %s AND deleted_at IS NULL "
            f"AND embedding IS NOT NULL ORDER BY dist ASC LIMIT 1",
            (embedding_vec, SENTINEL_FORGOTTEN),
        )
        row = cur.fetchone()
        if row and float(row[2]) <= max_dist:
            return {"id": row[0], "superseded_by": row[1], "mode": "semantic",
                    "similarity": round(1.0 - float(row[2]), 4)}
    return None


def _find_by_id(cur, memory_id: str, project: str | None) -> str | None:
    """Id of an existing memory, in the target scope or in the global one.

    `supersede=<id>` names a memory directly. Without this check the id is
    accepted blindly and the response reports a replacement that never
    happened.
    """
    if not memory_id:
        return None
    table = _mem_table(project)
    cur.execute(f"SELECT id FROM {table} WHERE id = %s", (memory_id,))
    if cur.fetchone():
        return memory_id
    if project:
        cur.execute("SELECT id FROM memories WHERE id = %s", (memory_id,))
        if cur.fetchone():
            return memory_id
    return None


def _find_by_content(cur, content: str, project: str | None,
                     entity: str | None = None) -> str | None:
    """Id of the active memory with exactly this content (optionally of an
    entity)."""
    table = _mem_table(project)
    c_hash = _content_hash((content or "").strip())
    params: list[Any] = [c_hash]
    sql = (f"SELECT id FROM {table} WHERE content_hash = %s "
           f"AND superseded_by IS NULL AND deleted_at IS NULL")
    if entity:
        sql += " AND entity = %s"
        params.append(entity)
    sql += " ORDER BY updated_at DESC LIMIT 1"
    cur.execute(sql, params)
    row = cur.fetchone()
    return row[0] if row else None


def _find_last_for_entity(cur, entity: str, project: str | None,
                          embedding: list[float] | None = None
                          ) -> tuple[str, str, float | None] | None:
    """The active memory of an entity that best matches the new fact.

    With `embedding` the candidate is the most SIMILAR active memory
    (ties go to the newest), not the newest one: `supersede_last` may only
    replace a fact about the same topic, and an unrelated memory that was
    stored more recently must not block the replacement of an actually
    matching one. Without an embedding the newest memory is the only
    deterministic fallback.
    """
    table = _mem_table(project)
    if embedding is not None:
        cur.execute(
            f"SELECT id, content, embedding <=> %s::vector AS dist "
            f"FROM {table} WHERE entity = %s "
            f"AND superseded_by IS NULL AND deleted_at IS NULL "
            f"ORDER BY dist ASC, updated_at DESC LIMIT 1",
            (embedding, entity),
        )
        row = cur.fetchone()
        if not row:
            return None
        dist = row[2]
        similarity = None if dist is None else round(1.0 - float(dist), 4)
        return row[0], row[1], similarity
    cur.execute(
        f"SELECT id FROM {table} WHERE entity = %s "
        f"AND superseded_by IS NULL AND deleted_at IS NULL "
        f"ORDER BY updated_at DESC LIMIT 1",
        (entity,),
    )
    row = cur.fetchone()
    return (row[0], None, None) if row else None


def _check_contradictions(cur, content: str, entity: str | None,
                          project: str | None,
                          embedding_vec: list[float] | None = None,
                          max_candidates: int = 15) -> list[dict]:
    """NLI-check the new fact against the nearest active memories of the scope.

    With an `entity`, that entity's memories are candidates first; any seats
    left over go to the nearest neighbours from *other* entities, so a conflict
    with a differently tagged memory is caught too (entity tagging should not
    hide it). Without an `entity`, every active memory of the scope competes
    for the candidate slots.

    Checks the project table *and* the global one, so a contradiction between
    a project fact and a global fact is detected (previously only the store
    table was read and cross-scope conflicts were silently missed).

    Every memory type is a candidate: a decision that contradicts a pattern
    is just as harmful as a fact that contradicts a fact. Restricting the
    comparison to `mem_type = 'fact'` hid most real conflicts, because the
    conflicting memory is often a decision, a pattern or a convention.
    """
    candidates: list[tuple[str, str, str | None]] = []
    seen: set[str] = set()
    new_hash = _content_hash(content)
    remaining = max_candidates

    def _fetch(sql_where: str, params: list) -> None:
        nonlocal remaining
        if remaining <= 0:
            return
        for table in _retrieval_tables(cur, project):
            if remaining <= 0:
                break
            if embedding_vec is not None:
                cur.execute(
                    f"SELECT id, content, entity FROM {table} "
                    f"WHERE {sql_where} AND content_hash <> %s "
                    f"AND embedding IS NOT NULL "
                    f"AND superseded_by IS NULL AND deleted_at IS NULL "
                    f"ORDER BY embedding <=> %s::vector ASC LIMIT %s",
                    params + [new_hash, embedding_vec, remaining],
                )
            else:
                cur.execute(
                    f"SELECT id, content, entity FROM {table} "
                    f"WHERE {sql_where} AND content_hash <> %s "
                    f"AND superseded_by IS NULL AND deleted_at IS NULL "
                    f"ORDER BY created_at DESC LIMIT %s",
                    params + [new_hash, remaining],
                )
            for row in cur.fetchall():
                if row[0] in seen:
                    continue
                seen.add(row[0])
                candidates.append((row[0], row[1], row[2]))
                remaining -= 1
            if remaining <= 0:
                break

    # 1. The entity's own memories first - conflicts inside one entity are
    #    the ones an agent notices, so they must get the candidate seats.
    if entity:
        _fetch("entity = %s", [entity])

    # 2. Leftover seats go to the nearest neighbours from other entities
    #    (all entities when no entity was given).
    if entity:
        _fetch("entity IS DISTINCT FROM %s", [entity])
    else:
        _fetch("TRUE", [])

    candidates = candidates[:max_candidates]
    if not candidates:
        return []

    candidate_contents = [c[1] for c in candidates]
    candidate_entities = [c[2] for c in candidates]
    # The entity labels are hints for the NLI model: a differing port of two
    # different services reads very differently when the model knows that the
    # two facts are stored under different entities.
    scores = nli(content, candidate_contents,
                 query_entity=entity, document_entities=candidate_entities)

    contradictions = []
    for (eid, econtent, _eentity), score in zip(candidates, scores):
        if score >= 0.7:
            contradictions.append({"existing_id": eid, "existing_content": econtent,
                                    "similarity": round(score, 3)})
    return contradictions


def _log_access(cur, mem_ids: list[str], project: str | None,
                tables: list[str] | None = None) -> None:
    if not mem_ids:
        return
    # Hits may come from the project table as well as from the global one.
    if tables is None:
        tables = _retrieval_tables(cur, project)
    for table in tables:
        cur.execute(
            f"UPDATE {table} SET access_count = COALESCE(access_count, 0) + 1 "
            f"WHERE id = ANY(%s) AND deleted_at IS NULL",
            (mem_ids,),
        )
