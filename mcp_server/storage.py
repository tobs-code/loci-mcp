import json
import logging
from typing import Any

from mcp_server.config import DEFAULT_DUPLICATE_THRESHOLD
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
                 dedupe: str = "exact") -> tuple[str, str, float | None] | None:
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
    cur.execute(
        f"SELECT id FROM {table} WHERE content_hash = %s "
        f"AND superseded_by IS NULL AND deleted_at IS NULL LIMIT 1",
        (c_hash,),
    )
    row = cur.fetchone()
    if row:
        return row[0], "exact", None

    if embedding_vec is not None and dedupe == "semantic":
        threshold = fuzzy_threshold if fuzzy_threshold is not None else DEFAULT_DUPLICATE_THRESHOLD
        max_dist = 1.0 - threshold
        cur.execute(
            f"SELECT id, embedding <=> %s::vector AS dist "
            f"FROM {table} WHERE superseded_by IS NULL AND deleted_at IS NULL AND embedding IS NOT NULL "
            f"ORDER BY dist ASC LIMIT 1",
            (embedding_vec,),
        )
        row = cur.fetchone()
        if row and float(row[1]) <= max_dist:
            return row[0], "semantic", round(1.0 - float(row[1]), 4)

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
    """The most recently updated active memory of an entity.

    With `embedding` the cosine similarity to that memory is returned as
    well. It is used to decide whether a new fact really replaces the old
    one or whether it is about a different topic.
    """
    table = _mem_table(project)
    if embedding is not None:
        cur.execute(
            f"SELECT id, content, embedding <=> %s::vector AS dist "
            f"FROM {table} WHERE entity = %s "
            f"AND superseded_by IS NULL AND deleted_at IS NULL "
            f"ORDER BY updated_at DESC LIMIT 1",
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
    """NLI-check the new fact against active memories of the same entity.

    Checks the project table *and* the global one, so a contradiction between
    a project fact and a global fact is detected (previously only the store
    table was read and cross-scope conflicts were silently missed).

    Every memory type is a candidate: a decision that contradicts a pattern
    is just as harmful as a fact that contradicts a fact. Restricting the
    comparison to `mem_type = 'fact'` hid most real conflicts, because the
    conflicting memory is often a decision, a pattern or a convention.
    """
    if not entity:
        return []

    candidates: list[tuple[str, str]] = []
    seen: set[str] = set()
    new_hash = _content_hash(content)
    for table in _retrieval_tables(cur, project):
        if embedding_vec is not None:
            cur.execute(
                f"SELECT id, content FROM {table} "
                f"WHERE entity = %s AND content_hash <> %s "
                f"AND embedding IS NOT NULL "
                f"AND superseded_by IS NULL AND deleted_at IS NULL "
                f"ORDER BY embedding <=> %s::vector ASC LIMIT %s",
                (entity, new_hash, embedding_vec, max_candidates),
            )
        else:
            cur.execute(
                f"SELECT id, content FROM {table} "
                f"WHERE entity = %s AND content_hash <> %s "
                f"AND superseded_by IS NULL AND deleted_at IS NULL "
                f"ORDER BY created_at DESC LIMIT %s",
                (entity, new_hash, max_candidates),
            )
        for row in cur.fetchall():
            if row[0] in seen:
                continue
            seen.add(row[0])
            candidates.append((row[0], row[1]))
        if len(candidates) >= max_candidates:
            break

    candidates = candidates[:max_candidates]
    if not candidates:
        return []

    candidate_contents = [c[1] for c in candidates]
    scores = nli(content, candidate_contents)

    contradictions = []
    for (eid, econtent), score in zip(candidates, scores):
        if score >= 0.7:
            contradictions.append({"existing_id": eid, "existing_content": econtent,
                                    "similarity": round(score, 3)})
    return contradictions


def _log_access(cur, mem_ids: list[str], project: str | None) -> None:
    if not mem_ids:
        return
    # Hits may come from the project table as well as from the global one.
    tables = _retrieval_tables(cur, project)
    for table in tables:
        cur.execute(
            f"UPDATE {table} SET access_count = COALESCE(access_count, 0) + 1 "
            f"WHERE id = ANY(%s) AND deleted_at IS NULL",
            (mem_ids,),
        )
