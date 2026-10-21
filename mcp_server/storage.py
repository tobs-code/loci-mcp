import json
import logging

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
                 fuzzy_threshold: float | None = None) -> str | None:
    c_hash = _content_hash(content)
    table = _mem_table(project)
    cur.execute(
        f"SELECT id FROM {table} WHERE content_hash = %s "
        f"AND superseded_by IS NULL AND deleted_at IS NULL LIMIT 1",
        (c_hash,),
    )
    row = cur.fetchone()
    if row:
        return row[0]

    if embedding_vec is not None and fuzzy_threshold is not None and fuzzy_threshold > 0:
        max_dist = 1.0 - fuzzy_threshold
        cur.execute(
            f"SELECT id, embedding <=> %s::vector AS dist "
            f"FROM {table} WHERE superseded_by IS NULL AND deleted_at IS NULL AND embedding IS NOT NULL "
            f"ORDER BY dist ASC LIMIT 1",
            (embedding_vec,),
        )
        row = cur.fetchone()
        if row and float(row[1]) <= max_dist:
            return row[0]

    return None


def _check_contradictions(cur, content: str, entity: str | None,
                          project: str | None,
                          embedding_vec: list[float] | None = None,
                          max_candidates: int = 15) -> list[dict]:
    """NLI-check the new fact against active facts of the same entity.

    Checks the project table *and* the global one, so a contradiction between
    a project fact and a global fact is detected (previously only the store
    table was read and cross-scope conflicts were silently missed).
    """
    if not entity:
        return []

    candidates: list[tuple[str, str]] = []
    seen: set[str] = set()
    for table in _retrieval_tables(cur, project):
        if embedding_vec is not None:
            cur.execute(
                f"SELECT id, content FROM {table} "
                f"WHERE entity = %s AND mem_type = 'fact' AND content != %s "
                f"AND embedding IS NOT NULL "
                f"AND superseded_by IS NULL AND deleted_at IS NULL "
                f"ORDER BY embedding <=> %s::vector ASC LIMIT %s",
                (entity, content, embedding_vec, max_candidates),
            )
        else:
            cur.execute(
                f"SELECT id, content FROM {table} "
                f"WHERE entity = %s AND mem_type = 'fact' AND content != %s "
                f"AND superseded_by IS NULL AND deleted_at IS NULL "
                f"ORDER BY created_at DESC LIMIT %s",
                (entity, content, max_candidates),
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
