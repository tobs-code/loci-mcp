import os
import re
import json
import uuid
import logging
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Any

import psycopg2
import psycopg2.extras

from mcp.server.fastmcp import FastMCP

from mcp_server.db import get_conn, ensure_schema
from mcp_server.inference_client import embed, nli

logger = logging.getLogger(__name__)

app = FastMCP(
    name="loci",
    instructions="""Persistent memory server powered by PostgreSQL + pgvector + Apache AGE.

Use `remember` to store facts, decisions, and observations about entities.
Use `recall` to search semantically, by keyword, or via graph traversal.
Use `relate` to connect entities in the knowledge graph.

=== WHEN TO REMEMBER ===
- User preferences and corrections
- Architectural decisions, conventions, commands
- Facts about people, projects, tools, concepts
- Recurring patterns worth retaining across sessions

=== WHEN TO RECALL ===
- At session start to load relevant context
- When you need information from a previous session
- When the user references something discussed before
- Use `include_decay=true` to prioritize recent / frequently accessed facts
""",
    streamable_http_path="/",
    json_response=True,
)

MEMORY_TYPES = {
    "fact", "decision", "learning", "preference",
    "pattern", "incident", "runbook", "context",
    "observation", "convention", "command", "note",
}
VALID_SCOPES = {"global", "project"}
DECAY_HALF_LIFE_DAYS = 90
DECAY_ACCESS_CAP = 20
# Maximum raw decay score: importance(1.0) * decay(1.0) * access_boost(1.5)
DECAY_RAW_MAX = 1.5
# Share of the final score contributed by decay (0.0 = relevance only).
DECAY_WEIGHT = 0.3
RRF_K = 60
# FTS hits are OR-matched and therefore noisier than vector hits; weight the
# text retriever slightly lower so a single common word cannot outrank the
# best semantic match.
FTS_WEIGHT = 0.6
CACHE_ACCESS_BOOST_DAYS = 7
SENTINEL_FORGOTTEN = '__forgotten__'
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _safe_ident(value: str, kind: str = "identifier") -> str:
    if not _IDENT_RE.match(value or ""):
        raise ValueError(
            f"invalid {kind} {value!r}: only letters, digits and '_' allowed "
            f"(must not start with a digit)"
        )
    return value


def _norm_project(project: str | None) -> str | None:
    """Canonical project name: trimmed, or None for the global scope.

    "global" is the reserved name of the shared scope, so it can never be a
    project of its own (previously `project="global"` created an isolated
    `Project_global` graph that was invisible to global traversal).
    """
    if project is None:
        return None
    name = str(project).strip()
    if not name or name.lower() == "global":
        return None
    return name


def _project_slug(project: str) -> str:
    """Stable, SQL-safe suffix for a project name.

    Plain lower-case identifiers up to 40 chars are used verbatim (backwards
    compatible). Everything else is lower-cased, truncated and suffixed with a
    hash of the *original* name, so case variants ("Team"/"team"), long names
    and names with special characters never share a table or index name.
    """
    project = _norm_project(project)
    if not project:
        raise ValueError("project name must not be empty")
    if _IDENT_RE.match(project) and project == project.lower() and len(project) <= 40:
        return project
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", project).strip("_").lower()
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"p_{cleaned}"
    digest = hashlib.sha1(project.encode()).hexdigest()[:8]
    return f"{cleaned[:40]}_{digest}"


def _project_suffix(project: str) -> str:
    """Stable, SQL-safe, length-bounded table suffix for a project name."""
    return _project_slug(project)


def _scope_key(project: str | None) -> str:
    """Scope value stored on entity rows and graph nodes."""
    return _norm_project(project) or "global"


def _scope_label(project: str | None) -> str:
    """Apache AGE node label for one scope (project or global)."""
    name = _norm_project(project)
    return f"Project_{_project_suffix(name)}" if name else "Entity"


def _table_exists(cur, table: str) -> bool:
    cur.execute(
        "SELECT EXISTS (SELECT FROM information_schema.tables "
        "WHERE table_schema = 'public' AND table_name = %s)",
        (table,),
    )
    return bool(cur.fetchone()[0])


def _ag_str(value) -> str:
    """Unwrap an AGE agtype string ('"foo"' -> 'foo')."""
    text = str(value)
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text[1:-1]
    return text


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _gen_id() -> str:
    return uuid.uuid4().hex[:16]


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


def _decay_score(created_at: datetime, updated_at: datetime | None,
                 access_count: int, importance: float) -> float:
    """Raw recency/frequency score in [0, DECAY_RAW_MAX]."""
    ref = updated_at or created_at
    age_days = (_now() - ref).total_seconds() / 86400
    decay = 0.5 ** (age_days / DECAY_HALF_LIFE_DAYS)
    access_boost = 1.0 + min(access_count, DECAY_ACCESS_CAP) * 1.5 / DECAY_ACCESS_CAP
    return importance * decay * access_boost


def _decay_norm(created_at: datetime, updated_at: datetime | None,
                access_count: int, importance: float) -> float:
    """Normalized decay score in [0, 1], comparable to a normalized RRF score."""
    raw = _decay_score(created_at, updated_at, access_count, importance)
    return max(0.0, min(1.0, raw / DECAY_RAW_MAX))


def _blend_scores(rrf_norm: float, decay_norm: float) -> float:
    """Blend relevance and decay so decay can reorder but never dominate."""
    return (1.0 - DECAY_WEIGHT) * rrf_norm + DECAY_WEIGHT * decay_norm


def _project_tables(cur) -> list[str]:
    """All project-scoped memory tables (mem_<project>)."""
    cur.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'public' AND table_name ~ '^mem_[A-Za-z0-9_]+$' "
        "ORDER BY table_name"
    )
    return [r[0] for r in cur.fetchall()]


def _all_mem_tables(cur) -> list[str]:
    """Global table first, then all project tables."""
    return ["memories"] + _project_tables(cur)


def _retrieval_tables(cur, project: str | None) -> list[str]:
    """Tables consulted by recall: the project table (if any) plus the global one.

    Project hits win over global duplicates, but global facts remain visible
    from inside a project (and vice versa).
    """
    tables: list[str] = []
    project = _norm_project(project)
    if project:
        safe = _project_suffix(project)
        if _table_exists(cur, f"mem_{safe}"):
            tables.append(f"mem_{safe}")
    tables.append("memories")
    return tables


_FTS_TOKEN_RE = re.compile(r"[^\s]+")


def _fts_query(text: str | None) -> str | None:
    """Build an OR-joined websearch query.

    websearch_to_tsquery() ANDs bare terms, so multi-word natural language
    queries almost never matched. Joining the terms with OR keeps the FTS
    retriever useful for any query containing a distinctive word.
    """
    if not text:
        return None
    tokens = []
    for raw in _FTS_TOKEN_RE.findall(text):
        token = re.sub(r'[():"&|!<>*\\]', " ", raw).strip()
        token = re.sub(r"\s+", " ", token)
        if token and token.upper() != "OR":
            tokens.append(token)
    if not tokens:
        return None
    return " OR ".join(tokens)



def _ensure_project_table(cur, project: str) -> None:
    project = _norm_project(project)
    if not project:
        raise ValueError("project name must not be empty")
    safe = _project_suffix(project)
    # Short, hash-based index names: long project names would otherwise
    # truncate `idx_mem_<name>_embedding` and `..._hash` to the same 63-byte
    # identifier and fail with "relation already exists".
    idx_digest = hashlib.sha1(safe.encode()).hexdigest()[:10]
    emb_idx = f"idx_{idx_digest}_emb"
    hash_idx = f"idx_{idx_digest}_hash"
    tsv_idx = f"idx_{idx_digest}_tsv"
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS mem_{safe} (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            content_hash TEXT NOT NULL DEFAULT '',
            embedding vector(768),
            mem_type TEXT NOT NULL DEFAULT 'fact',
            scope TEXT NOT NULL DEFAULT 'project',
            entity TEXT,
            importance FLOAT DEFAULT 1.0,
            metadata JSONB DEFAULT '{{}}',
            access_count INT DEFAULT 0,
            superseded_by TEXT,
            deleted_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    cur.execute(f"ALTER TABLE mem_{safe} ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ")
    # Look indexes up by table + definition: long project names get truncated
    # by PostgreSQL to 63 characters, so the exact index name is unreliable.
    cur.execute(
        "SELECT 1 FROM pg_indexes WHERE schemaname = 'public' AND tablename = %s "
        "AND indexdef LIKE '%%ivfflat%%'",
        (f"mem_{safe}",),
    )
    if not cur.fetchone():
        # Dynamic lists based on expected scaling
        # For 768d: 100 up to ~50k, 200-400 up to ~200k records
        cur.execute(f"SELECT COUNT(*) FROM mem_{safe}")
        count = cur.fetchone()[0] if cur.rowcount > 0 else 0
        
        if count < 50000:
            lists = 100
        elif count < 200000:
            lists = 200
        else:
            lists = 400
        
        cur.execute(f"""
            CREATE INDEX {emb_idx}
            ON mem_{safe} USING ivfflat (embedding vector_cosine_ops) WITH (lists = {lists})
        """)
    cur.execute(
        "SELECT 1 FROM pg_indexes WHERE schemaname = 'public' AND tablename = %s "
        "AND indexdef LIKE '%%content_hash%%'",
        (f"mem_{safe}",),
    )
    if not cur.fetchone():
        cur.execute(f"CREATE INDEX {hash_idx} ON mem_{safe}(content_hash)")
    cur.execute(
        "SELECT 1 FROM pg_indexes WHERE schemaname = 'public' AND tablename = %s "
        "AND indexdef LIKE '%%to_tsvector%%'",
        (f"mem_{safe}",),
    )
    if not cur.fetchone():
        cur.execute(
            f"CREATE INDEX {tsv_idx} "
            f"ON mem_{safe} USING GIN (to_tsvector('german', content))"
        )
    cur.execute(
        "DELETE FROM projects WHERE name = %s AND suffix <> %s",
        (project, safe),
    )
    cur.execute(
        "INSERT INTO projects (suffix, name) VALUES (%s, %s) "
        "ON CONFLICT (suffix) DO UPDATE SET name = EXCLUDED.name",
        (safe, project),
    )


def _project_names(cur) -> dict[str, str]:
    """Map table suffix -> project name from the registry."""
    try:
        cur.execute("SELECT suffix, name FROM projects")
        return {row[0]: row[1] for row in cur.fetchall()}
    except Exception:
        return {}


def _mem_table(project: str | None) -> str:
    project = _norm_project(project)
    if project:
        return f"mem_{_project_suffix(project)}"
    return "memories"


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


def _search_memories_rrf(
    cur,
    query_embedding: list[float] | None,
    query_text: str | None,
    mem_type: str | None,
    entity: str | None,
    limit: int,
    project: str | None,
    include_decay: bool = False,
    include_embedding: bool = False,
    include_superseded: bool = False,
    min_similarity: float | None = None,
    k: int = RRF_K,
    tables: list[str] | None = None,
) -> list[dict]:
    """
    Reciprocal Rank Fusion over vector + FTS results.

    Retrieval runs against the project table (if any) and the global table;
    project rows win when both contain the same content. Scores are normalized
    so that decay can reorder results without drowning out relevance.
    """
    tables = tables if tables is not None else _retrieval_tables(cur, project)
    if not tables:
        return []

    max_distance = None if min_similarity is None else 1.0 - min_similarity
    fts_query = _fts_query(query_text)

    # (score, id, table) tuples, merged across tables before ranking
    vec_candidates: list[tuple[float, str, str]] = []
    text_candidates: list[tuple[float, str, str]] = []

    for table in tables:
        conditions = ["TRUE"]
        params: list[Any] = []
        if include_superseded:
            # Superseded history is retrievable, intentionally deleted rows are not.
            conditions.append("deleted_at IS NULL")
        else:
            conditions.append("superseded_by IS NULL AND deleted_at IS NULL")
        if mem_type:
            conditions.append("mem_type = %s")
            params.append(mem_type)
        if entity:
            conditions.append("entity = %s")
            params.append(entity)
        where = " AND ".join(conditions)

        # 1. Vector Search (Top 50 per table)
        if query_embedding is not None:
            sql = (
                f"SELECT id, embedding <=> %s::vector AS distance FROM {table} "
                f"WHERE {where} AND embedding IS NOT NULL"
            )
            vec_params: list[Any] = [query_embedding] + params
            if max_distance is not None:
                sql += " AND embedding <=> %s::vector <= %s"
                vec_params += [query_embedding, max_distance]
            sql += " ORDER BY distance ASC LIMIT 50"
            cur.execute(sql, vec_params)
            vec_candidates.extend(
                (float(dist), row_id, table) for row_id, dist in cur.fetchall()
            )

        # 2. FTS Search (Top 50 per table), OR-joined terms
        if fts_query:
            sql = (
                f"SELECT id, ts_rank(to_tsvector('german', content), "
                f"websearch_to_tsquery('german', %s)) AS rank FROM {table} "
                f"WHERE to_tsvector('german', content) @@ "
                f"websearch_to_tsquery('german', %s)"
            )
            fts_params: list[Any] = [fts_query, fts_query]
            if max_distance is not None and query_embedding is not None:
                sql += " AND embedding IS NOT NULL AND embedding <=> %s::vector <= %s"
                fts_params += [query_embedding, max_distance]
            sql += f" AND {where} ORDER BY rank DESC LIMIT 50"
            cur.execute(sql, fts_params + params)
            text_candidates.extend(
                (float(rank), row_id, table) for row_id, rank in cur.fetchall()
            )

    vec_candidates.sort(key=lambda c: c[0])
    text_candidates.sort(key=lambda c: -c[0])

    # 3. RRF Fusion
    vector_results: dict[str, float] = {}
    text_results: dict[str, float] = {}
    similarities: dict[str, float] = {}
    table_of: dict[str, str] = {}

    for rank, (distance, row_id, table) in enumerate(vec_candidates[:50], 1):
        vector_results[row_id] = 1.0 / (k + rank)
        similarities[row_id] = 1.0 - distance
        table_of.setdefault(row_id, table)
    for rank, (_, row_id, table) in enumerate(text_candidates[:50], 1):
        text_results[row_id] = FTS_WEIGHT / (k + rank)
        table_of.setdefault(row_id, table)

    fused_scores = {
        row_id: vector_results.get(row_id, 0.0) + text_results.get(row_id, 0.0)
        for row_id in set(vector_results) | set(text_results)
    }
    if not fused_scores:
        return []

    max_rrf = max(fused_scores.values()) or 1.0
    # Do not cut before cross-table dedup: many duplicate contents could
    # otherwise hide unique rows that would fit into the limit.
    top_ids = sorted(fused_scores, key=lambda x: -fused_scores[x])

    # 4. Load details per table (project first), dropping cross-table duplicates
    rows: list[dict] = []
    seen_hashes: set[str] = set()
    for table in tables:
        ids = [row_id for row_id in top_ids if table_of.get(row_id) == table]
        if not ids:
            continue
        cur.execute(f"SELECT * FROM {table} WHERE id = ANY(%s)", (ids,))
        colnames = [desc[0] for desc in cur.description]
        by_id = {r[0]: dict(zip(colnames, r)) for r in cur.fetchall()}
        for row_id in ids:
            d = by_id.get(row_id)
            if d is None:
                continue
            c_hash = d.get("content_hash") or ""
            if c_hash and c_hash in seen_hashes:
                continue
            if c_hash:
                seen_hashes.add(c_hash)
            d["_rrf_score"] = fused_scores.get(row_id, 0.0)
            d["_rrf_norm"] = d["_rrf_score"] / max_rrf
            if row_id in similarities:
                d["_similarity"] = round(similarities[row_id], 4)
                d["_distance"] = round(1.0 - similarities[row_id], 4)
            if not include_embedding:
                d.pop("embedding", None)
            if include_decay:
                d["_decay_score"] = _decay_score(
                    d.get("created_at"), d.get("updated_at"),
                    d.get("access_count", 0), d.get("importance", 1.0),
                )
                d["_decay_norm"] = _decay_norm(
                    d.get("created_at"), d.get("updated_at"),
                    d.get("access_count", 0), d.get("importance", 1.0),
                )
            rows.append(d)

    # 5. Re-rank
    if include_decay:
        for r in rows:
            r["_final_score"] = _blend_scores(r["_rrf_norm"], r["_decay_norm"])
        rows.sort(key=lambda x: -x.get("_final_score", 0))
    else:
        rows.sort(key=lambda x: -x.get("_rrf_score", 0))

    return rows[:limit]


def _search_memories(
    cur,
    query_embedding: list[float] | None,
    query_text: str | None,
    mem_type: str | None,
    entity: str | None,
    limit: int,
    project: str | None,
    include_decay: bool = False,
    include_embedding: bool = False,
    include_superseded: bool = False,
    min_similarity: float | None = None,
    hybrid: bool = True,  # Default: RRF aktivieren
    tables: list[str] | None = None,
) -> list[dict]:
    """
    Wrapper for search functions. Uses RRF when hybrid=True, otherwise legacy search.

    `tables` overrides the table set (default: project table + global).
    """
    if hybrid and (query_embedding is not None or query_text):
        return _search_memories_rrf(
            cur, query_embedding, query_text, mem_type, entity, limit, project,
            include_decay, include_embedding, include_superseded, min_similarity,
            tables=tables,
        )

    # Legacy fallback for individual retrievers
    tables = tables if tables is not None else _retrieval_tables(cur, project)
    if not tables:
        return []

    max_distance = None if min_similarity is None else 1.0 - min_similarity
    has_vec = query_embedding is not None
    fts_query = _fts_query(query_text)

    rows: list[dict] = []
    seen_hashes: set[str] = set()
    for table in tables:
        conditions: list[str] = ["TRUE"]
        params: list[Any] = []
        params_vec: list[Any] = []

        if has_vec:
            col = "*, embedding <=> %s::vector AS distance"
            conditions.append("embedding IS NOT NULL")
            order = "ORDER BY distance ASC"
            params_vec.append(query_embedding)
            if max_distance is not None:
                conditions.append("embedding <=> %s::vector <= %s")
                params_vec.extend([query_embedding, max_distance])
        else:
            col = "*"
            order = "ORDER BY updated_at DESC"

        if fts_query:
            conditions.append(
                "to_tsvector('german', content) @@ websearch_to_tsquery('german', %s)"
            )
            params.append(fts_query)
        if mem_type:
            conditions.append("mem_type = %s")
            params.append(mem_type)
        if entity:
            conditions.append("entity = %s")
            params.append(entity)

        if include_superseded:
            conditions.append("deleted_at IS NULL")
        else:
            conditions.append("superseded_by IS NULL AND deleted_at IS NULL")

        where = " AND ".join(conditions)
        cur.execute(
            f"SELECT {col}, created_at, updated_at, access_count, importance "
            f"FROM {table} WHERE {where} {order} LIMIT %s",
            params_vec + params + [limit],
        )

        colnames = [desc[0] for desc in cur.description] if cur.description else []
        for row in cur.fetchall():
            d = dict(zip(colnames, row))
            if "distance" in d and d["distance"] is not None:
                d["_distance"] = float(d.pop("distance"))
                d["_similarity"] = round(1.0 - d["_distance"], 4)
            else:
                d.pop("distance", None)
            c_hash = d.get("content_hash") or ""
            if c_hash and c_hash in seen_hashes:
                continue
            if c_hash:
                seen_hashes.add(c_hash)
            if not include_embedding:
                d.pop("embedding", None)
            if include_decay:
                d["_decay_score"] = _decay_score(
                    d.get("created_at"), d.get("updated_at"),
                    d.get("access_count", 0), d.get("importance", 1.0),
                )
                d["_decay_norm"] = _decay_norm(
                    d.get("created_at"), d.get("updated_at"),
                    d.get("access_count", 0), d.get("importance", 1.0),
                )
            rows.append(d)

    if has_vec:
        rows.sort(key=lambda x: x.get("_distance", 1.0))
    else:
        rows.sort(key=lambda x: x.get("updated_at") or _now(), reverse=True)

    if include_decay:
        for r in rows:
            if has_vec and "_similarity" in r:
                r["_final_score"] = _blend_scores(r["_similarity"], r["_decay_norm"])
            else:
                # Nothing relevant to blend with: recency/frequency only.
                r["_final_score"] = r["_decay_norm"]
        rows.sort(key=lambda x: -x.get("_final_score", 0))

    return rows[:limit]


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


# -------------------------------------------------------------------
# TOOLS — WRITE
# -------------------------------------------------------------------


@app.tool()
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


@app.tool()
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


@app.tool()
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


@app.tool()
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


# -------------------------------------------------------------------
# TOOLS — READ / SEARCH
# -------------------------------------------------------------------


@app.tool()
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


def _search_memories_graph_boost(
    existing_results: list[dict],
    entity: str,
    limit: int,
    project: str | None,
    include_decay: bool = False,
    graph_boost_weight: float = 0.3,  # 30% boost for graph candidates
) -> list[dict]:
    """
    Augments search results with memories from connected entities via the graph.
    Merges graph candidates with RRF scores.
    """
    tables = None
    name_json = json.dumps(entity)
    graph_entities: list[tuple[str, float]] = []
    graph_results: list[dict] = []

    # Dedicated connection: a failing Cypher call must not abort
    # the running recall transaction.
    conn = get_conn()
    try:
        cur = conn.cursor()
        tables = _retrieval_tables(cur, project)

        # 1. Find connected entities via Apache AGE (any edge direction/type)
        cur.execute("SET search_path TO ag_catalog, public")
        label = _scope_label(project)
        cur.execute(
            f"SELECT * FROM cypher('memory_graph', $$ "
            f"MATCH (a:{label} {{name: {name_json}}})-[r]-(b) "
            f"RETURN b.name, coalesce(r.weight, 1.0) LIMIT 10 $$) AS "
            f"(name agtype, weight agtype)"
        )
        graph_entities = [
            (_ag_str(name), float(str(weight)))
            for name, weight in cur.fetchall()
        ]

        # 2. Fetch memories of connected entities
        existing_ids = {r["id"] for r in existing_results}
        for entity_name, weight in graph_entities:
            for table in tables:
                cur.execute(
                    f"SELECT * FROM {table} "
                    f"WHERE entity = %s AND superseded_by IS NULL AND deleted_at IS NULL "
                    f"LIMIT 5",
                    (entity_name,)
                )
                for row in cur.fetchall():
                    d = dict(zip([desc[0] for desc in cur.description], row))
                    d.pop("embedding", None)
                    if d["id"] in existing_ids:
                        continue
                    d["_graph_weight"] = weight
                    d["_graph_source"] = entity_name
                    if include_decay:
                        d["_decay_score"] = _decay_score(
                            d.get("created_at"), d.get("updated_at"),
                            d.get("access_count", 0), d.get("importance", 1.0),
                        )
                        d["_decay_norm"] = _decay_norm(
                            d.get("created_at"), d.get("updated_at"),
                            d.get("access_count", 0), d.get("importance", 1.0),
                        )
                    existing_ids.add(d["id"])
                    graph_results.append(d)

        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.warning("Graph boost failed: %s", e)
        return existing_results
    finally:
        conn.close()

    if not graph_results:
        return existing_results

    # 3. Fuse with existing results
    # Keep the boost relative to the best RRF score so graph candidates
    # cannot jump to the top on weight alone.
    max_rrf = max((r.get("_rrf_score", 0.0) for r in existing_results), default=0.0)
    if max_rrf <= 0:
        max_rrf = 1.0 / 61.0

    all_results = existing_results + graph_results
    for r in all_results:
        rrf_norm = r.get("_rrf_norm")
        if rrf_norm is None:
            rrf_norm = r.get("_rrf_score", 0.0) / max_rrf
            r["_rrf_norm"] = rrf_norm
        if "_graph_weight" in r:
            r["_graph_boost"] = graph_boost_weight * r["_graph_weight"]
        graph_boost = r.get("_graph_boost", 0.0)
        decay_norm = r.get("_decay_norm", 0.0) if include_decay else 0.0

        # Final score: relevance (RRF) + graph boost + at most 30% decay
        r["_final_score"] = _blend_scores(rrf_norm, decay_norm) + graph_boost

    # Sort by final score and limit
    all_results.sort(key=lambda x: -x.get("_final_score", 0))
    return all_results[:limit * 2]  # 2x for filter overhead


@app.tool()
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


@app.tool()
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


@app.tool()
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


# -------------------------------------------------------------------
# TOOLS — GRAPH (Apache AGE)
# -------------------------------------------------------------------


def _ensure_graph_entity(cur, name: str, project: str | None,
                         ent_type: str | None = None) -> None:
    """Create the AGE node and SQL entity row for one scope (idempotent)."""
    label = _scope_label(project)
    scope = _scope_key(project)
    ent_json = json.dumps(name)
    scope_json = json.dumps(scope)
    cur.execute(
        f"SELECT * FROM cypher('memory_graph', $$ "
        f"MATCH (n:{label} {{name: {ent_json}}}) RETURN n $$) AS (n agtype)"
    )
    if not cur.fetchone():
        cur.execute(
            f"SELECT * FROM cypher('memory_graph', $$ "
            f"CREATE (n:{label} {{name: {ent_json}, project: {scope_json}}}) RETURN n $$) AS (n agtype)"
        )
    cur.execute(
        "INSERT INTO entities (name, type, project) VALUES (%s, %s, %s) "
        "ON CONFLICT (name, project) DO UPDATE SET type = EXCLUDED.type",
        (name, ent_type or label, scope),
    )


def _relate_edges(cur, edges: list[dict], project: str | None) -> list[dict]:
    """Create/update directed AGE edges and the SQL mirror rows.

    Caller must set search_path to ag_catalog, public and handle the
    transaction. Uses MERGE, so it is idempotent and keeps both directions
    of the same relation type independent.
    """
    label = _scope_label(project)
    prepared: list[dict] = []
    for edge in edges:
        frm = edge.get("from_entity")
        to = edge.get("to_entity")
        rtype = edge.get("relation_type")
        if not frm or not to:
            raise ValueError("each edge needs 'from_entity' and 'to_entity'")
        _safe_ident(rtype or "", "relation_type")
        prepared.append({
            "from_entity": frm,
            "to_entity": to,
            "relation_type": rtype,
            "weight": float(edge.get("weight", 1.0)),
            "context": edge.get("context"),
        })

    entities: list[str] = []
    seen: set[str] = set()
    for e in prepared:
        for ent in (e["from_entity"], e["to_entity"]):
            if ent not in seen:
                seen.add(ent)
                entities.append(ent)

    scope = _scope_key(project)
    for ent in entities:
        _ensure_graph_entity(cur, ent, project)

    results: list[dict] = []
    for e in prepared:
        from_json = json.dumps(e["from_entity"])
        to_json = json.dumps(e["to_entity"])
        w_json = json.dumps(e["weight"])
        rtype = e["relation_type"]

        cur.execute(
            f"SELECT * FROM cypher('memory_graph', $$ "
            f"MATCH (a:{label} {{name: {from_json}}})-[r:{rtype}]->(b:{label} {{name: {to_json}}}) "
            f"RETURN r.weight $$) AS (w agtype)"
        )
        existed = cur.fetchone() is not None

        cur.execute(
            f"SELECT * FROM cypher('memory_graph', $$ "
            f"MATCH (a:{label} {{name: {from_json}}}), (b:{label} {{name: {to_json}}}) "
            f"MERGE (a)-[r:{rtype}]->(b) SET r.weight = {w_json} "
            f"RETURN r $$) AS (r agtype)"
        )
        cur.execute(
            "INSERT INTO relations (source_id, target_id, relation_type, weight) "
            "SELECT e1.id, e2.id, %s, %s "
            "FROM entities e1, entities e2 "
            "WHERE e1.name = %s AND e2.name = %s "
            "AND e1.project = %s AND e2.project = %s "
            "ON CONFLICT (source_id, target_id, relation_type) DO UPDATE SET weight = EXCLUDED.weight",
            (rtype, e["weight"], e["from_entity"], e["to_entity"], scope, scope),
        )
        results.append({
            "from_entity": e["from_entity"],
            "to_entity": e["to_entity"],
            "relation_type": rtype,
            "weight": e["weight"],
            "status": "updated" if existed else "created",
            "context_memory": None,
        })

    context_edges = [(e, r) for e, r in zip(prepared, results) if e["context"]]
    if context_edges:
        embeddings = embed([e["context"] for e, _ in context_edges])
        ctx_table = _mem_table(project)
        for (e, r), emb in zip(context_edges, embeddings):
            ctx_entity = f"{e['from_entity']}_{e['relation_type']}_{e['to_entity']}"
            cur.execute(
                f"SELECT id FROM {ctx_table} WHERE entity = %s "
                f"AND superseded_by IS NULL AND deleted_at IS NULL LIMIT 1",
                (ctx_entity,),
            )
            row = cur.fetchone()
            mem_id = _store_memory(
                cur, e["context"], emb, "fact", ctx_entity, 0.8,
                {"relation_type": e["relation_type"], "from": e["from_entity"],
                 "to": e["to_entity"], "weight": e["weight"]},
                project, None,
            )
            if row:
                _supersede(cur, project, mem_id, row[0])
            r["context_memory"] = mem_id

    return results


@app.tool()
def relate(
    from_entity: str,
    to_entity: str,
    relation_type: str,
    context: str | None = None,
    project: str | None = None,
    weight: float = 1.0,
) -> str:
    """Create a directed relation between two entities in the AGE graph."""
    logger.info("relate %s ->[%s]-> %s weight=%s", from_entity, relation_type, to_entity, weight)
    project = _norm_project(project)
    if not relation_type:
        return "ERROR: relation_type is required"

    conn = get_conn()
    mem_id = None
    try:
        cur = conn.cursor()
        if context and project:
            _ensure_project_table(cur, project)
        cur.execute("SET search_path TO ag_catalog, public")
        results = _relate_edges(cur, [{
            "from_entity": from_entity,
            "to_entity": to_entity,
            "relation_type": relation_type,
            "weight": weight,
            "context": context,
        }], project)
        conn.commit()
    except Exception as e:
        conn.rollback()
        return f"ERROR: {e}"
    finally:
        conn.close()

    edge = results[0]
    action = "Updated" if edge["status"] == "updated" else "Created"
    mem_id = edge["context_memory"]
    parts = [f"{action} {relation_type} from {from_entity} -> {to_entity}" + (f" (w={weight})" if weight != 1.0 else "")]
    if mem_id:
        parts.append(f"context: {mem_id}")
    return " | ".join(parts)


@app.tool()
def relate_batch(
    edges: list[dict],
    project: str | None = None,
) -> list[dict]:
    """Create or update multiple directed relations in one transaction.

    Each edge needs: from_entity, to_entity, relation_type.
    Optional: weight (default 1.0), context (stored as a memory), project
    (overrides the tool-level project for that edge).
    Entities are auto-created; existing edges are updated (MERGE), so the
    call is idempotent and both directions of a relation can coexist.
    """
    logger.info("relate_batch count=%s project=%s", len(edges or []), project)
    project = _norm_project(project)
    if not edges:
        return []
    conn = get_conn()
    try:
        cur = conn.cursor()
        if any(e.get("context") for e in edges):
            for p in {_norm_project(e.get("project", project))
                      for e in edges if e.get("context")}:
                if p:
                    _ensure_project_table(cur, p)
        cur.execute("SET search_path TO ag_catalog, public")
        # Edges may target different scopes; keep the caller's order.
        indexed: dict[Any, list[tuple[int, dict]]] = {}
        for i, edge in enumerate(edges):
            indexed.setdefault(_norm_project(edge.get("project", project)),
                               []).append((i, edge))
        results: list[Any] = [None] * len(edges)
        for scope, pairs in indexed.items():
            batch = _relate_edges(cur, [edge for _, edge in pairs], scope)
            for (pos, _), result in zip(pairs, batch):
                results[pos] = result
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.exception("relate_batch failed")
        return [{"error": str(e)}]
    finally:
        conn.close()
    return results



@app.tool()
def find_connections(
    entity_name: str,
    direction: str = "both",
    relation_type: str | None = None,
    limit: int = 20,
    project: str | None = None,
) -> list[dict]:
    """Traverse the graph from an entity to find connected entities.

    `direction` is "outgoing", "incoming" or "both"; each result reports its
    own direction. `project` selects that project's graph; without it the
    global graph is traversed.
    """
    logger.info("find_connections entity=%s direction=%s relation_type=%s limit=%s project=%s",
                entity_name, direction, relation_type, limit, project)
    project = _norm_project(project)
    if direction not in ("outgoing", "incoming", "both"):
        return [{"error": "direction must be 'outgoing', 'incoming' or 'both'"}]
    if relation_type:
        _safe_ident(relation_type, "relation_type")
    label = _scope_label(project)
    rtype_cypher = f":{relation_type}" if relation_type else ""
    name_json = json.dumps(entity_name)
    query = f"""
        MATCH (a:{label})-[r{rtype_cypher}]-(b)
        WHERE a.name = {name_json}
        RETURN b.name AS connected_entity, type(r) AS relation_type,
               coalesce(r.weight, 1.0) AS w, startNode(r).name AS src,
               endNode(r).name AS tgt
        LIMIT {int(limit) * 4}
    """

    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("SET search_path TO ag_catalog, public")
        cur.execute(
            f"SELECT * FROM cypher('memory_graph', $$ {query} $$) AS "
            f"(connected_entity agtype, relation_type agtype, weight agtype, "
            f"src agtype, tgt agtype)",
        )
        results = []
        for row in cur.fetchall():
            src = _ag_str(row[3])
            tgt = _ag_str(row[4])
            edge_dir = "outgoing" if src == entity_name else "incoming"
            if direction != "both" and edge_dir != direction:
                continue
            results.append({
                "entity": _ag_str(row[0]),
                "relation": _ag_str(row[1]),
                "weight": float(str(row[2])),
                "direction": edge_dir,
                "from_entity": src,
                "to_entity": tgt,
            })
        # Strongest edges first, then alphabetical for stable output.
        results.sort(key=lambda r: (-r["weight"], r["entity"], r["relation"]))
        results = results[:limit]
        conn.commit()
    except Exception as e:
        conn.rollback()
        return [{"error": str(e)}]
    finally:
        conn.close()
    return results


@app.tool()
def find_paths(from_entity: str, to_entity: str, max_hops: int = 5,
               project: str | None = None) -> dict:
    """Find paths between two entities in the graph.

    `project` selects that project's graph; without it the global graph is used.
    """
    logger.info("find_paths from=%s to=%s max_hops=%s project=%s",
                from_entity, to_entity, max_hops, project)
    project = _norm_project(project)
    max_hops = max(1, min(int(max_hops), 10))
    label = _scope_label(project)
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("SET search_path TO ag_catalog, public")
        fe_json = json.dumps(from_entity)
        te_json = json.dumps(to_entity)
        cur.execute(
            f"""SELECT * FROM cypher('memory_graph', $$
                MATCH p = (a:{label})-[*1..{max_hops}]-(b:{label})
                WHERE a.name = {fe_json} AND b.name = {te_json}
                RETURN nodes(p) AS ns, relationships(p) AS rs
                LIMIT 20
            $$) AS (ns agtype, rs agtype)""")
        rows = cur.fetchall()
        conn.commit()
    except Exception as e:
        conn.rollback()
        return {"paths": [], "error": str(e)}
    finally:
        conn.close()
    paths = []
    seen_paths: set[tuple] = set()
    for r in rows:
        nodes_str = str(r[0]).replace("::vertex", "").replace("::edge", "")
        rels_str = str(r[1]).replace("::vertex", "").replace("::edge", "")
        try:
            parsed = json.loads(nodes_str)
            parsed_rels = json.loads(rels_str)
        except json.JSONDecodeError:
            continue
        names = [n.get("properties", {}).get("name", "?") for n in parsed]
        ids = [n.get("id") for n in parsed]
        if len(names) != len(set(names)):
            continue
        signature = tuple(ids)
        if signature in seen_paths:
            continue
        seen_paths.add(signature)
        nodes = [{"id": ids[i], "name": names[i],
                  "label": parsed[i].get("label")}
                 for i in range(len(names))]
        relations = []
        for rel in parsed_rels:
            props = rel.get("properties", {})
            start = rel.get("start_id", rel.get("start"))
            end = rel.get("end_id", rel.get("end"))
            relations.append({
                "type": rel.get("label"),
                "from": names[ids.index(start)] if start in ids else start,
                "to": names[ids.index(end)] if end in ids else end,
                "weight": props.get("weight", 1.0),
            })
        paths.append({"nodes": nodes, "relations": relations,
                      "hops": len(nodes) - 1})
    return {"paths": paths, "count": len(paths)}


# -------------------------------------------------------------------
# TOOLS — MAINTENANCE
# -------------------------------------------------------------------


@app.tool()
def contradictions_report(entity: str | None = None,
                         project: str | None = None,
                         limit: int = 20) -> list[dict]:
    """Find potentially conflicting facts stored for the same entity.

    Uses vector similarity to find candidate pairs, then NLI to confirm.
    Returns pairs with confidence scores in [0, 1] (higher = more likely contradictory).
    """
    logger.info("contradictions_report entity=%s limit=%s", entity, limit)
    project = _norm_project(project)
    if not entity:
        return [{"error": "entity parameter is required. contradictions_report works on one entity at a time."}]
    conn = get_conn()
    try:
        cur = conn.cursor()
        # Only active facts are seeds: already superseded pairs are resolved
        # and must not be reported again.
        facts = _search_memories(cur, None, None, "fact", entity, 200, project,
                                  include_embedding=True, include_superseded=False)
        if len(facts) < 2:
            conn.commit()
            return []

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
        return [{"error": "query failed"}]
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

    return contradictions[:limit]


@app.tool()
def server_health() -> dict:
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
    return {"postgres": pg_ok, "inference": _health_check()}


# -------------------------------------------------------------------
# TOOLS — EXPORT / IMPORT
# -------------------------------------------------------------------


@app.tool()
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


@app.tool()
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


# -------------------------------------------------------------------
# RESOURCES
# -------------------------------------------------------------------


@app.resource("memory://stats")
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
        "projects": [names.get(t[len("mem_"):], t[len("mem_"):])
                     for t in tables if t != "memories"],
        "entities": entities,
        "relations": relations,
        "inference_healthy": _health_check(),
    }, indent=2)


# -------------------------------------------------------------------
# INTERNAL
# -------------------------------------------------------------------


def _health_check() -> bool:
    try:
        from mcp_server.inference_client import health as h
        return h()
    except Exception:
        return False


def _count_entities() -> int:
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM entities")
        return cur.fetchone()[0]
    except Exception:
        return 0
    finally:
        conn.close()


# -------------------------------------------------------------------
# ENTRYPOINT
# -------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    host = os.environ.get("MCP_HOST", "0.0.0.0")
    port = int(os.environ.get("MCP_PORT", "8766"))

    try:
        ensure_schema()
        logger.info("Schema OK")
    except Exception as e:
        logger.warning("Schema may already exist: %s", e)

    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("UPDATE memories SET superseded_by = %s WHERE superseded_by = 'forgotten'",
                    (SENTINEL_FORGOTTEN,))
        cur.execute("SELECT table_name FROM information_schema.tables WHERE table_name LIKE 'mem_%'")
        for t in cur.fetchall():
            cur.execute(f"UPDATE {t[0]} SET superseded_by = %s WHERE superseded_by = 'forgotten'",
                        (SENTINEL_FORGOTTEN,))
        # Add deleted_at column if missing (migration)
        cur.execute("ALTER TABLE memories ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ")
        cur.execute("SELECT table_name FROM information_schema.tables WHERE table_name LIKE 'mem_%'")
        for t in cur.fetchall():
            cur.execute(f"ALTER TABLE {t[0]} ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ")
        # Migrate existing __forgotten__ sentinel to deleted_at
        cur.execute("UPDATE memories SET deleted_at = updated_at WHERE superseded_by = %s AND deleted_at IS NULL",
                    (SENTINEL_FORGOTTEN,))
        cur.execute("SELECT table_name FROM information_schema.tables WHERE table_name LIKE 'mem_%'")
        for t in cur.fetchall():
            cur.execute(f"UPDATE {t[0]} SET deleted_at = updated_at WHERE superseded_by = %s AND deleted_at IS NULL",
                        (SENTINEL_FORGOTTEN,))
        conn.commit()
        conn.close()
        logger.info("deleted_at column migrated")
    except Exception as e:
        logger.warning("Schema migration skipped: %s", e)

    # Deduplicate AGE edges (keep only the first edge per directed pair+type)
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SET search_path TO ag_catalog, public")
        cur.execute("""
            SELECT * FROM cypher('memory_graph', $$
                MATCH (a)-[r1]->(b)
                WITH a, b, type(r1) AS t, collect(r1) AS rels
                WHERE size(rels) > 1
                RETURN a.name, b.name, t, size(rels)
            $$) AS (src agtype, tgt agtype, rel_type agtype, cnt agtype)
        """)
        dup_rows = cur.fetchall()
        for src, tgt, rel_type, cnt in dup_rows:
            for _ in range(int(str(cnt)) - 1):
                cur.execute(
                    f"SELECT * FROM cypher('memory_graph', $$ "
                    f"MATCH (a {{name: {json.dumps(str(src).strip('\"'))}}})"
                    f"-[r:{str(rel_type).strip('\"')}]->"
                    f"(b {{name: {json.dumps(str(tgt).strip('\"'))}}}) "
                    f"DELETE r RETURN 1 LIMIT 1 $$) AS (x agtype)"
                )
        conn.commit()
        conn.close()
        if dup_rows:
            logger.info("Deduplicated %s duplicate AGE edge groups", len(dup_rows))
    except Exception as e:
        logger.warning("AGE edge dedup skipped: %s", e)

    if transport == "sse":
        import uvicorn
        asgi_app = app.sse_app()
        logger.info("Starting SSE MCP server on %s:%s", host, port)
        uvicorn.run(asgi_app, host=host, port=port, log_level="info")
    elif transport == "streamable-http":
        import uvicorn
        asgi_app = app.streamable_http_app()
        logger.info("Starting Streamable HTTP MCP server on %s:%s", host, port)
        uvicorn.run(asgi_app, host=host, port=port, log_level="info")
    else:
        logger.info("Starting MCP server (transport=stdio)")
        app.run(transport="stdio")