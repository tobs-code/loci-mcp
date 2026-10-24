"""Creation and migration of the schema a scope needs.

Everything here is DDL or index maintenance and must stay on the write path
(or startup): a read that calls into this module would create tables as a
side effect. The functions are idempotent, so calling them again is safe.
"""

import hashlib

from mcp_server.config import FTS_LANGUAGE
from mcp_server.db import drop_vector_indexes, get_conn, ivfflat_lists
from mcp_server.scope_catalog import _norm_project, _project_suffix, _safe_ident


def ensure_fts_index(cur, table: str, index_name: str) -> None:
    """Make sure the full-text index of `table` uses the configured dictionary.

    An index built with a different dictionary cannot be used by the query
    side, so it is dropped and rebuilt.
    """
    _safe_ident(table, "table name")
    cur.execute(
        "SELECT indexname, indexdef FROM pg_indexes WHERE schemaname = 'public' "
        "AND tablename = %s AND indexdef LIKE '%%to_tsvector%%'",
        (table,),
    )
    for name, indexdef in cur.fetchall():
        if f"to_tsvector('{FTS_LANGUAGE}'" not in indexdef:
            cur.execute(f'DROP INDEX IF EXISTS "{name}"')
    cur.execute(
        "SELECT 1 FROM pg_indexes WHERE schemaname = 'public' AND tablename = %s "
        "AND indexdef LIKE '%%to_tsvector%%'",
        (table,),
    )
    if not cur.fetchone():
        cur.execute(
            f"CREATE INDEX {index_name} ON {table} USING GIN "
            f"(to_tsvector('{FTS_LANGUAGE}', content))"
        )


def migrate_fts_indexes() -> int:
    """Rebuild full-text indexes that use a different dictionary.

    Returns the number of rebuilt indexes.
    """
    conn = get_conn()
    rebuilt = 0
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_name ~ '^mem_[A-Za-z0-9_]+$'"
        )
        for (table,) in cur.fetchall():
            cur.execute(
                "SELECT indexdef FROM pg_indexes WHERE schemaname = 'public' "
                "AND tablename = %s AND indexdef LIKE '%%to_tsvector%%'",
                (table,),
            )
            stale = [d for (d,) in cur.fetchall()
                     if f"to_tsvector('{FTS_LANGUAGE}'" not in d]
            if not stale:
                continue
            safe = table[len("mem_"):]
            idx_digest = hashlib.sha1(safe.encode()).hexdigest()[:10]
            ensure_fts_index(cur, table, f"idx_{idx_digest}_tsv")
            rebuilt += 1
        conn.commit()
    finally:
        conn.close()
    return rebuilt


def ensure_project_table(cur, project: str) -> None:
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
            expires_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    cur.execute(f"ALTER TABLE mem_{safe} ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ")
    cur.execute(f"ALTER TABLE mem_{safe} ADD COLUMN IF NOT EXISTS expires_at TIMESTAMPTZ")
    # Look indexes up by table + definition: long project names get truncated
    # by PostgreSQL to 63 characters, so the exact index name is unreliable.
    cur.execute(f"SELECT COUNT(*) FROM mem_{safe}")
    count = cur.fetchone()[0]
    lists = ivfflat_lists(count)
    if lists is None:
        # A table this small is faster and, above all, exact without an ANN
        # index: IVFFlat with default probes misses rows in unprobed lists.
        drop_vector_indexes(cur, f"mem_{safe}")
    else:
        cur.execute(
            "SELECT 1 FROM pg_indexes WHERE schemaname = 'public' AND tablename = %s "
            "AND indexdef LIKE '%%ivfflat%%'",
            (f"mem_{safe}",),
        )
        if not cur.fetchone():
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
    ensure_fts_index(cur, f"mem_{safe}", tsv_idx)
    cur.execute(
        "DELETE FROM projects WHERE name = %s AND suffix <> %s",
        (project, safe),
    )
    cur.execute(
        "INSERT INTO projects (suffix, name) VALUES (%s, %s) "
        "ON CONFLICT (suffix) DO UPDATE SET name = EXCLUDED.name",
        (safe, project),
    )
