"""Creation and migration of the schema a scope needs.

Everything here is DDL or index maintenance and must stay on the write path
(or startup): a read that calls into this module would create tables as a
side effect. The functions are idempotent, so calling them again is safe.
"""

import hashlib
import logging

from mcp_server import config
from mcp_server.db import drop_vector_indexes, get_conn, ivfflat_lists
from mcp_server.scope_catalog import _norm_project, _project_suffix, _safe_ident
from mcp_server.startup_status import record_warning

logger = logging.getLogger(__name__)


def ensure_fts_language() -> None:
    """Fall back to `english` when the configured dictionary is not installed.

    `FTS_LANGUAGE` is read from the environment at import time; a typo would
    otherwise make every full-text query fail at runtime with an undefined
    text-search configuration. Checking once at startup turns that into a
    warning plus a working fallback. The query side reads
    `config.FTS_LANGUAGE` per call, so the fallback also reaches SQL that was
    built after startup.
    """
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM pg_ts_config WHERE cfgname = %s",
                    (config.FTS_LANGUAGE,))
        exists = cur.fetchone() is not None
    finally:
        conn.close()
    if exists:
        return
    message = (f"FTS_LANGUAGE={config.FTS_LANGUAGE!r} is not an installed "
               f"PostgreSQL text-search dictionary; falling back to "
               f"'english'")
    logger.warning(message)
    record_warning(message)
    config.FTS_LANGUAGE = "english"


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
        if f"to_tsvector('{config.FTS_LANGUAGE}'" not in indexdef:
            cur.execute(f'DROP INDEX IF EXISTS "{name}"')
    cur.execute(
        "SELECT 1 FROM pg_indexes WHERE schemaname = 'public' AND tablename = %s "
        "AND indexdef LIKE '%%to_tsvector%%'",
        (table,),
    )
    if not cur.fetchone():
        cur.execute(
            f"CREATE INDEX {index_name} ON {table} USING GIN "
            f"(to_tsvector('{config.FTS_LANGUAGE}', content))"
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
                     if f"to_tsvector('{config.FTS_LANGUAGE}'" not in d]
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


def migrate_idempotency_keys() -> int:
    """Add `idempotency_key` and its active-row unique index to every scope.

    `ensure_schema`/`ensure_project_table` already do this for the tables they
    create, but a table created by an older build never passes through them
    again until a write touches that scope. The startup pass covers every
    existing table so an idempotent replay works before the first write.

    Returns the number of tables that were changed.
    """
    conn = get_conn()
    changed = 0
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' "
            "AND table_name ~ '^(memories|mem_[A-Za-z0-9_]+)$'"
        )
        for (table,) in cur.fetchall():
            _safe_ident(table, "table name")
            if table == "memories":
                index = "uq_memories_idempotency_key"
            else:
                digest = hashlib.sha1(
                    table[len("mem_"):].encode()).hexdigest()[:10]
                index = f"uq_{digest}_idem"
            cur.execute("SELECT 1 FROM information_schema.columns WHERE "
                        "table_schema = 'public' AND table_name = %s AND "
                        "column_name = 'idempotency_key'", (table,))
            had_column = cur.fetchone() is not None
            cur.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS "
                        "idempotency_key TEXT")
            cur.execute(
                f"CREATE UNIQUE INDEX IF NOT EXISTS {index} "
                f"ON {table}(idempotency_key) "
                "WHERE idempotency_key IS NOT NULL "
                "AND superseded_by IS NULL AND deleted_at IS NULL")
            if not had_column:
                changed += 1
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return changed


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
    cur.execute(f"ALTER TABLE mem_{safe} ADD COLUMN IF NOT EXISTS idempotency_key TEXT")
    cur.execute(
        f"CREATE UNIQUE INDEX IF NOT EXISTS uq_{idx_digest}_idem "
        f"ON mem_{safe}(idempotency_key) "
        "WHERE idempotency_key IS NOT NULL "
        "AND superseded_by IS NULL AND deleted_at IS NULL")
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
