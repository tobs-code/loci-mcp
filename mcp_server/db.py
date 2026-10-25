import logging
import os

import psycopg2
import psycopg2.extras
from contextlib import contextmanager

from mcp_server import config
from mcp_server.startup_status import record_warning
logger = logging.getLogger(__name__)

# No password in the default: the credential belongs in MEMORY_DATABASE_URL.
# Docker Compose supplies a DSN built from POSTGRES_USER/POSTGRES_PASSWORD.
DATABASE_URL = os.environ.get(
    "MEMORY_DATABASE_URL",
    "postgresql://memory@localhost:5432/memory",
)

# An IVFFlat index probes only `ivfflat.probes` lists (default 1). With 100
# lists on a table that holds a handful of rows, that routinely probes an
# empty list, so nearby rows are never returned (missed near-duplicates,
# empty contradiction scans). Small tables are therefore scanned exactly;
# the ANN index is only built once it can actually help.
IVFFLAT_MIN_ROWS = 10_000


def ivfflat_lists(row_count: int) -> int | None:
    """Number of IVFFlat lists for a table, or None to stay exact."""
    if row_count < IVFFLAT_MIN_ROWS:
        return None
    return min(400, max(10, row_count // 1000))


def drop_vector_indexes(cur, table: str) -> None:
    """Drop any approximate vector index of `table` (exact scan remains)."""
    cur.execute(
        "SELECT indexname FROM pg_indexes "
        "WHERE schemaname = 'public' AND tablename = %s "
        "AND indexdef LIKE '%%USING ivfflat%%'",
        (table,),
    )
    for (name,) in cur.fetchall():
        cur.execute(f'DROP INDEX IF EXISTS "{name}"')


def get_conn():
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = False
    return conn


@contextmanager
def tx():
    conn = get_conn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def ensure_schema():
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute("CREATE EXTENSION IF NOT EXISTS age")
        cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
        cur.execute("SET search_path TO ag_catalog, public")
        try:
            cur.execute("SELECT create_graph('memory_graph')")
        except Exception:
            conn.rollback()
            cur = conn.cursor()
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS ag_catalog.memories CASCADE")
        cur.execute("DROP TABLE IF EXISTS ag_catalog.entities CASCADE")
        cur.execute("DROP TABLE IF EXISTS ag_catalog.relations CASCADE")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                content_hash TEXT NOT NULL DEFAULT '',
                embedding vector(768),
                mem_type TEXT NOT NULL DEFAULT 'fact',
                scope TEXT NOT NULL DEFAULT 'global',
                entity TEXT,
                importance FLOAT DEFAULT 1.0,
                metadata JSONB DEFAULT '{}',
                access_count INT DEFAULT 0,
                superseded_by TEXT,
                deleted_at TIMESTAMPTZ,
                expires_at TIMESTAMPTZ,
                valid_from TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                valid_to TIMESTAMPTZ,
                source TEXT,
                agent_id TEXT,
                confidence FLOAT CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1)),
                tags TEXT[] NOT NULL DEFAULT '{}',
                pinned BOOLEAN NOT NULL DEFAULT FALSE,
                hard_deleted_at TIMESTAMPTZ,
                idempotency_key TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                type TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        # Entities are scoped per project ('global' or the project name) so the
        # same name can exist in several project graphs without collision.
        for stmt in (
            "ALTER TABLE entities ADD COLUMN IF NOT EXISTS project TEXT NOT NULL DEFAULT 'global'",
            "ALTER TABLE entities DROP CONSTRAINT IF EXISTS entities_name_key",
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_entities_name_project ON entities(name, project)",
        ):
            try:
                cur.execute("SAVEPOINT ddl")
                cur.execute(stmt)
                cur.execute("RELEASE SAVEPOINT ddl")
            except Exception as e:
                cur.execute("ROLLBACK TO SAVEPOINT ddl")
                logger.warning("Schema statement skipped (%s): %s", str(stmt)[:80], e)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS relations (
                id SERIAL PRIMARY KEY,
                source_id INT REFERENCES entities(id) ON DELETE CASCADE,
                target_id INT REFERENCES entities(id) ON DELETE CASCADE,
                relation_type TEXT,
                weight FLOAT DEFAULT 1.0,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE (source_id, target_id, relation_type)
            )
        """)
        # Maps sanitized table suffixes back to the project name as provided.
        cur.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                suffix TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        # Dynamic IVFFlat lists based on the current size
        cur.execute("SELECT COUNT(*) FROM memories")
        count = cur.fetchone()[0] if cur.rowcount > 0 else 0
        lists = ivfflat_lists(count)
        if lists is None:
            drop_vector_indexes(cur, "memories")

        # The full-text index must use the same dictionary as the query side
        # (config.FTS_LANGUAGE). If the configured language changed, the old
        # index is dropped so it can be rebuilt with the new definition.
        cur.execute(
            "SELECT indexdef FROM pg_indexes "
            "WHERE schemaname = 'public' AND indexname = 'idx_memories_tsvector'"
        )
        row = cur.fetchone()
        if row and f"to_tsvector('{config.FTS_LANGUAGE}'" not in row[0]:
            cur.execute("DROP INDEX IF EXISTS idx_memories_tsvector")
            logger.info("Recreating full-text index with dictionary %s", config.FTS_LANGUAGE)

        index_statements = []
        if lists is not None:
            index_statements.append(
                "CREATE INDEX IF NOT EXISTS idx_memories_embedding ON memories "
                f"USING ivfflat (embedding vector_cosine_ops) WITH (lists = {lists})"
            )
        for idx in index_statements + [
            f"CREATE INDEX IF NOT EXISTS idx_memories_tsvector ON memories USING GIN (to_tsvector('{config.FTS_LANGUAGE}', content))",
            "CREATE INDEX IF NOT EXISTS idx_memories_hash ON memories(content_hash)",
            "CREATE INDEX IF NOT EXISTS idx_memories_entity ON memories(entity)",
            "CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(mem_type)",
            "CREATE INDEX IF NOT EXISTS idx_memories_active ON memories(superseded_by) WHERE superseded_by IS NULL AND deleted_at IS NULL",
            "ALTER TABLE memories ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ",
            "ALTER TABLE memories ADD COLUMN IF NOT EXISTS expires_at TIMESTAMPTZ",
            "ALTER TABLE memories ADD COLUMN IF NOT EXISTS valid_from TIMESTAMPTZ",
            "ALTER TABLE memories ADD COLUMN IF NOT EXISTS valid_to TIMESTAMPTZ",
            "ALTER TABLE memories ADD COLUMN IF NOT EXISTS source TEXT",
            "ALTER TABLE memories ADD COLUMN IF NOT EXISTS agent_id TEXT",
            "ALTER TABLE memories ADD COLUMN IF NOT EXISTS confidence FLOAT",
            "ALTER TABLE memories ADD COLUMN IF NOT EXISTS tags TEXT[] NOT NULL DEFAULT '{}'",
            "ALTER TABLE memories ADD COLUMN IF NOT EXISTS pinned BOOLEAN NOT NULL DEFAULT FALSE",
            "ALTER TABLE memories ADD COLUMN IF NOT EXISTS hard_deleted_at TIMESTAMPTZ",
            "CREATE INDEX IF NOT EXISTS idx_memories_validity ON memories(valid_from, valid_to)",
            "CREATE INDEX IF NOT EXISTS idx_memories_tags ON memories USING GIN (tags)",
            "CREATE INDEX IF NOT EXISTS idx_memories_pinned ON memories(id) WHERE pinned",
            # One active row per idempotency key. Expiry cannot be part of the
            # predicate (a partial index needs an IMMUTABLE expression), so an
            # expired row's key is cleared lazily before the next insert.
            "ALTER TABLE memories ADD COLUMN IF NOT EXISTS idempotency_key TEXT",
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_memories_idempotency_key "
            "ON memories(idempotency_key) WHERE idempotency_key IS NOT NULL "
            "AND superseded_by IS NULL AND deleted_at IS NULL",
            "CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name)",
            "CREATE INDEX IF NOT EXISTS idx_relations_source ON relations(source_id)",
            "CREATE INDEX IF NOT EXISTS idx_relations_target ON relations(target_id)",
        ]:
            try:
                cur.execute("SAVEPOINT ddl")
                cur.execute(idx)
                cur.execute("RELEASE SAVEPOINT ddl")
            except Exception as e:
                cur.execute("ROLLBACK TO SAVEPOINT ddl")
                logger.warning("Schema statement skipped (%s): %s", str(idx)[:80], e)
                if "tsvector" in idx:
                    # A missing/broken full-text index degrades every search
                    # silently; expose it through server_health instead of a
                    # log line alone.
                    record_warning(f"full-text index unavailable: {e}")
        dedup_sql = """
            DELETE FROM relations WHERE id IN (
                SELECT id FROM (
                    SELECT id, ROW_NUMBER() OVER (
                        PARTITION BY source_id, target_id, relation_type ORDER BY id
                    ) AS rn FROM relations
                ) sub WHERE rn > 1
            )
        """
        for stmt in (
            dedup_sql,
            "ALTER TABLE relations ADD CONSTRAINT uq_relations_edge UNIQUE (source_id, target_id, relation_type)",
        ):
            try:
                cur.execute("SAVEPOINT ddl")
                cur.execute(stmt)
                cur.execute("RELEASE SAVEPOINT ddl")
            except Exception as e:
                cur.execute("ROLLBACK TO SAVEPOINT ddl")
                logger.warning("Schema statement skipped (%s): %s", str(stmt)[:80], e)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()