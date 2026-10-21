import logging
import os
import psycopg2
import psycopg2.extras
from contextlib import contextmanager

logger = logging.getLogger(__name__)

DATABASE_URL = os.environ.get(
    "MEMORY_DATABASE_URL",
    "postgresql://memory:geheim@localhost:5432/memory",
)


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
        
        if count < 50000:
            lists = 100
        elif count < 200000:
            lists = 200
        else:
            lists = 400
        
        for idx in [
            f"CREATE INDEX IF NOT EXISTS idx_memories_embedding ON memories USING ivfflat (embedding vector_cosine_ops) WITH (lists = {lists})",
            "CREATE INDEX IF NOT EXISTS idx_memories_tsvector ON memories USING GIN (to_tsvector('german', content))",
            "CREATE INDEX IF NOT EXISTS idx_memories_hash ON memories(content_hash)",
            "CREATE INDEX IF NOT EXISTS idx_memories_entity ON memories(entity)",
            "CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(mem_type)",
            "CREATE INDEX IF NOT EXISTS idx_memories_active ON memories(superseded_by) WHERE superseded_by IS NULL AND deleted_at IS NULL",
            "ALTER TABLE memories ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ",
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