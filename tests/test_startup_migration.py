"""Startup migration only touches our memory tables.

A project table lives in `public`; AGE keeps label tables in its own schema.
A label whose name looks like a project table must not be read or migrated,
otherwise startup aborts halfway and real rows stay unmigrated.
"""

from mcp_server.config import SENTINEL_FORGOTTEN
from mcp_server.db import get_conn
from mcp_server.schema_migrator import ensure_project_table
from mcp_server.scope_catalog import _mem_table
from mcp_server.server import migrate_memory_tables

PROJECT = "startup-probe"
DECOY_SCHEMA = "startup_probe"


def test_migration_ignores_mem_tables_outside_public():
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(f"CREATE SCHEMA {DECOY_SCHEMA}")
        cur.execute(
            f"CREATE TABLE {DECOY_SCHEMA}.mem_decoy ("
            "content text, superseded_by text, "
            "deleted_at timestamptz, updated_at timestamptz)"
        )
        cur.execute(
            f"INSERT INTO {DECOY_SCHEMA}.mem_decoy "
            "(content, superseded_by, updated_at) "
            "VALUES ('decoy', 'forgotten', NOW())"
        )
        ensure_project_table(cur, PROJECT)
        cur.execute(
            f"INSERT INTO {_mem_table(PROJECT)} "
            "(id, content, content_hash, superseded_by) "
            "VALUES ('probe', 'real project row', 'hash', 'forgotten')"
        )
        conn.commit()

        migrate_memory_tables()

        cur.execute(
            f"SELECT superseded_by, deleted_at FROM {DECOY_SCHEMA}.mem_decoy"
        )
        assert cur.fetchone() == ("forgotten", None)

        cur.execute(
            f"SELECT superseded_by, deleted_at FROM {_mem_table(PROJECT)} "
            "WHERE id = 'probe'"
        )
        superseded_by, deleted_at = cur.fetchone()
        assert superseded_by == SENTINEL_FORGOTTEN
        assert deleted_at is not None
    finally:
        cur.execute(f"DROP SCHEMA IF EXISTS {DECOY_SCHEMA} CASCADE")
        cur.execute(f"DROP TABLE IF EXISTS {_mem_table(PROJECT)} CASCADE")
        conn.commit()
        conn.close()
