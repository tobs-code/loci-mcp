"""Maintenance operations over memory scopes.

Supersede-link inspection, pruning and the scope DDL of `drop_project`.
These are the statements that repair or remove rows, so they live next to the
store instead of a tool module: a tool asks for an operation and renders its
result.

Each function takes the :class:`~mcp_server.store.MemoryStore` of the call.
"""

from __future__ import annotations

from mcp_server.config import SENTINEL_FORGOTTEN


def all_ids(store, tables: list[str]) -> set[str]:
    """Every memory id across the given tables (link validity is global)."""
    ids: set[str] = set()
    for table in tables:
        store.cur.execute(f"SELECT id FROM {table}")
        ids |= {row[0] for row in store.cur.fetchall()}
    return ids


def supersede_links(store, table: str) -> list[tuple]:
    """(id, superseded_by, entity, content) for real supersede markers."""
    store.cur.execute(
        f"SELECT id, superseded_by, entity, content FROM {table} "
        f"WHERE superseded_by IS NOT NULL "
        "AND superseded_by <> %s", (SENTINEL_FORGOTTEN,))
    return store.cur.fetchall()


def expired_active_count(store, table: str) -> int:
    """Active-looking rows whose TTL has already passed."""
    return store.scalar(
        f"SELECT COUNT(*) FROM {table} WHERE superseded_by IS NULL "
        f"AND deleted_at IS NULL AND expires_at IS NOT NULL "
        f"AND expires_at <= NOW()") or 0


def clear_supersede(store, table: str, memory_id: str) -> int:
    """Drop one broken supersede link so the row is readable history again."""
    return store.execute(
        f"UPDATE {table} SET superseded_by = NULL WHERE id = %s",
        (memory_id,))


def prune_rows(store, table: str, where: str, params: tuple | list, *,
               dry_run: bool) -> int:
    """Count or delete the inactive rows matching the prune predicate."""
    if dry_run:
        return store.scalar(
            f"SELECT COUNT(*) FROM {table} WHERE {where}", params) or 0
    return store.execute(f"DELETE FROM {table} WHERE {where}", params)


def mirror_counts(store) -> tuple[int, int]:
    """Graph mirror sizes: (entities, relations)."""
    store.cur.execute("""
            SELECT
                (SELECT COUNT(*) FROM entities),
                (SELECT COUNT(*) FROM relations)
        """)
    row = store.cur.fetchone()
    return row if row else (0, 0)


def graph_scopes(store) -> set[str | None]:
    store.cur.execute("SELECT DISTINCT project FROM entities")
    return {row[0] for row in store.cur.fetchall()}


def relation_scopes(store) -> set[str | None]:
    """Scopes of the entities a relation connects."""
    store.cur.execute(
        "SELECT DISTINCT e.project FROM relations r "
        "JOIN entities e ON e.id = r.source_id")
    return {row[0] for row in store.cur.fetchall()}


def entity_count(store, scope: str) -> int:
    return store.scalar(
        "SELECT COUNT(*) FROM entities WHERE project = %s", (scope,)) or 0


def relation_count(store, scope: str) -> int:
    return store.scalar(
        "SELECT COUNT(*) FROM relations WHERE source_id IN "
        "(SELECT id FROM entities WHERE project = %s) OR target_id IN "
        "(SELECT id FROM entities WHERE project = %s)",
        (scope, scope)) or 0


def drop_scope_table(store, table: str, suffix: str) -> None:
    """Remove one project's memory table and its registry entry."""
    store.cur.execute("SET search_path TO public")
    store.cur.execute(f"DROP TABLE IF EXISTS {table}")
    store.cur.execute("DELETE FROM projects WHERE suffix = %s", (suffix,))
