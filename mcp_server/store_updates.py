"""Mutation-shaped operations of a memory row.

`update_memory`, `update_batch` and the entity branch of `forget` all locate
an active row, read the stored metadata/entity and write a partial update.
Those statements live here, so the tools phrase an intent ("update these
fields", "forget this entity's rows") and never spell out a table name.

Every function takes the :class:`~mcp_server.store.MemoryStore` of the call:
savepoints, commit and rollback stay the caller's decision.
"""

from __future__ import annotations

import json
from typing import Any

from mcp_server.config import SENTINEL_FORGOTTEN

_RELATION_CONTEXT_FILTER = (
    "metadata->>'relation_context' = 'true' "
    "AND (metadata->>'from' = %s OR metadata->>'to' = %s)"
)


def lifecycle_row(store, memory_id: str,
                  tables: list[str]) -> tuple[str, dict] | None:
    """The row's lifecycle markers and its table, whatever its state."""
    for table in tables:
        row = store.one(
            f"SELECT superseded_by, deleted_at, expires_at FROM {table} "
            f"WHERE id = %s", (memory_id,))
        if row is not None:
            return table, row
    return None


def active_table(store, memory_id: str, tables: list[str], *,
                 revive_expired: bool = False) -> str | None:
    """The first table holding this active row (an expired one is inactive)."""
    active_sql = ("id = %s AND superseded_by IS NULL "
                  "AND deleted_at IS NULL")
    if not revive_expired:
        active_sql += " AND (expires_at IS NULL OR expires_at > NOW())"
    for table in tables:
        if store.scalar(f"SELECT 1 FROM {table} WHERE {active_sql}",
                        (memory_id,)):
            return table
    return None


def restore_rows(store, memory_id: str,
                 tables: list[str]) -> tuple[str | None, int]:
    """Undo a soft delete in the first table that holds one."""
    for table in tables:
        restored = store.execute(
            f"UPDATE {table} SET deleted_at = NULL, "
            f"superseded_by = NULLIF(superseded_by, %s), "
            f"updated_at = NOW() WHERE id = %s "
            f"AND deleted_at IS NOT NULL",
            (SENTINEL_FORGOTTEN, memory_id))
        if restored:
            return table, restored
    return None, 0


def stored_metadata(store, table: str, memory_id: str) -> dict:
    """The stored metadata object (empty when absent or unparseable)."""
    row = store.one(f"SELECT metadata FROM {table} WHERE id = %s",
                    (memory_id,))
    if not row or row.get("metadata") is None:
        return {}
    raw = row["metadata"]
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def stored_entity(store, table: str, memory_id: str) -> str | None:
    row = store.one(f"SELECT entity FROM {table} WHERE id = %s",
                    (memory_id,))
    return row.get("entity") if row else None


def apply_fields(store, table: str, memory_id: str, sets: list[str],
                 params: list[Any]) -> int:
    """Write the prepared field updates to an active row."""
    return store.execute(
        f"UPDATE {table} SET {', '.join(sets)} "
        "WHERE id = %s AND superseded_by IS NULL AND deleted_at IS NULL",
        tuple(params) + (memory_id,))


def forget_entity_rows(store, table: str, entity: str, *, soft: bool,
                       dry_run: bool) -> int:
    """Hide, delete or count every row of one entity in one table."""
    if dry_run:
        where = " AND deleted_at IS NULL" if soft else ""
        return store.scalar(
            f"SELECT COUNT(*) FROM {table} WHERE entity = %s{where}",
            (entity,)) or 0
    if soft:
        return store.execute(
            f"UPDATE {table} SET "
            f"superseded_by = COALESCE(superseded_by, %s), "
            f"deleted_at = NOW() "
            f"WHERE entity = %s AND deleted_at IS NULL",
            (SENTINEL_FORGOTTEN, entity))
    return store.execute(f"DELETE FROM {table} WHERE entity = %s", (entity,))


def forget_relation_context_rows(store, table: str, entity: str, *,
                                 soft: bool, dry_run: bool) -> int:
    """The same for the edge sentences that mention the entity."""
    if dry_run:
        where = " AND deleted_at IS NULL" if soft else ""
        return store.scalar(
            f"SELECT COUNT(*) FROM {table} "
            f"WHERE {_RELATION_CONTEXT_FILTER}{where}",
            (entity, entity)) or 0
    if soft:
        return store.execute(
            f"UPDATE {table} SET superseded_by = COALESCE(superseded_by, %s), "
            f"deleted_at = NOW() "
            f"WHERE deleted_at IS NULL AND {_RELATION_CONTEXT_FILTER}",
            (SENTINEL_FORGOTTEN, entity, entity))
    return store.execute(
        f"DELETE FROM {table} WHERE {_RELATION_CONTEXT_FILTER}",
        (entity, entity))


def count_entity_rows_incl_context(store, table: str, entity: str) -> int:
    """Rows of the entity plus the edge sentences naming it as either end."""
    return store.scalar(
        f"SELECT COUNT(*) FROM {table} WHERE entity = %s "
        f"OR ({_RELATION_CONTEXT_FILTER})",
        (entity, entity, entity)) or 0
