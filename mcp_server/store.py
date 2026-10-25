"""Memory store: the one owner of a memory's persistence.

Tools validate arguments and shape the envelope; this module owns the
mechanics of a unit of work *and* the domain operations on a memory table.
Table naming, SQL, parameters and savepoint policy live here, so a tool never
interpolates a table name or spells out a lifecycle update.

A store keeps its connection open across several operations, commits only when
the caller asks, and rolls back+closes when the block raises. That is what
keeps a page from being read through two separate connections.

The store is also the seam for tests: every operation is an ordinary method, so
a fake store that records what was asked can replace the PostgreSQL adapter.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any, Iterator

from mcp_server.config import SENTINEL_FORGOTTEN
from mcp_server.db import get_conn
from mcp_server.lifecycle import ACTIVE_ROW_SQL, counts as lifecycle_counts
from mcp_server.scope_catalog import (
    _all_mem_tables,
    _mem_table,
    _project_exists,
    _scope_of_table,
)
from mcp_server.schema_migrator import ensure_project_table
from mcp_server.scoring import _content_hash, _gen_id, _now


class MemoryStore:
    """A connection and cursor with explicit transaction and domain policy."""

    def __init__(self, conn=None):
        self.conn = conn if conn is not None else get_conn()
        self.cur = self.conn.cursor()
        self._closed = False

    def __enter__(self) -> "MemoryStore":
        return self

    def commit(self) -> None:
        self.conn.commit()

    def rollback(self) -> None:
        self.conn.rollback()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self.conn.close()

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if exc_type is not None:
                self.conn.rollback()
            else:
                self.conn.commit()
        finally:
            self.close()

    # -- generic query helpers ------------------------------------------------

    def execute(self, sql: str, params: tuple | list = ()) -> int:
        """Run one statement and return the affected row count."""
        self.cur.execute(sql, params)
        return max(self.cur.rowcount, 0)

    def scalar(self, sql: str, params: tuple | list = ()) -> Any:
        self.cur.execute(sql, params)
        row = self.cur.fetchone()
        return row[0] if row else None

    def one(self, sql: str, params: tuple | list = ()) -> dict | None:
        self.cur.execute(sql, params)
        row = self.cur.fetchone()
        return self._as_dict(row) if row is not None else None

    def all(self, sql: str, params: tuple | list = ()) -> list[dict]:
        self.cur.execute(sql, params)
        return [self._as_dict(row) for row in self.cur.fetchall()]

    def _as_dict(self, row) -> dict:
        columns = [d[0] for d in self.cur.description]
        return dict(zip(columns, row))

    @contextmanager
    def attempt(self, name: str = "memory_store") -> Iterator[None]:
        """A savepoint around one entry; a failure rolls back only that entry."""
        self.cur.execute(f"SAVEPOINT {name}")
        try:
            yield
        except Exception:
            self.cur.execute(f"ROLLBACK TO SAVEPOINT {name}")
            self.cur.execute(f"RELEASE SAVEPOINT {name}")
            raise
        else:
            self.cur.execute(f"RELEASE SAVEPOINT {name}")

    savepoint = attempt

    def open_savepoint(self, name: str) -> None:
        """Open a savepoint whose lifetime the caller manages manually."""
        self.cur.execute(f"SAVEPOINT {name}")

    def rollback_to(self, name: str) -> None:
        """Discard the work of an open savepoint (a per-entry failure)."""
        self.cur.execute(f"ROLLBACK TO SAVEPOINT {name}")

    def release(self, name: str) -> None:
        """Keep an open savepoint's work and free the name."""
        self.cur.execute(f"RELEASE SAVEPOINT {name}")

    def ping(self) -> None:
        """One round trip; raises when the connection is unusable."""
        self.cur.execute("SELECT 1")

    # -- scope / table -------------------------------------------------------

    @staticmethod
    def table(project: str | None) -> str:
        """The memory table of one scope (the global table for `None`)."""
        return _mem_table(project)

    def tables(self, project: str | None = None, *,
               every_scope: bool = False) -> list[str]:
        """Tables to consult for a lookup.

        `every_scope` covers every project table then the global one; a project
        covers that project plus global; without a project only global.
        """
        if every_scope:
            return _all_mem_tables(self.cur)
        if project:
            return [_mem_table(project), "memories"]
        return ["memories"]

    def ensure_scope(self, project: str | None) -> bool:
        """Create a project table on demand; return True when it was new."""
        if not project:
            return False
        created = not _project_exists(self.cur, project)
        ensure_project_table(self.cur, project)
        return created

    def scope_of_table(self, table: str) -> str:
        return _scope_of_table(self.cur, table)

    # -- memory lookups ------------------------------------------------------

    def find(self, memory_id: str, *, project: str | None = None,
             tables: list[str] | None = None, where: str = "TRUE",
             columns: str = "*") -> tuple[dict | None, str | None]:
        """The first matching row and its table, or (None, None)."""
        candidates = tables if tables is not None else self.tables(project)
        for table in candidates:
            row = self.one(
                f"SELECT {columns} FROM {table} WHERE id = %s AND {where}",
                (memory_id,),
            )
            if row is not None:
                return row, table
        return None, None

    def locate(self, memory_id: str, *, project: str | None = None,
               tables: list[str] | None = None) -> tuple[dict | None, str | None]:
        """A row regardless of visibility, with its lifecycle columns."""
        return self.find(
            memory_id, project=project, tables=tables,
            columns="superseded_by, deleted_at, expires_at")

    def scopes_of(self, memory_id: str) -> list[str]:
        """Readable scopes that hold this id (a copy can live in several)."""
        found: list[str] = []
        for table in _all_mem_tables(self.cur):
            if self.scalar(f"SELECT 1 FROM {table} WHERE id = %s",
                           (memory_id,)):
                found.append(_scope_of_table(self.cur, table))
        return found

    def nearest(self, *, project: str | None, embedding: list[float] | None,
                limit: int, **kwargs) -> list:
        """Nearest active memories; the scan module owns the one definition."""
        from mcp_server.contradiction_scan import nearest_active
        return nearest_active(
            self.cur, project=project, embedding=embedding, limit=limit,
            **kwargs)

    def counts(self, project: str | None = None) -> dict[str, int]:
        return lifecycle_counts(self.cur, _mem_table(project))

    def lock_idempotency_key(self, table: str, key: str) -> None:
        """Serialize first-writes that carry the same key in one scope.

        The lock is transaction-scoped: a concurrent caller with the same key
        waits here, then sees the committed row in the replay check instead of
        running dedupe/supersede side effects and losing the insert race.
        """
        self.cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))",
                         (f"{table}:{key}",))

    def clear_stale_idempotency_keys(self, key: str, *, table: str) -> int:
        """Free a key held by an expired row (the index cannot see expiry)."""
        return self.execute(
            f"UPDATE {table} SET idempotency_key = NULL "
            "WHERE idempotency_key = %s AND superseded_by IS NULL "
            "AND deleted_at IS NULL AND expires_at IS NOT NULL "
            "AND expires_at <= NOW()",
            (key,))

    def find_idempotency_key(self, key: str, *,
                             table: str) -> dict | None:
        """The active row already stored for `key` in one scope, or None."""
        return self.one(
            f"SELECT * FROM {table} WHERE idempotency_key = %s "
            "AND superseded_by IS NULL AND deleted_at IS NULL "
            "AND (expires_at IS NULL OR expires_at > NOW())",
            (key,))

    # -- domain writes -------------------------------------------------------

    def insert(self, content: str, embedding_vec: list[float] | None,
               mem_type: str, entity: str | None, importance: float,
               metadata: dict, *, project: str | None = None,
               scope: str | None = None, expires_at: str | None = None,
               superseded_by: str | None = None,
               memory_id: str | None = None,
               idempotency_key: str | None = None,
               valid_from: str | None = None,
               valid_to: str | None = None,
               source: str | None = None,
               agent_id: str | None = None,
               confidence: float | None = None,
               tags: list[str] | None = None,
               pinned: bool | None = None) -> str:
        """Insert one active memory and return its id."""
        mem_id = memory_id or _gen_id()
        now = _now()
        table = _mem_table(project)
        effective_scope = scope or ("project" if project else "global")
        params: list[Any] = [
            mem_id, content, _content_hash(content), embedding_vec, mem_type,
            effective_scope, entity, importance, json.dumps(metadata),
            superseded_by, now, now,
        ]
        extra_columns = ""
        extra_values = ""
        if expires_at is not None:
            extra_columns += ", expires_at"
            extra_values += ", %s::timestamptz"
            params.append(expires_at)
        if idempotency_key is not None:
            extra_columns += ", idempotency_key"
            extra_values += ", %s"
            params.append(idempotency_key)
        if valid_from is not None:
            extra_columns += ", valid_from"
            extra_values += ", %s::timestamptz"
            params.append(valid_from)
        if valid_to is not None:
            extra_columns += ", valid_to"
            extra_values += ", %s::timestamptz"
            params.append(valid_to)
        if source is not None:
            extra_columns += ", source"
            extra_values += ", %s"
            params.append(source)
        if agent_id is not None:
            extra_columns += ", agent_id"
            extra_values += ", %s"
            params.append(agent_id)
        if confidence is not None:
            extra_columns += ", confidence"
            extra_values += ", %s"
            params.append(confidence)
        if tags is not None:
            extra_columns += ", tags"
            extra_values += ", %s::text[]"
            params.append(list(tags))
        if pinned:
            extra_columns += ", pinned"
            extra_values += ", %s"
            params.append(True)
        self.cur.execute(
            f"""INSERT INTO {table}
                (id, content, content_hash, embedding, mem_type, scope, entity,
                 importance, metadata, superseded_by, created_at, updated_at
                 {extra_columns})
            VALUES (%s, %s, %s, %s::vector, %s, %s, %s, %s, %s, %s, %s, %s
                    {extra_values})""",
            tuple(params),
        )
        return mem_id

    def mark_superseded(self, memory_id: str, superseded_by: str, *,
                        project: str | None = None,
                        target: str | None = None) -> int:
        """Mark a memory as replaced, in the table the id was resolved in.

        `target` is the scope the id belongs to ("project" or "global").
        A project write that replaces a shared fact must still hide that fact
        in every scope, but a same-named id in the other table is a different
        memory: updating it would invalidate an unrelated row. Without a
        resolved target the historical behaviour (project table plus global)
        is kept for callers that cannot know the scope.

        The row's validity window is closed at the current time: a superseded
        fact stopped being true when it was replaced, and an `as_of` query for
        a later moment must not match it anymore. An existing `valid_to` is
        kept (COALESCE), so a deliberately backdated window is not widened.

        Only ACTIVE rows are marked. This is the helper every ID-based
        supersede path funnels through, so the active check belongs in the
        statement as well as in the caller's lookup: a row forgotten or
        invalidated between the two keeps its sentinel and stays restorable.
        """
        if target == "global":
            tables = ["memories"]
        elif target == "project":
            tables = [_mem_table(project)]
        else:
            tables = [_mem_table(project)]
            if project:
                tables.append("memories")
        touched = 0
        for table in dict.fromkeys(tables):
            touched += self.execute(
                f"UPDATE {table} SET superseded_by = %s, "
                f"valid_to = COALESCE(valid_to, NOW()), updated_at = NOW() "
                f"WHERE id = %s AND {ACTIVE_ROW_SQL}",
                (superseded_by, memory_id),
            )
        return touched

    def rows_by_ids(self, memory_ids: list[str], *, table: str | None = None,
                    project: str | None = None,
                    columns: str = "*") -> list[dict]:
        """Rows of the given ids from one table (id order is not kept)."""
        target = table or _mem_table(project)
        if not memory_ids:
            return []
        return self.all(
            f"SELECT {columns} FROM {target} WHERE id = ANY(%s)",
            (list(memory_ids),),
        )

    def supersede_many(self, memory_ids: list[str], superseded_by: str, *,
                       table: str | None = None,
                       project: str | None = None) -> int:
        """Mark ACTIVE rows of ONE table as replaced by the same id.

        An inactive source is never rewritten: the merge caller validates the
        whole set first, and the predicate keeps that promise if a row is
        forgotten before this statement runs.
        """
        target = table or _mem_table(project)
        if not memory_ids:
            return 0
        return self.execute(
            f"UPDATE {target} SET superseded_by = %s, "
            f"valid_to = COALESCE(valid_to, NOW()), updated_at = NOW() "
            f"WHERE id = ANY(%s) AND {ACTIVE_ROW_SQL}",
            (superseded_by, list(memory_ids)),
        )

    def forget_rows(self, memory_id: str, *, project: str | None = None,
                    tables: list[str] | None = None, soft: bool = True,
                    dry_run: bool = False) -> tuple[int, list[str]]:
        """Forget one id in the selected tables; return (count, tables)."""
        candidates = tables if tables is not None else self.tables(project)
        total = 0
        touched: list[str] = []
        for table in candidates:
            if dry_run:
                where = ("id = %s AND deleted_at IS NULL" if soft
                         else "id = %s")
                count = self.scalar(f"SELECT COUNT(*) FROM {table} WHERE {where}",
                                    (memory_id,)) or 0
            elif soft:
                count = self.execute(
                    f"UPDATE {table} SET superseded_by = "
                    f"COALESCE(superseded_by, %s), deleted_at = NOW() "
                    f"WHERE id = %s AND deleted_at IS NULL",
                    (SENTINEL_FORGOTTEN, memory_id))
            else:
                count = self.execute(
                    f"UPDATE {table} SET superseded_by = "
                    f"COALESCE(superseded_by, %s), "
                    f"deleted_at = COALESCE(deleted_at, NOW()), "
                    f"hard_deleted_at = NOW(), updated_at = NOW() "
                    f"WHERE id = %s",
                    (SENTINEL_FORGOTTEN, memory_id))
            if count:
                total += count
                touched.append(table)
        return total, touched

    def forget_scope(self, project: str | None, *, soft: bool = True,
                     dry_run: bool = False) -> tuple[int, dict[str, int]]:
        """Forget every memory of one scope; return (count, lifecycle counts)."""
        table = _mem_table(project)
        summary = lifecycle_counts(self.cur, table)
        if dry_run:
            where = ("WHERE deleted_at IS NULL" if soft
                     else "WHERE hard_deleted_at IS NULL")
            count = self.scalar(f"SELECT COUNT(*) FROM {table} {where}") or 0
        elif soft:
            count = self.execute(
                f"UPDATE {table} SET superseded_by = "
                f"COALESCE(superseded_by, %s), deleted_at = NOW() "
                f"WHERE deleted_at IS NULL", (SENTINEL_FORGOTTEN,))
        else:
            count = self.execute(
                f"UPDATE {table} SET superseded_by = "
                f"COALESCE(superseded_by, %s), "
                f"deleted_at = COALESCE(deleted_at, NOW()), "
                f"hard_deleted_at = NOW(), updated_at = NOW() "
                f"WHERE hard_deleted_at IS NULL", (SENTINEL_FORGOTTEN,))
        return count, summary

    def restore(self, memory_id: str, *, project: str | None = None,
                tables: list[str] | None = None) -> int:
        """Undo a soft delete (never a real supersede)."""
        candidates = tables if tables is not None else self.tables(project)
        total = 0
        for table in candidates:
            total += self.execute(
                f"UPDATE {table} SET deleted_at = NULL, "
                f"hard_deleted_at = NULL, "
                f"superseded_by = CASE WHEN superseded_by = %s THEN NULL "
                f"ELSE superseded_by END, updated_at = NOW() "
                f"WHERE id = %s AND deleted_at IS NOT NULL",
                (SENTINEL_FORGOTTEN, memory_id))
        return total

    def touch(self, memory_id: str, *, project: str | None = None,
              tables: list[str] | None = None) -> int:
        candidates = tables if tables is not None else self.tables(project)
        total = 0
        for table in candidates:
            total += self.execute(
                f"UPDATE {table} SET updated_at = NOW() WHERE id = %s",
                (memory_id,))
        return total


def open_store() -> MemoryStore:
    """Open a store for code that stages several pages before committing."""
    return MemoryStore()

# Capability modules take this store; re-exported for discoverability.
from mcp_server import (  # noqa: E402,F401
    store_maintenance,
    store_rows,
    store_updates,
)