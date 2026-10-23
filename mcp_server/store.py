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
from mcp_server.lifecycle import counts as lifecycle_counts
from mcp_server.scopes import (
    _all_mem_tables,
    _mem_table,
    _project_exists,
    _ensure_project_table,
    _scope_of_table,
)
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
        _ensure_project_table(self.cur, project)
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

    def counts(self, project: str | None = None) -> dict[str, int]:
        return lifecycle_counts(self.cur, _mem_table(project))

    # -- domain writes -------------------------------------------------------

    def insert(self, content: str, embedding_vec: list[float] | None,
               mem_type: str, entity: str | None, importance: float,
               metadata: dict, *, project: str | None = None,
               scope: str | None = None, expires_at: str | None = None,
               superseded_by: str | None = None,
               memory_id: str | None = None) -> str:
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
        ttl = ""
        if expires_at is not None:
            ttl = ", expires_at"
            params.append(expires_at)
        self.cur.execute(
            f"""INSERT INTO {table}
                (id, content, content_hash, embedding, mem_type, scope, entity,
                 importance, metadata, superseded_by, created_at, updated_at{ttl})
            VALUES (%s, %s, %s, %s::vector, %s, %s, %s, %s, %s, %s, %s, %s
                    {', %s::timestamptz' if expires_at is not None else ''})""",
            tuple(params),
        )
        return mem_id

    def mark_superseded(self, memory_id: str, superseded_by: str, *,
                        project: str | None = None) -> int:
        """Mark a memory as replaced, in its scope and (for a project) global.

        The global update is deliberately unconditional: a project write may
        replace a shared fact, and that fact must disappear from every scope.
        """
        touched = self.execute(
            f"UPDATE {_mem_table(project)} SET superseded_by = %s, "
            f"updated_at = NOW() WHERE id = %s",
            (superseded_by, memory_id),
        )
        if project:
            touched += self.execute(
                "UPDATE memories SET superseded_by = %s, updated_at = NOW() "
                "WHERE id = %s", (superseded_by, memory_id))
        return touched

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
                count = self.execute(f"DELETE FROM {table} WHERE id = %s",
                                     (memory_id,))
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
            where = "WHERE deleted_at IS NULL" if soft else ""
            count = self.scalar(f"SELECT COUNT(*) FROM {table} {where}") or 0
        elif soft:
            count = self.execute(
                f"UPDATE {table} SET superseded_by = "
                f"COALESCE(superseded_by, %s), deleted_at = NOW() "
                f"WHERE deleted_at IS NULL", (SENTINEL_FORGOTTEN,))
        else:
            count = self.execute(f"DELETE FROM {table}")
        return count, summary

    def restore(self, memory_id: str, *, project: str | None = None,
                tables: list[str] | None = None) -> int:
        """Undo a soft delete (never a real supersede)."""
        candidates = tables if tables is not None else self.tables(project)
        total = 0
        for table in candidates:
            total += self.execute(
                f"UPDATE {table} SET deleted_at = NULL, "
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
