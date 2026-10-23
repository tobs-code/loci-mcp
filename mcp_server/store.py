"""Memory store: one owner of connection, transaction and cursor lifetime.

Tools validate arguments and shape the envelope; this module owns the
mechanics of a unit of work. A store keeps its connection open across several
queries, commits only when the caller asks, and rolls back+closes when the
block raises. That is what keeps a page from being read through two separate
connections.
"""

from __future__ import annotations

from mcp_server.db import get_conn


class MemoryStore:
    """A connection and cursor with an explicit transaction boundary."""

    def __init__(self):
        self.conn = get_conn()
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


def open_store() -> MemoryStore:
    """Open a store for code that stages several pages before committing."""
    return MemoryStore()
