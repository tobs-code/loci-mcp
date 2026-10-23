"""Tools delegate the transaction seam to MemoryStore."""

from pathlib import Path

import pytest

import mcp_server
from mcp_server import store
from mcp_server.store import MemoryStore

SOURCE_ROOT = Path(mcp_server.__file__).parent


class _FakeCursor:
    pass


class _FakeConn:
    def __init__(self):
        self.cursor_obj = _FakeCursor()
        self.commits = 0
        self.rollbacks = 0
        self.closed = 0

    def cursor(self):
        return self.cursor_obj

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        self.closed += 1


def _store(monkeypatch):
    conn = _FakeConn()
    monkeypatch.setattr(store, "get_conn", lambda: conn)
    return MemoryStore(), conn


def test_store_owns_cursor_commit_and_close(monkeypatch):
    db, conn = _store(monkeypatch)

    with db:
        assert db.cur is conn.cursor_obj

    assert (conn.commits, conn.rollbacks, conn.closed) == (1, 0, 1)


def test_store_rolls_back_and_closes_on_error(monkeypatch):
    db, conn = _store(monkeypatch)

    with pytest.raises(RuntimeError):
        with db:
            raise RuntimeError("boom")

    assert (conn.commits, conn.rollbacks, conn.closed) == (0, 1, 1)


def test_tools_do_not_open_their_own_connections():
    offenders = []
    for folder in (SOURCE_ROOT / "tools",):
        for path in folder.glob("*.py"):
            if "get_conn(" in path.read_text(encoding="utf-8"):
                offenders.append(path.relative_to(SOURCE_ROOT).as_posix())
    for path in ("graph.py", "search.py", "backup.py"):
        if "get_conn(" in (SOURCE_ROOT / path).read_text(encoding="utf-8"):
            offenders.append(path)

    assert offenders == []
