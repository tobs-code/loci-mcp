"""The store owns table naming, SQL and savepoint policy."""

import pytest

from mcp_server.store import MemoryStore


class FakeCursor:
    def __init__(self):
        self.executed: list[tuple[str, object]] = []
        self.rowcount = 0
        self.description = [("value",)]
        self._row = (1,)

    def execute(self, sql, params=()):
        self.executed.append((sql, params))
        if sql.lstrip().upper().startswith("SELECT COUNT"):
            self._row = (1,)
            self.description = [("value",)]
        else:
            self._row = None
            self.description = [("value",)]
        return self

    def fetchone(self):
        return self._row

    def fetchall(self):
        return [] if self._row is None else [self._row]

    def close(self):
        pass


class FakeConn:
    def __init__(self):
        self.cursor_instance = FakeCursor()
        self.committed = 0
        self.rolled_back = 0
        self.closed = False

    def cursor(self):
        return self.cursor_instance

    def commit(self):
        self.committed += 1

    def rollback(self):
        self.rolled_back += 1

    def close(self):
        self.closed = True


def make_store():
    conn = FakeConn()
    return MemoryStore(conn=conn), conn


def test_table_and_tables_resolve_the_scope():
    store, _ = make_store()

    assert store.table("api") == "mem_api"
    assert store.table(None) == "memories"
    assert store.tables(None) == ["memories"]
    assert store.tables("api") == ["mem_api", "memories"]


def test_soft_forget_is_one_statement_with_the_sentinel():
    store, conn = make_store()
    conn.cursor_instance.rowcount = 1

    removed, tables = store.forget_rows("abc", tables=["memories"], soft=True)

    sql, params = conn.cursor_instance.executed[0]
    assert "UPDATE memories" in sql
    assert "COALESCE(superseded_by" in sql
    assert params == ("__forgotten__", "abc")
    assert removed == 1
    assert tables == ["memories"]


def test_hard_delete_has_no_sentinel_and_a_dry_run_writes_nothing():
    store, conn = make_store()

    store.forget_rows("abc", tables=["memories"], soft=False, dry_run=True)
    sql, params = conn.cursor_instance.executed[0]
    assert sql.startswith("SELECT COUNT(*)")
    assert params == ("abc",)

    conn.cursor_instance.executed.clear()
    store.forget_rows("abc", tables=["memories"], soft=False)
    sql, params = conn.cursor_instance.executed[0]
    assert sql == "DELETE FROM memories WHERE id = %s"
    assert params == ("abc",)


def test_attempt_rolls_back_only_the_failed_entry():
    store, conn = make_store()

    with pytest.raises(RuntimeError):
        with store.attempt("entry_0"):
            raise RuntimeError("boom")

    statements = [sql for sql, _ in conn.cursor_instance.executed]
    assert statements == [
        "SAVEPOINT entry_0",
        "ROLLBACK TO SAVEPOINT entry_0",
        "RELEASE SAVEPOINT entry_0",
    ]


def test_mark_superseded_touches_project_and_global_rows():
    store, conn = make_store()

    store.mark_superseded("old", "new", project="api")

    statements = [sql for sql, _ in conn.cursor_instance.executed]
    assert statements[0].startswith("UPDATE mem_api SET superseded_by")
    assert statements[1].startswith("UPDATE memories SET superseded_by")
    assert conn.cursor_instance.executed[0][1] == ("new", "old")
