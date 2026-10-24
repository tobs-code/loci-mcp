"""The remaining SQL of the tool modules is an operation on the store."""

import pytest

from mcp_server import store_maintenance, store_rows, store_updates
from mcp_server.store import MemoryStore


class FakeCursor:
    def __init__(self):
        self.executed: list[tuple[str, object]] = []
        self.rowcount = 0
        self.description = [("value",)]
        self.rows: list[tuple] = []

    def execute(self, sql, params=()):
        self.executed.append((sql, params))
        if sql.lstrip().upper().startswith("SELECT COUNT"):
            self.rows = [(0,)]
        return self

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return list(self.rows)

    def close(self):
        pass


class FakeConn:
    def __init__(self):
        self.cursor_instance = FakeCursor()
        self.committed = 0
        self.rolled_back = 0

    def cursor(self):
        return self.cursor_instance

    def commit(self):
        self.committed += 1

    def rollback(self):
        self.rolled_back += 1

    def close(self):
        pass


def make_store():
    conn = FakeConn()
    return MemoryStore(conn=conn), conn.cursor_instance


def test_lifecycle_row_returns_the_table_and_markers():
    store, cur = make_store()
    cur.description = [("superseded_by",), ("deleted_at",), ("expires_at",)]
    cur.rows = [(None, "x", None)]

    found = store_updates.lifecycle_row(store, "abc", ["memories"])

    assert found[0] == "memories"
    assert found[1]["deleted_at"] == "x"
    assert cur.executed[0][1] == ("abc",)


def test_active_table_adds_the_ttl_predicate_unless_reviving():
    store, cur = make_store()

    store_updates.active_table(store, "abc", ["memories"])
    assert "expires_at > NOW()" in cur.executed[0][0]

    cur.executed.clear()
    store_updates.active_table(store, "abc", ["memories"], revive_expired=True)
    assert "expires_at > NOW()" not in cur.executed[0][0]


def test_restore_rows_uses_nullif_and_the_sentinel():
    store, cur = make_store()
    cur.rowcount = 1

    table, restored = store_updates.restore_rows(store, "abc", ["memories"])

    sql, params = cur.executed[0]
    assert "NULLIF(superseded_by, %s)" in sql
    assert params == ("__forgotten__", "abc")
    assert (table, restored) == ("memories", 1)


def test_stored_metadata_parses_json_and_defaults_to_empty():
    store, cur = make_store()
    cur.description = [("metadata",)]
    cur.rows = [('{"a": 1}',)]

    assert store_updates.stored_metadata(store, "memories", "abc") == {"a": 1}

    cur.rows = [("not json",)]
    assert store_updates.stored_metadata(store, "memories", "abc") == {}


def test_apply_fields_keeps_the_active_guard():
    store, cur = make_store()
    cur.rowcount = 1

    updated = store_updates.apply_fields(
        store, "memories", "abc", ["importance = %s"], [0.9])

    sql, params = cur.executed[0]
    assert sql == ("UPDATE memories SET importance = %s WHERE id = %s "
                   "AND superseded_by IS NULL AND deleted_at IS NULL")
    assert params == (0.9, "abc")
    assert updated == 1


def test_forget_entity_rows_dry_run_counts_only_active_rows():
    store, cur = make_store()

    store_updates.forget_entity_rows(
        store, "memories", "API", soft=True, dry_run=True)
    sql, params = cur.executed[0]
    assert sql.startswith("SELECT COUNT(*) FROM memories")
    assert "deleted_at IS NULL" in sql
    assert params == ("API",)

    cur.executed.clear()
    cur.rowcount = 2
    store_updates.forget_entity_rows(
        store, "memories", "API", soft=False, dry_run=False)
    assert cur.executed[0] == ("DELETE FROM memories WHERE entity = %s",
                               ("API",))


def test_relation_context_forget_passes_the_entity_twice():
    store, cur = make_store()
    cur.rowcount = 1

    store_updates.forget_relation_context_rows(
        store, "memories", "API", soft=False, dry_run=False)

    sql, params = cur.executed[0]
    assert "metadata->>'from' = %s OR metadata->>'to' = %s" in sql
    assert params == ("API", "API")


def test_count_entity_rows_incl_context_passes_the_entity_three_times():
    store, cur = make_store()

    store_updates.count_entity_rows_incl_context(store, "memories", "API")

    assert cur.executed[0][1] == ("API", "API", "API")


def test_entity_present_and_graph_projects_read_the_selected_tables():
    store, cur = make_store()

    assert store_rows.entity_present(store, "mem_api", "API") is False
    assert cur.executed[0] == (
        "SELECT 1 FROM mem_api WHERE entity = %s LIMIT 1", ("API",))

    cur.rows = [("api",), (None,)]
    assert store_rows.graph_projects(store, "API") == ["api", None]


def test_memory_only_names_use_the_active_predicate():
    store, cur = make_store()
    cur.rows = [("API",)]

    assert store_rows.memory_only_names(store, "memories") == ["API"]
    sql = cur.executed[0][0]
    assert "superseded_by IS NULL" in sql
    assert "relation_context" in sql


def test_relation_context_rows_name_the_scope_and_drop_the_embedding():
    store, cur = make_store()
    cur.description = [("id",), ("content",), ("embedding",)]
    cur.rows = [("abc", "API uses Postgres", [0.1])]

    rows = store_rows.relation_context_rows(
        store, "API", ["memories"], include_superseded=False)

    assert rows == [{"id": "abc", "content": "API uses Postgres",
                     "_project": "global"}]


def test_all_ids_and_supersede_links():
    store, cur = make_store()

    cur.rows = [("a",)]
    assert store_maintenance.all_ids(store, ["memories"]) == {"a"}

    cur.executed.clear()
    cur.rows = [("old", "new", "API", "text")]
    assert store_maintenance.supersede_links(store, "memories") == [
        ("old", "new", "API", "text")]
    assert cur.executed[0][1] == ("__forgotten__",)


def test_clear_supersede_and_expired_active_count():
    store, cur = make_store()
    cur.rowcount = 1

    store_maintenance.clear_supersede(store, "memories", "abc")
    assert cur.executed[0] == (
        "UPDATE memories SET superseded_by = NULL WHERE id = %s", ("abc",))

    store_maintenance.expired_active_count(store, "memories")
    assert "expires_at <= NOW()" in cur.executed[-1][0]


def test_prune_rows_counts_on_dry_run_and_deletes_otherwise():
    store, cur = make_store()

    store_maintenance.prune_rows(
        store, "memories", "deleted_at IS NOT NULL", (1,), dry_run=True)
    assert cur.executed[0][0].startswith("SELECT COUNT(*)")

    cur.executed.clear()
    cur.rowcount = 3
    removed = store_maintenance.prune_rows(
        store, "memories", "deleted_at IS NOT NULL", (1,), dry_run=False)
    assert cur.executed[0] == (
        "DELETE FROM memories WHERE deleted_at IS NOT NULL", (1,))
    assert removed == 3


def test_drop_scope_table_leaves_the_graph_schema_first():
    store, cur = make_store()

    store_maintenance.drop_scope_table(store, "mem_api", "api")

    statements = [sql for sql, _ in cur.executed]
    assert statements == [
        "SET search_path TO public",
        "DROP TABLE IF EXISTS mem_api",
        "DELETE FROM projects WHERE suffix = %s",
    ]


def test_store_exposes_savepoint_controls_and_a_ping():
    store, cur = make_store()

    store.rollback_to("entry_0")
    store.release("entry_0")
    store.ping()

    assert [sql for sql, _ in cur.executed] == [
        "ROLLBACK TO SAVEPOINT entry_0",
        "RELEASE SAVEPOINT entry_0",
        "SELECT 1",
    ]


def test_relation_count_counts_both_endpoints():
    store, cur = make_store()

    store_maintenance.relation_count(store, "api")

    sql, params = cur.executed[0]
    assert "source_id IN" in sql and "target_id IN" in sql
    assert params == ("api", "api")


@pytest.mark.parametrize("name", [
    "entity_present", "graph_projects", "memory_only_names",
    "relation_context_rows", "entity_row_count",
    "lifecycle_row", "active_table", "apply_fields", "forget_entity_rows",
    "all_ids", "supersede_links", "prune_rows", "drop_scope_table",
])
def test_public_operation_is_exposed(name):
    assert any(hasattr(module, name) for module in
               (store_rows, store_updates, store_maintenance))
