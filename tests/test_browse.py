"""One browse pipeline owns visibility, cross-scope dedupe and paging."""

import re
from pathlib import Path

from mcp_server.browse import BrowseRequest, browse
from mcp_server.store import MemoryStore

INDEX_COLUMNS = [
    "id", "content_hash", "superseded_by", "deleted_at", "expires_at",
    "_order",
]
FULL_COLUMNS = [
    "id", "content", "content_hash", "superseded_by", "deleted_at",
    "expires_at", "updated_at",
]


class FakeCursor:
    def __init__(self, index_rows, full_rows, project_names=None,
                 existing_tables=None, project_tables=None):
        self.index_rows = index_rows
        self.full_rows = full_rows
        self.project_names = project_names or {"api": "api"}
        self.existing_tables = set(existing_tables or {"mem_api"})
        self.project_tables = list(project_tables or ["mem_api"])
        self.executed = []
        self.description = []
        self._rows = []
        self._row = None

    def _set(self, rows, columns):
        self._rows = list(rows)
        self._row = self._rows[0] if self._rows else None
        self.description = [(column,) for column in columns]

    def execute(self, sql, params=()):
        self.executed.append((sql, params))
        text = " ".join(sql.split())
        if "information_schema.tables" in text:
            if "EXISTS" in text:
                name = params[0] if params else None
                self._set([(1,)] if name in self.existing_tables else [(0,)],
                          ["exists"])
            else:
                self._set([(table,) for table in self.project_tables],
                          ["table_name"])
        elif "FROM projects" in text:
            self._set([(suffix, name)
                       for suffix, name in self.project_names.items()],
                      ["suffix", "name"])
        elif "FROM entities WHERE project" in text:
            self._set([(0,)], ["count"])
        elif text.startswith("SELECT id, content_hash"):
            table = re.search(r"FROM (mem_[A-Za-z0-9_]+|memories)",
                              text).group(1)
            self._set(self.index_rows.get(table, []), INDEX_COLUMNS)
        elif "SELECT * FROM" in text and "id = ANY" in text:
            table = re.search(r"FROM (mem_[A-Za-z0-9_]+|memories)",
                              text).group(1)
            wanted = set(params[0])
            rows = [row for row in self.full_rows.get(table, [])
                    if row["id"] in wanted]
            self._set([tuple(row.get(column) for column in FULL_COLUMNS)
                       for row in rows], FULL_COLUMNS)
        else:
            raise AssertionError(f"unexpected SQL: {text}")
        return self

    def fetchone(self):
        return self._row

    def fetchall(self):
        return self._rows

    def close(self):
        pass


class FakeConn:
    def __init__(self, cursor):
        self.cursor_instance = cursor
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


def make_store(cursor):
    return MemoryStore(conn=FakeConn(cursor))


def shared_copy_rows():
    """Two project facts, each with an identical active copy in global."""
    index_rows = {
        "mem_api": [
            ("p1", "h1", None, None, None, "2026-10-02"),
            ("p2", "h2", None, None, None, "2026-10-01"),
        ],
        "memories": [
            ("g1", "h1", None, None, None, "2026-10-03"),
            ("g2", "h2", None, None, None, "2026-10-04"),
        ],
    }
    full_rows = {
        "mem_api": [
            {"id": "p1", "content": "project fact 1", "content_hash": "h1",
             "updated_at": "2026-10-02"},
            {"id": "p2", "content": "project fact 2", "content_hash": "h2",
             "updated_at": "2026-10-01"},
        ],
        "memories": [
            {"id": "g1", "content": "project fact 1", "content_hash": "h1",
             "updated_at": "2026-10-03"},
            {"id": "g2", "content": "project fact 2", "content_hash": "h2",
             "updated_at": "2026-10-04"},
        ],
    }
    return index_rows, full_rows


def test_browse_dedupes_cross_scope_copies_and_counts_each_scope():
    index_rows, full_rows = shared_copy_rows()
    store = make_store(FakeCursor(index_rows, full_rows))

    page = browse(store, BrowseRequest(project="api", limit=10))

    assert page.total == 2
    assert page.duplicates_hidden == 2
    assert page.total_by_scope == {"api": 2, "global": 2}
    assert page.listed_by_scope == {"api": 2}
    assert [row["id"] for row in page.results] == ["p1", "p2"]
    assert page.scopes_searched == ["api", "global"]
    assert any("identical content exists in another scope"
               in warning for warning in page.warnings)


def test_include_duplicates_lists_every_copy():
    index_rows, full_rows = shared_copy_rows()
    store = make_store(FakeCursor(index_rows, full_rows))

    page = browse(store, BrowseRequest(
        project="api", limit=10, include_duplicates=True))

    assert page.total == 4
    assert page.duplicates_hidden == 0
    assert page.listed_by_scope == {"api": 2, "global": 2}
    assert {row["id"] for row in page.results} == {"p1", "p2", "g1", "g2"}


def test_paging_reports_the_deduped_total_and_next_offset():
    index_rows, full_rows = shared_copy_rows()
    store = make_store(FakeCursor(index_rows, full_rows))

    page = browse(store, BrowseRequest(project="api", limit=1, offset=0))

    assert page.total == 2
    assert [row["id"] for row in page.results] == ["p1"]
    assert page.has_more is True
    assert page.next_offset == 1


def test_offset_beyond_the_result_set_names_the_total():
    index_rows, full_rows = shared_copy_rows()
    store = make_store(FakeCursor(index_rows, full_rows))

    page = browse(store, BrowseRequest(project="api", limit=10, offset=5))

    assert page.results == []
    assert page.total == 2
    assert any("offset=5 is beyond the 2 matching memories" in warning
               for warning in page.warnings)


def test_visibility_flags_are_part_of_the_single_index_query():
    index_rows, full_rows = shared_copy_rows()
    cursor = FakeCursor(index_rows, full_rows)
    store = make_store(cursor)

    browse(store, BrowseRequest(project="api", include_deleted=True,
                                include_expired=True))

    index_sql = next(sql for sql, _ in cursor.executed
                     if sql.startswith("SELECT id, content_hash"))
    where = index_sql.split("WHERE", 1)[1]
    assert "deleted_at IS NOT NULL OR superseded_by IS NULL" in where
    assert "expires_at > NOW()" not in where


def test_an_empty_filtered_browse_names_the_filter():
    store = make_store(FakeCursor({"mem_api": [], "memories": []},
                                  {"mem_api": [], "memories": []}))

    page = browse(store, BrowseRequest(project="api", entity="Nobody"))

    assert page.total == 0
    assert any("filter(s) entity narrowed the result to zero" in warning
               for warning in page.warnings)


def test_an_unknown_project_falls_back_to_global_with_a_warning():
    store = make_store(FakeCursor(
        {"memories": [("g1", "h1", None, None, None, "2026-10-01")]},
        {"memories": [{"id": "g1", "content": "shared fact",
                       "content_hash": "h1", "updated_at": "2026-10-01"}]},
        existing_tables=set()))
    store.cur.project_tables = []

    page = browse(store, BrowseRequest(project="ghost"))

    assert page.scopes_searched == ["global"]
    assert any("does not exist" in warning for warning in page.warnings)


def test_project_rows_marks_forgotten_and_expired_rows():
    cursor = FakeCursor(
        {"mem_api": [("p1", "h1", "__forgotten__", "2026-01-01",
                      "2020-01-01", "2026-10-01")]},
        {"mem_api": [{"id": "p1", "content": "old fact",
                      "content_hash": "h1", "superseded_by": "__forgotten__",
                      "deleted_at": "2026-01-01", "expires_at": "2020-01-01",
                      "updated_at": "2026-10-01"}]})
    store = make_store(cursor)

    page = browse(store, BrowseRequest(
        project="api", include_deleted=True, include_expired=True,
        include_superseded=True, limit=10,
        fields=["id", "content", "superseded_by", "forgotten", "expired"]))

    assert page.results[0]["forgotten"] is True
    assert page.results[0]["superseded_by"] is None
    assert page.results[0]["expired"] is True
    assert any("expired TTL" in warning for warning in page.warnings)


def test_the_cross_scope_dedupe_rule_lives_in_one_module():
    root = Path(__file__).resolve().parents[1] / "mcp_server"
    read_source = (root / "tools" / "read.py").read_text(encoding="utf-8")
    duplicates_source = (root / "duplicates.py").read_text(encoding="utf-8")
    browse_source = (root / "browse.py").read_text(encoding="utf-8")

    # Tool bodies must not carry their own copy-suppression predicate; they
    # call the one entry point in browse.py.
    assert "dedupe_across_scopes" not in read_source
    assert "is_cross_scope_duplicate" not in read_source
    assert "dedupe_across_scopes" in browse_source
    assert duplicates_source.count("def is_cross_scope_duplicate(") == 1
