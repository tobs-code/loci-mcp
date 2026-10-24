"""Scope naming is pure, lookup reads only, and lifecycle counts agree.

The split exists so a read can never create a table and so the superseded
count has exactly one definition. These tests pin both promises.
"""

import hashlib

import pytest

from mcp_server.db import get_conn
from mcp_server.lifecycle import counts as lifecycle_counts
from mcp_server.lifecycle import superseded_marker_sql
from mcp_server.schema_migrator import ensure_project_table
from mcp_server.scope_catalog import (
    _all_mem_tables,
    _is_global_scope,
    _mem_table,
    _norm_entity,
    _norm_project,
    _project_names,
    _project_slug,
    _retrieval_tables,
    _safe_ident,
    _scope_key,
    _scope_label,
    _scope_of_table,
    _table_exists,
    _unknown_project_warning,
)
from mcp_server.server import forget, remember

PROJECT = "catalog-probe"


def _derived_slug(project: str, cleaned: str) -> str:
    digest = hashlib.sha1(project.encode()).hexdigest()[:8]
    return f"{cleaned[:40]}_{digest}"


@pytest.fixture
def cur():
    conn = get_conn()
    try:
        yield conn.cursor()
    finally:
        conn.close()


# -- naming ---------------------------------------------------------------

@pytest.mark.parametrize("project,expected", [
    (None, None),
    ("global", None),
    ("  global  ", None),
    ("api", "api"),
    ("my-project", "my-project"),
    ("Team", "Team"),
])
def test_norm_project_keeps_the_caller_name(project, expected):
    assert _norm_project(project) == expected


@pytest.mark.parametrize("project", ["", "   ", "\t"])
def test_norm_project_refuses_a_blank_name(project):
    with pytest.raises(ValueError, match="project must not be"):
        _norm_project(project)


def test_norm_entity_trims_and_refuses_blank():
    assert _norm_entity("  API  ") == "API"
    assert _norm_entity(None) is None
    with pytest.raises(ValueError, match="entity must not be"):
        _norm_entity("  ")


@pytest.mark.parametrize("project,slug", [
    (None, "memories"),
    ("global", "memories"),
    ("api", "mem_api"),
    ("API", "mem_" + _derived_slug("API", "api")),
    ("Team", "mem_" + _derived_slug("Team", "team")),
    ("my project", "mem_" + _derived_slug("my project", "my_project")),
    ("1st", "mem_" + _derived_slug("1st", "p_1st")),
    ("a" * 60, "mem_" + _derived_slug("a" * 60, "a" * 60)),
])
def test_table_names_are_stable_and_sql_safe(project, slug):
    assert _mem_table(project) == slug


def test_scope_key_and_label_follow_the_naming():
    assert _scope_key(None) == "global"
    assert _scope_key("api") == "api"
    assert _scope_label(None) == "Entity"
    assert _scope_label("api") == "Project_api"
    assert _is_global_scope(" global ") is True
    assert _is_global_scope(None) is False


def test_project_slug_refuses_an_empty_project():
    with pytest.raises(ValueError, match="project must not be"):
        _project_slug("   ")


@pytest.mark.parametrize("value", ["1st", "a-b", "a b", "", "a.b"])
def test_safe_ident_rejects_unsafe_identifiers(value):
    with pytest.raises(ValueError, match="invalid relation_type"):
        _safe_ident(value, "relation_type")


def test_safe_ident_accepts_plain_identifiers():
    assert _safe_ident("Project_api_1", "relation_type") == "Project_api_1"


# -- lookup has no DDL ----------------------------------------------------

class RecordingCursor:
    """A cursor stand-in that records SQL and serves canned rows."""

    def __init__(self, one_rows=None):
        self.statements: list[str] = []
        self.one_rows = list(one_rows or [])

    def execute(self, sql, params=()):
        self.statements.append(sql.strip())

    def fetchone(self):
        return self.one_rows.pop(0) if self.one_rows else None

    def fetchall(self):
        return []


def test_lookup_reads_only_and_never_creates():
    cur = RecordingCursor([(False,), (False,), (0,), (0,)])

    assert _retrieval_tables(cur, "api") == ["memories"]
    assert _table_exists(cur, "mem_api") is False
    assert _all_mem_tables(cur) == ["memories"]
    assert _project_names(cur) == {}
    assert _scope_of_table(cur, "mem_api") == "api"
    assert _unknown_project_warning(cur, "api") is not None

    forbidden = ("CREATE", "ALTER", "DROP", "INSERT", "UPDATE", "DELETE")
    for statement in cur.statements:
        assert statement.upper().startswith("SELECT"), statement
        assert not any(word in statement.upper() for word in forbidden)


# -- migration ------------------------------------------------------------

def test_ensure_project_table_is_idempotent(cur):
    ensure_project_table(cur, PROJECT)
    ensure_project_table(cur, PROJECT)

    table = _mem_table(PROJECT)
    cur.execute("SELECT to_regclass(%s)", (f"public.{table}",))
    assert cur.fetchone()[0] is not None

    cur.execute("SELECT COUNT(*) FROM projects WHERE name = %s", (PROJECT,))
    assert cur.fetchone()[0] == 1

    cur.execute(
        "SELECT COUNT(*) FROM pg_indexes WHERE schemaname = 'public' "
        "AND tablename = %s AND indexdef LIKE '%%to_tsvector%%'",
        (table,),
    )
    assert cur.fetchone()[0] == 1


def test_lifecycle_supersede_count_matches_the_lifecycle_rule(cur):
    first = remember("catalog probe original", project=PROJECT)
    remember("catalog probe replacement", project=PROJECT,
             supersede=first["id"])
    forgotten = remember("catalog probe forgotten", project=PROJECT)
    forget(memory_id=forgotten["id"], project=PROJECT)

    counts = lifecycle_counts(cur, _mem_table(PROJECT))
    assert counts["superseded"] == 1
    assert counts["forgotten"] == 1

    cur.execute(
        f"SELECT COUNT(*) FROM {_mem_table(PROJECT)} "
        f"WHERE {superseded_marker_sql()}"
    )
    assert cur.fetchone()[0] == counts["superseded"]

    # The forgotten row carries the sentinel in `superseded_by`; a raw
    # IS NOT NULL count would report it as superseded history.
    cur.execute(
        f"SELECT COUNT(*) FROM {_mem_table(PROJECT)} "
        "WHERE superseded_by IS NOT NULL"
    )
    assert cur.fetchone()[0] == counts["superseded"] + counts["forgotten"]


def test_superseded_marker_supports_a_table_alias():
    assert superseded_marker_sql("m").startswith("m.superseded_by IS NOT NULL")
