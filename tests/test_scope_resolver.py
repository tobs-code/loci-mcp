"""The scope rules are resolved once, not re-derived per read tool."""

import pytest

from mcp_server.db import get_conn
from mcp_server.scope import (
    MODE_BOOTSTRAP,
    MODE_BROWSE,
    MODE_LOOKUP,
    MODE_PROJECT,
    resolve_scope,
)
from mcp_server.scope_catalog import _mem_table
from mcp_server.server import remember


PROJECT = "scope-resolver-probe"


@pytest.fixture
def cur():
    conn = get_conn()
    try:
        yield conn.cursor()
    finally:
        conn.close()


@pytest.fixture
def project_table():
    remember("Scope resolver project fact", project=PROJECT)
    return _mem_table(PROJECT)


def test_project_mode_reads_project_then_global(project_table, cur):
    scope = resolve_scope(PROJECT)

    tables = scope.tables(cur, MODE_PROJECT)

    assert tables == [project_table, "memories"]
    assert scope.project == PROJECT
    assert scope.explicit is True
    assert scope.all_projects is False


def test_star_covers_every_scope_even_in_project_mode(project_table, cur):
    scope = resolve_scope("*")

    tables = scope.tables(cur, MODE_PROJECT)

    assert project_table in tables
    assert "memories" in tables


def test_include_global_false_drops_the_shared_table(project_table, cur):
    scope = resolve_scope(PROJECT, include_global=False)

    tables = scope.tables(cur, MODE_PROJECT)

    assert tables == [project_table]


def test_browse_without_project_covers_every_scope(project_table, cur):
    scope = resolve_scope(None)

    tables = scope.tables(cur, MODE_BROWSE)

    assert project_table in tables
    assert "memories" in tables


def test_explicit_global_covers_only_the_shared_table(project_table, cur):
    scope = resolve_scope("global")

    tables = scope.tables(cur, MODE_BROWSE)

    assert tables == ["memories"]
    assert scope.global_only is True
    assert scope.project is None


def test_bootstrap_without_project_reads_only_global(project_table, cur):
    scope = resolve_scope(None)

    tables = scope.tables(cur, MODE_BOOTSTRAP)

    assert tables == ["memories"]


def test_lookup_without_project_searches_every_scope(project_table, cur):
    scope = resolve_scope(None)

    tables = scope.tables(cur, MODE_LOOKUP)

    assert project_table in tables
    assert "memories" in tables


def test_unknown_project_warns_and_falls_back_to_global(cur):
    scope = resolve_scope("scope-resolver-does-not-exist")

    tables = scope.tables(cur, MODE_PROJECT)
    warning = scope.unknown_warning(cur)

    assert tables == ["memories"]
    assert warning is not None
    assert "does not exist" in warning


def test_graph_filter_follows_the_selection():
    assert resolve_scope(PROJECT).graph_filter() == (
        "(project = %s OR project = 'global')", [PROJECT])
    assert resolve_scope(PROJECT, include_global=False).graph_filter() == (
        "project = %s", [PROJECT])
    assert resolve_scope("global").graph_filter() == (
        "project = 'global'", [])
    assert resolve_scope("*", include_global=False).graph_filter() == (
        "project <> 'global'", [])
    assert resolve_scope("*").graph_filter() == ("TRUE", [])
