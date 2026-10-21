"""Tests for the two tools that were missing for agent workflows:
unrelate (delete a single edge) and list_memories (browse without a query)."""

from mcp_server.server import (
    find_connections,
    list_memories,
    recall,
    relate,
    remember,
    unrelate,
)


def test_unrelate_removes_a_single_edge():
    relate("A", "B", "KNOWS")
    relate("A", "C", "LIKES")

    result = unrelate("A", "B", "KNOWS")

    assert result["ok"] is True
    assert result["status"] == "deleted"
    assert {r["entity"] for r in find_connections("A")["results"]} == {"C"}


def test_unrelate_reports_an_unknown_edge():
    result = unrelate("A", "Missing", "KNOWS")

    assert result["ok"] is False
    assert result["status"] == "not_found"
    assert "KNOWS" in result["error"]


def test_unrelate_only_touches_the_selected_scope():
    relate("A", "B", "KNOWS")
    relate("A", "B", "KNOWS", project="alpha")

    result = unrelate("A", "B", "KNOWS", project="alpha")

    assert result["ok"] is True
    assert find_connections("A")["results"]
    assert not find_connections("A", project="alpha")["results"]


def test_list_memories_browses_without_a_query():
    remember("Browse fact one", entity="Browser")
    remember("Browse fact two", entity="Browser")

    result = list_memories(entity="Browser")

    assert result["ok"] is True
    assert result["count"] == 2
    assert all(r["entity"] == "Browser" for r in result["results"])


def test_list_memories_can_filter_by_type_and_scope():
    remember("Browse typed fact", entity="Typed", mem_type="decision")
    remember("Browse other fact", entity="Typed", mem_type="note")

    result = list_memories(entity="Typed", mem_type="decision")

    assert result["count"] == 1
    assert result["results"][0]["mem_type"] == "decision"
