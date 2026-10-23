"""The internal search interface is a named SearchRequest."""

import inspect

from mcp_server.search import SearchRequest, _search_memories
from mcp_server.server import recall, remember


def test_search_request_carries_named_intent():
    request = SearchRequest(
        query_embedding=None,
        query_text="deploy steps",
        mem_type="runbook",
        entity="Deploy",
        limit=3,
        project="api",
        include_decay=True,
        min_similarity=None,
        hybrid=False,
    )

    assert request.mem_type == "runbook"
    assert request.entity == "Deploy"
    assert request.limit == 3
    assert request.min_similarity is None
    assert request.include_deleted is False


def test_search_memories_takes_a_request_not_positional_booleans():
    parameters = inspect.signature(_search_memories).parameters

    assert list(parameters) == ["cur", "request"]


def test_graph_boost_failure_keeps_query_hits(monkeypatch):
    project = "search-request-graph-failure"
    hub = "SearchRequestHub"
    remember("SearchRequest deploy runbook fact", project=project, entity=hub)

    def _boom(cur, *args, **kwargs):
        cur.execute("SELECT 1 / 0")

    monkeypatch.setattr("mcp_server.search.boost_neighbors", _boom)

    result = recall("SearchRequest deploy runbook fact", project=project,
                    entity=hub, graph_boost=True)

    assert result["ok"] is True
    assert result["results"]
