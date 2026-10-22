"""Contract tests for the graph expansion of `recall`.

Three regressions are covered here:
- a graph hit must report the project name, not the scope kind ("project")
- `graph_entity` expands the graph without filtering the query search
- `import_memories` accepts a payload whose arrays were packed into objects
  (some MCP clients serialize a JSON array as {"item": ...})
"""

from mcp_server.server import (
    export_memories,
    import_memories,
    recall,
    remember,
    relate,
)


def test_graph_hit_reports_the_project_name():
    remember("Graph scope: the indexer runs at midnight",
             entity="Indexer", project="graph-scope")
    remember("Graph scope: the scheduler restarts nightly",
             entity="Scheduler", project="graph-scope")
    relate("Indexer", "Scheduler", "TRIGGERS", project="graph-scope")

    result = recall("indexer runs at midnight", project="graph-scope",
                    graph_boost=True, graph_entity="Indexer", limit=10)

    hits = [r for r in result["results"] if r.get("_graph_boost")]
    assert hits, "expected a graph hit next to the query hit"
    for hit in hits:
        assert hit["project"] == "graph-scope"


def test_graph_entity_keeps_unrelated_query_hits():
    remember("Graph scope: unrelated fact about coffee", entity="Coffee",
             project="graph-scope-2")
    remember("Graph scope: the indexer runs at midnight", entity="Indexer",
             project="graph-scope-2")
    relate("Indexer", "Scheduler", "TRIGGERS", project="graph-scope-2")

    result = recall("coffee", project="graph-scope-2", graph_boost=True,
                    graph_entity="Indexer", limit=10)

    contents = [r["content"] for r in result["results"]]
    assert any("coffee" in c for c in contents), \
        "the query search must not be filtered by the expansion entity"


def test_import_accepts_a_packed_array_payload():
    remember("Graph scope: a fact to export", project="graph-scope-3")
    payload = export_memories(format="json", project="graph-scope-3")
    records = payload["data"]["records"]
    packed = {
        "type": "memory_export",
        "version": "0.2",
        "records": {"item": records[0]} if len(records) == 1 else {"item": records},
    }

    result = import_memories(packed, project="graph-scope-3")

    assert result["ok"] is True
    assert result["error"] is None
    assert result["imported"] + result["skipped"] == len(records)
