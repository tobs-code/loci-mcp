from mcp_server.server import (
    find_connections,
    find_paths,
    forget,
    get_entity,
    relate,
    relate_batch,
    recall,
    remember,
    unrelate,
)


def test_relate_creates_and_updates_an_edge():
    created = relate("A", "B", "KNOWS", weight=0.5)
    assert created["status"] == "created"
    updated = relate("A", "B", "KNOWS", weight=0.9)
    assert updated["status"] == "updated"
    assert updated["weight"] == 0.9

    connections = find_connections("A")
    assert len(connections["results"]) == 1
    assert connections["results"][0] == {
        "entity": "B",
        "relation": "KNOWS",
        "weight": 0.9,
        "direction": "outgoing",
        "from_entity": "A",
        "to_entity": "B",
    }


def test_relate_rejects_missing_or_unsafe_relation_types():
    assert relate("A", "B", "")["ok"] is False
    assert relate("A", "B", "BAD TYPE")["ok"] is False


def test_relate_stores_an_optional_context_memory():
    result = relate("A", "B", "KNOWS", context="They met in Paris")

    assert result["status"] == "created"
    memory_id = result["context_memory"]
    hits = recall("Paris", limit=5)
    hit = next(r for r in hits["results"] if r["id"] == memory_id)
    assert hit["entity"] == "A_KNOWS_B"
    assert hit["metadata"]["relation_type"] == "KNOWS"


def test_unrelate_removes_the_relation_context_memory():
    relation = relate("ContextA", "ContextB", "KNOWS", context="Met in Berlin")
    context_id = relation["context_memory"]

    assert unrelate("ContextA", "ContextB", "KNOWS")["status"] == "deleted"
    result = recall("Met in Berlin", include_superseded=True, min_similarity=0.0)

    assert all(memory["id"] != context_id for memory in result["results"])


def test_forget_entity_removes_relation_context_memories():
    relation = relate("ForgetA", "ForgetB", "KNOWS", context="Temporary link")
    context_id = relation["context_memory"]

    result = forget(entity="ForgetA", soft=False)

    assert result["status"] == "deleted"
    memories = recall("Temporary link", include_superseded=True, min_similarity=0.0)
    assert all(memory["id"] != context_id for memory in memories["results"])


def test_get_entity_exposes_graph_connections():
    remember("Carol leads the team", entity="Carol")
    relate("Carol", "Dave", "LEADS")

    result = get_entity("Carol")

    assert result["details"]["name"] == "Carol"
    assert result["details"]["connections"] == [{
        "entity": "Dave", "relation": "LEADS", "weight": 1.0,
        "direction": "outgoing",
    }]


def test_find_connections_filters_by_direction_and_relation_type():
    relate("A", "B", "KNOWS", weight=0.4)
    relate("A", "C", "LIKES", weight=0.8)
    relate("D", "A", "KNOWS", weight=0.6)

    outgoing = find_connections("A", direction="outgoing")
    incoming = find_connections("A", direction="incoming")
    knows = find_connections("A", relation_type="KNOWS")

    assert [r["entity"] for r in outgoing["results"]] == ["C", "B"]
    assert [r["entity"] for r in incoming["results"]] == ["D"]
    assert {r["entity"] for r in knows["results"]} == {"B", "D"}
    invalid = find_connections("A", direction="sideways")
    assert invalid["ok"] is False
    assert "direction" in invalid["error"]


def test_graph_tools_reject_non_positive_limits():
    assert find_connections("A", limit=0)["ok"] is False
    assert "limit" in find_connections("A", limit=0)["error"]
    assert find_paths("A", "B", max_hops=0)["ok"] is False
    assert "max_hops" in find_paths("A", "B", max_hops=0)["error"]


def test_find_connections_reports_unknown_entities_as_not_found():
    result = find_connections("Unknown")

    assert result["ok"] is False
    assert result["status"] == "not_found"
    assert result["entity"] == "Unknown"


def test_find_paths_rejects_non_integer_hops():
    result = find_paths("A", "B", max_hops="many")
    assert result["ok"] is False
    assert "max_hops" in result["error"]


def test_find_paths_returns_shortest_paths_and_respects_max_hops():
    relate("A", "B", "KNOWS")
    relate("B", "C", "KNOWS")

    result = find_paths("A", "C")
    assert result["count"] == 1
    assert "paths" not in result
    path = result["results"][0]
    assert [n["name"] for n in path["nodes"]] == ["A", "B", "C"]
    assert path["hops"] == 2

    assert find_paths("A", "C", max_hops=1)["count"] == 0
    missing = find_paths("A", "Z")
    assert missing["ok"] is False
    assert missing["status"] == "not_found"
    assert missing["entity"] == "Z"


def test_relate_batch_keeps_projects_isolated_and_is_idempotent():
    result = relate_batch([
        {"from_entity": "A", "to_entity": "B", "relation_type": "KNOWS"},
        {"from_entity": "A", "to_entity": "C", "relation_type": "KNOWS",
         "project": "alpha", "weight": 0.5},
    ])

    assert [r["status"] for r in result["results"]] == ["created", "created"]
    assert [r["entity"] for r in find_connections("A")["results"]] == ["B"]
    assert [r["entity"] for r in find_connections("A", project="alpha")["results"]] == ["C"]

    again = relate_batch([
        {"from_entity": "A", "to_entity": "B", "relation_type": "KNOWS",
         "weight": 0.7},
    ])
    assert again["results"][0]["status"] == "updated"
    assert find_connections("A")["results"][0]["weight"] == 0.7


def test_find_connections_hint_mentions_the_searched_project():
    result = find_connections("Unknown", project="hint-scope")

    assert result["ok"] is False
    assert "hint-scope" in result["error"]
    assert "without project" not in result["error"]
