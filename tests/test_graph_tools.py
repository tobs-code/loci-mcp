from mcp_server.server import (
    find_connections,
    find_paths,
    get_entity,
    relate,
    relate_batch,
    recall,
    remember,
)


def test_relate_creates_and_updates_an_edge():
    assert relate("A", "B", "KNOWS", weight=0.5) == "Created KNOWS from A -> B (w=0.5)"
    assert relate("A", "B", "KNOWS", weight=0.9) == "Updated KNOWS from A -> B (w=0.9)"

    connections = find_connections("A")
    assert len(connections) == 1
    assert connections[0] == {
        "entity": "B",
        "relation": "KNOWS",
        "weight": 0.9,
        "direction": "outgoing",
        "from_entity": "A",
        "to_entity": "B",
    }


def test_relate_rejects_missing_or_unsafe_relation_types():
    assert relate("A", "B", "") == "ERROR: relation_type is required"
    assert relate("A", "B", "BAD TYPE").startswith("ERROR:")


def test_relate_stores_an_optional_context_memory():
    result = relate("A", "B", "KNOWS", context="They met in Paris")

    assert result.startswith("Created KNOWS from A -> B")
    memory_id = result.split("context: ")[1]
    hits = recall("Paris", limit=5)
    hit = next(r for r in hits if r["id"] == memory_id)
    assert hit["entity"] == "A_KNOWS_B"
    assert hit["metadata"]["relation_type"] == "KNOWS"


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

    assert [r["entity"] for r in outgoing] == ["C", "B"]
    assert [r["entity"] for r in incoming] == ["D"]
    assert {r["entity"] for r in knows} == {"B", "D"}
    assert find_connections("A", direction="sideways") == [{
        "error": "direction must be 'outgoing', 'incoming' or 'both'"
    }]


def test_find_paths_returns_shortest_paths_and_respects_max_hops():
    relate("A", "B", "KNOWS")
    relate("B", "C", "KNOWS")

    result = find_paths("A", "C")
    assert result["count"] == 1
    path = result["paths"][0]
    assert [n["name"] for n in path["nodes"]] == ["A", "B", "C"]
    assert path["hops"] == 2

    assert find_paths("A", "C", max_hops=1)["count"] == 0
    assert find_paths("A", "Z")["count"] == 0


def test_relate_batch_keeps_projects_isolated_and_is_idempotent():
    results = relate_batch([
        {"from_entity": "A", "to_entity": "B", "relation_type": "KNOWS"},
        {"from_entity": "A", "to_entity": "C", "relation_type": "KNOWS",
         "project": "alpha", "weight": 0.5},
    ])

    assert [r["status"] for r in results] == ["created", "created"]
    assert [r["entity"] for r in find_connections("A")] == ["B"]
    assert [r["entity"] for r in find_connections("A", project="alpha")] == ["C"]

    again = relate_batch([
        {"from_entity": "A", "to_entity": "B", "relation_type": "KNOWS",
         "weight": 0.7},
    ])
    assert again[0]["status"] == "updated"
    assert find_connections("A")[0]["weight"] == 0.7
