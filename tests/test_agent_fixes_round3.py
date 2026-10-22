"""Regression tests for the consistency and transparency fixes.

Every test here locks in a finding from the audit of the MCP surface:
a scope that does not exist must say so, `count` and `total` must agree,
the shared scope must be named where it is included, and a malformed
batch entry must not cost the caller the whole batch.
"""

from mcp_server.server import (
    contradictions_report,
    drop_project,
    find_paths,
    get_entity,
    init_context,
    list_entities,
    list_memories,
    recall,
    relate,
    relate_batch,
    remember,
    remember_batch,
)


# --- list_entities: count/total and the graph_node flag ------------------

def test_list_entities_reports_the_same_count_and_total():
    remember("Entity contract: a fact about Gamma", entity="Gamma")
    remember("Entity contract: a fact about Delta", entity="Delta")

    result = list_entities()

    assert result["ok"] is True
    assert result["count"] == len(result["results"])
    assert result["count"] == result["total"]


def test_list_entities_type_filter_keeps_count_and_total_in_sync():
    remember("Entity contract: a fact about Epsilon", entity="Epsilon")
    relate("Epsilon", "Zeta", "KNOWS")

    unfiltered = list_entities()
    graph_types = {row["type"] for row in unfiltered["results"]
                   if row["graph_node"]}
    assert graph_types

    for graph_type in graph_types:
        filtered = list_entities(entity_type=graph_type)
        assert filtered["count"] == filtered["total"], graph_type
        assert filtered["count"] == len(filtered["results"]), graph_type


def test_list_entities_never_reports_a_graph_node_as_memory_only():
    relate("NodeOnly", "Other", "KNOWS")
    remember("Entity contract: a fact about NodeOnly", entity="NodeOnly")

    rows = list_entities()["results"]
    node = next(r for r in rows if r["name"] == "NodeOnly")

    assert node["graph_node"] is True
    assert node["id"] is not None
    assert node["type"] != "memory-only"


def test_list_entities_does_not_turn_a_filtered_node_into_a_memory_only():
    relate("HiddenNode", "HiddenOther", "KNOWS")
    remember("Entity contract: a fact about HiddenNode", entity="HiddenNode")

    unfiltered = list_entities()["results"]
    node = next(r for r in unfiltered if r["name"] == "HiddenNode")
    other_type = "Entity" if node["type"] != "Entity" else "memory-only"

    filtered = list_entities(entity_type=other_type)["results"]
    assert not any(r["name"] == "HiddenNode" and r["graph_node"] is False
                   for r in filtered)


def test_list_entities_type_filter_selects_one_kind_of_entity():
    remember("Entity contract: a fact about OnlyMemory", entity="OnlyMemory")
    relate("GraphOnly", "GraphPartner", "KNOWS")

    only_memory = list_entities(entity_type="memory-only")
    assert only_memory["count"] == only_memory["total"]
    assert only_memory["count"] >= 1
    assert all(r["type"] == "memory-only" for r in only_memory["results"])
    assert any(r["name"] == "OnlyMemory" for r in only_memory["results"])

    only_graph = list_entities(entity_type="Entity")
    assert only_graph["count"] == only_graph["total"]
    assert all(r["graph_node"] is True for r in only_graph["results"])
    assert not any(r["name"] == "OnlyMemory" for r in only_graph["results"])


# --- init_context: the shared scope must be named ------------------------

def test_init_context_names_the_shared_scope_it_covered():
    remember("Bootstrap contract: a global fact")
    remember("Bootstrap contract: a project fact", project="beta")

    result = init_context()

    assert result["ok"] is True
    assert "global" in result["projects"]


def test_init_context_names_the_shared_scope_next_to_a_project():
    remember("Bootstrap contract: a project fact", project="beta")

    result = init_context(project="beta")

    assert set(result["projects"]) == {"beta", "global"}


def test_init_context_counts_the_graph_nodes_of_its_scope_only():
    relate("ScopeNode", "ScopeOther", "KNOWS", project="beta")
    remember("Bootstrap contract: a project fact", project="beta")

    scoped = init_context(project="beta", include_global=False)
    every = init_context(project="*")

    assert scoped["projects"] == ["beta"]
    assert scoped["graph_entities_count"] <= every["graph_entities_count"]


def test_init_context_reports_the_memory_breakdown():
    remember("Breakdown contract: an active fact", project="beta")
    remember("Breakdown contract: an old fact", project="beta")
    remember("Breakdown contract: the replacement", project="beta",
             supersede_content="Breakdown contract: an old fact")

    result = init_context(project="beta")

    assert result["active_memories"] >= 1
    assert result["superseded_memories"] >= 1
    assert result["total_memories"] == (
        result["active_memories"]
        + result["superseded_memories"]
        + result["forgotten_memories"]
    )


# --- a project that does not exist must say so --------------------------

def test_recall_warns_about_an_unknown_project():
    remember("Unknown project contract: a global fact")

    result = recall("unknown project contract", project="does-not-exist")

    assert result["ok"] is True
    assert any("does-not-exist" in w for w in result["warnings"])
    assert any("global" in w for w in result["warnings"])


def test_list_memories_warns_about_an_unknown_project():
    result = list_memories(project="does-not-exist")

    assert result["ok"] is True
    assert any("does-not-exist" in w for w in result["warnings"])


def test_list_entities_warns_about_an_unknown_project():
    result = list_entities(project="does-not-exist")

    assert result["ok"] is True
    assert any("does-not-exist" in w for w in result["warnings"])


def test_a_known_project_produces_no_scope_warning():
    remember("Known project contract: a project fact", project="beta")

    result = recall("known project contract", project="beta")

    assert not any("does not exist" in w for w in result["warnings"])


# --- recall across every scope ------------------------------------------

def test_recall_with_the_star_project_searches_every_scope():
    project_id = remember("Star contract: a project fact", project="beta")["id"]
    global_id = remember("Star contract: a global fact")["id"]

    result = recall("star contract", project="*", limit=20)

    ids = {row["id"] for row in result["results"]}
    assert project_id in ids
    assert global_id in ids
    assert result["count"] == len(result["results"])


def test_recall_with_the_star_project_reports_the_scopes_it_covered():
    remember("Star contract: a project fact", project="beta")

    result = recall("star contract", project="*", limit=20)

    assert result["ok"] is True
    assert result["count"] >= 1


# --- remember_batch: one malformed fact must not lose the batch ---------

def test_remember_batch_reports_a_missing_content_individually():
    result = remember_batch(facts=[
        {"content": "Batch contract: the first fact", "entity": "BatchA"},
        {"entity": "BatchB"},
    ], project="beta")

    assert result["ok"] is True
    assert result["status"] == "partial"
    assert result["results"][0]["status"] == "stored"
    assert result["results"][1]["status"] == "error"
    assert "content" in result["results"][1]["error"]

    stored = list_memories(project="beta", contains="the first fact")
    assert stored["count"] == 1


def test_remember_batch_reports_a_non_object_fact_individually():
    result = remember_batch(facts=[
        "not an object",
        {"content": "Batch contract: the second fact", "entity": "BatchC"},
    ], project="beta")

    assert result["ok"] is True
    assert result["status"] == "partial"
    assert result["results"][0]["status"] == "error"
    assert result["results"][1]["status"] == "stored"


# --- relation contexts must be reachable --------------------------------

def test_get_entity_shows_the_relation_context_of_its_edges():
    relation = relate("EdgeA", "EdgeB", "USES",
                      context="EdgeA uses EdgeB for the nightly batch",
                      project="beta")

    detail = get_entity("EdgeA", project="beta")

    ids = {m["id"] for m in detail["memories"]}
    assert relation["context_memory"] in ids


def test_the_relation_context_weight_follows_the_edge():
    relate("WeightA", "WeightB", "USES", context="WeightA uses WeightB",
           project="beta", weight=0.9)

    relate_batch(edges=[{"from_entity": "WeightA", "to_entity": "WeightB",
                         "relation_type": "USES", "weight": 0.2}],
                 project="beta")

    rows = list_memories(project="beta", contains="WeightA uses WeightB")
    assert rows["count"] == 1
    assert rows["results"][0]["metadata"]["weight"] == 0.2


# --- the scope boost must be visible ------------------------------------

def test_recall_publishes_the_scope_boost():
    remember("Boost contract: a project fact", project="beta")

    result = recall("boost contract", project="beta", limit=5)

    hit = next(r for r in result["results"] if r["content"].startswith("Boost"))
    # The boost is compressed towards 1 with the relevance, so it stays in
    # (0, 0.05] instead of saturating every strong hit at relevance 1.0.
    assert 0.0 < hit["_scope_boost"] <= 0.05
    assert hit["_relevance_score"] <= 1.0


# --- messages that must not mislead -------------------------------------

def test_drop_project_refuses_the_shared_scope_with_a_clear_message():
    result = drop_project(project="global")

    assert result["ok"] is False
    assert "global" in result["error"]
    assert "forget" in result["error"]


def test_find_paths_says_when_there_is_no_path():
    relate("PathA", "PathB", "KNOWS")
    relate("PathC", "PathD", "KNOWS")

    result = find_paths("PathA", "PathD")

    assert result["ok"] is True
    assert result["count"] == 0
    assert result["warnings"]
    assert any("no path" in w.lower() for w in result["warnings"])


def test_contradictions_report_says_when_there_is_nothing_to_compare():
    remember("Contradiction contract: only one fact", entity="Lonely")

    result = contradictions_report(entity="Lonely")

    assert result["ok"] is True
    assert result["count"] == 0
    assert result["warnings"]
    assert any("nothing to compare" in w for w in result["warnings"])
