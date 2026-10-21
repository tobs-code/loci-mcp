"""Contract tests for the read tools (recall, get_entity, list_entities,
init_context): uniform envelope, slim default payload, field selection."""

from mcp_server.server import (
    get_entity,
    init_context,
    list_entities,
    recall,
    remember,
    relate,
    list_memories,
)


def _hit(result, memory_id):
    return next(r for r in result["results"] if r["id"] == memory_id)


def test_recall_answers_with_the_envelope_and_a_result_list():
    remember("Read contract: a yellow kayak in the harbor")

    result = recall("yellow kayak harbor", limit=5)

    assert result["ok"] is True
    assert result["status"] == "ok"
    assert result["error"] is None
    assert result["count"] == len(result["results"])
    assert result["count"] >= 1


def test_recall_hides_internal_fields_by_default():
    memory_id = remember("Read contract: slim rows have no content hash")["id"]

    result = recall("slim rows content hash", limit=5)

    hit = _hit(result, memory_id)
    assert "content_hash" not in hit
    assert "deleted_at" not in hit
    assert hit["content"]


def test_recall_can_return_only_the_requested_fields():
    memory_id = remember("Read contract: field selection wins")["id"]

    result = recall("field selection wins", limit=5, fields=["id", "content"])

    hit = _hit(result, memory_id)
    assert set(hit) == {"id", "content"}


def test_recall_can_return_the_original_project_name():
    memory_id = remember("Read contract: project names are preserved",
                         project="Team Alpha")["id"]

    result = recall("project names are preserved", project="Team Alpha",
                    limit=5, fields=["id", "project"])

    hit = _hit(result, memory_id)
    assert hit == {"id": memory_id, "project": "Team Alpha"}


def test_memory_projection_accepts_internal_project_alias():
    memory_id = remember("Read contract: internal project alias",
                         project="projection-alias")["id"]

    result = recall("internal project alias", project="projection-alias",
                    limit=5, fields=["id", "_project"])

    assert _hit(result, memory_id) == {
        "id": memory_id,
        "_project": "projection-alias",
    }


def test_project_recall_prioritizes_project_hits_over_global_ties():
    remember("Scope ranking: deployment uses blue green rollout", project="scope-rank")
    remember("Scope ranking: deployment requires signed release", scope="global")

    result = recall("deployment rollout", project="scope-rank", limit=5)

    assert result["results"][0]["scope"] == "project"


def test_project_recall_can_exclude_global_hits():
    remember("Scope filter: project-only deployment fact", project="scope-filter")
    remember("Scope filter: global deployment fact", scope="global")

    result = recall("deployment fact", project="scope-filter",
                    include_global=False, limit=10)

    assert result["ok"] is True
    assert all(hit["scope"] == "project" for hit in result["results"])


def test_project_browse_and_context_can_exclude_global_hits():
    remember("Scope browse: project fact", project="scope-browse")
    remember("Scope browse: global fact", scope="global")

    browsed = list_memories(project="scope-browse", include_global=False)
    context = init_context(project="scope-browse", include_global=False)

    assert all(hit["scope"] == "project" for hit in browsed["results"])
    assert all(hit.get("scope") == "project"
               for hit in context["recent_memories"])


def test_project_entity_lookup_can_exclude_global_hits():
    remember("Entity scope: project deployment owner", entity="ScopeOwner",
             project="entity-scope")
    remember("Entity scope: global deployment owner", entity="ScopeOwner",
             scope="global")

    result = get_entity(entity="ScopeOwner", project="entity-scope",
                        include_global=False)

    assert result["ok"] is True
    assert all(memory["scope"] == "project" for memory in result["memories"])


def test_recall_warns_when_graph_boost_has_no_entity():
    remember("Read contract: graph boost without entity")

    result = recall("graph boost without entity", limit=5, graph_boost=True)

    assert result["ok"] is True
    assert any("entity" in w for w in result["warnings"])


def test_recall_can_return_only_superseded_rows():
    old_id = remember("Read contract: the queue runs on port 7000")["id"]
    new_id = remember("Read contract: the queue runs on port 7100",
                      supersede=old_id)["id"]

    result = recall("queue runs on port", limit=10, only_superseded=True)

    ids = {r["id"] for r in result["results"]}
    assert old_id in ids
    assert new_id not in ids


def test_get_entity_answers_with_the_envelope():
    memory_id = remember("Read contract: Nina reviews the dashboard",
                         entity="Nina")["id"]

    result = get_entity(entity="Nina")

    assert result["ok"] is True
    assert result["status"] == "ok"
    assert result["entity"] == "Nina"
    assert [m["id"] for m in result["memories"]] == [memory_id]


def test_get_entity_can_limit_the_memory_fields():
    memory_id = remember("Read contract: field selection for entities",
                         entity="Otto")["id"]

    result = get_entity(entity="Otto", fields=["id", "content"])

    memory = next(m for m in result["memories"] if m["id"] == memory_id)
    assert set(memory) == {"id", "content"}


def test_get_entity_returns_details_for_graph_only_entities():
    relate("GraphOnlyA", "GraphOnlyB", "KNOWS")

    result = get_entity(entity="GraphOnlyA")

    assert result["ok"] is True
    assert result["details"]["name"] == "GraphOnlyA"
    assert result["details"]["connections"][0]["entity"] == "GraphOnlyB"


def test_relation_context_is_searchable_but_hidden_from_memory_browse():
    relation = relate("BrowseA", "BrowseB", "KNOWS", context="Met in Berlin")

    assert relation["context_memory"]
    assert any(r["id"] == relation["context_memory"]
               for r in recall("Met in Berlin")["results"])
    assert any(r["id"] == relation["context_memory"]
               for r in list_memories(limit=20)["results"])


def test_read_tools_reject_unknown_fields():
    assert recall("unknown field", fields=["bogus"])["ok"] is False
    assert list_memories(fields=["bogus"])["ok"] is False
    assert get_entity(entity="Nina", fields=["bogus"])["ok"] is False


def test_list_entities_answers_with_the_envelope():
    remember("Read contract: Uma writes the release notes", entity="Uma")

    result = list_entities()

    assert result["ok"] is True
    assert result["count"] == len(result["results"])
    assert any(r["name"] == "Uma" for r in result["results"])


def test_list_entities_rejects_non_positive_limits():
    result = list_entities(limit=0)
    assert result["ok"] is False
    assert "limit" in result["error"]


def test_list_entities_can_exclude_global_entities():
    remember("Read contract: project entity scope", entity="ProjectOnly",
             project="entity-list-scope")
    remember("Read contract: global entity scope", entity="GlobalOnly",
             scope="global")

    result = list_entities(project="entity-list-scope", include_global=False)

    assert result["ok"] is True
    assert all(row["project"] == "entity-list-scope"
               for row in result["results"])
    assert any(row["name"] == "ProjectOnly" for row in result["results"])


def test_init_context_slim_bootstrap_drops_internal_fields():
    remember("Read contract: bootstrap fact for alpha", project="alpha")

    result = init_context(project="alpha")

    assert result["ok"] is True
    assert result["status"] == "ok"
    memory = result["recent_memories"][0]
    assert set(memory) <= {"id", "content", "entity", "mem_type", "importance",
                            "updated_at", "scope", "project"}
    assert memory["project"] == "alpha"


def test_init_context_removes_near_duplicates():
    remember("Read contract: the deploy pipeline runs on monday",
             entity="Deploy")
    remember("Read contract: the deploy pipeline runs on Mondays",
             entity="Deploy")

    result = init_context()

    pipeline = [m for m in result["recent_memories"]
                if "deploy pipeline" in m["content"].lower()]
    assert len(pipeline) == 1


def test_init_context_defaults_to_global_and_can_select_all_projects():
    remember("Bootstrap scope: project-only marker", project="bootstrap-scope")

    default_context = init_context()
    all_context = init_context(project="*")

    assert all(m.get("scope") == "global"
               for m in default_context["recent_memories"])
    assert any(m.get("scope") == "project" and
               "project-only marker" in m["content"]
               for m in all_context["recent_memories"])


def test_init_context_detail_full_keeps_the_internal_fields():
    remember("Read contract: full detail bootstrap", project="beta")

    result = init_context(project="beta", detail="full")

    memory = result["recent_memories"][0]
    assert "access_count" in memory
    assert memory["project"] == "beta"
    # Internal bookkeeping is never exposed, not even in the full detail.
    assert "content_hash" not in memory
    assert "embedding" not in memory
    assert "deleted_at" not in memory
