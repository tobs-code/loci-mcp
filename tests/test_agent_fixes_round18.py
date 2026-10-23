"""Regression tests for the eighteenth round of fixes.

Covers:
- a near-duplicate across two different entities is never superseded
  automatically (the wording may match, the subject does not);
- `supersede_last` without an entity refuses to replace a tagged entity's
  memory;
- `list_memories` names the scopes it searched and splits the counts;
- `get_entity` reports when a name exists in more than one scope;
- re-relating an edge with the same context reuses the context memory;
- `export_memories(include_expired=false)` excludes expired rows;
- `recall(detail="slim")` returns a token-cheap projection;
- `forget(entity=...)` without a project names the scopes it touched.
"""

from mcp_server.server import (
    export_memories,
    forget,
    get_entity,
    list_memories,
    recall,
    relate,
    remember,
)


def test_near_duplicate_across_entities_is_not_auto_superseded():
    project = "round18-near-entity"
    old = remember(
        "Round18 deployment uses Kubernetes for the staging cluster.",
        entity="Staging18", project=project)
    assert old["ok"] is True

    new = remember(
        "Round18 deployment uses Kubernetes for the production cluster.",
        entity="Prod18", project=project,
        on_near_duplicate="supersede")

    assert new["ok"] is True
    assert new["status"] == "stored"
    assert any("near-duplicate across entities" in w
               for w in new["warnings"]), new["warnings"]

    stored = list_memories(memory_id=old["id"], project=project,
                           include_superseded=True)["memory"]
    assert stored["superseded_by"] is None


def test_supersede_last_without_entity_does_not_replace_tagged_memory():
    project = "round18-supersede-last"
    old = remember(
        "Round18 cache TTL is 300 seconds.",
        entity="Cache18", project=project)

    new = remember(
        "Round18 cache TTL is 600 seconds.",
        project=project, supersede_last=True)

    assert new["ok"] is True
    assert new["status"] == "stored"
    assert any("belongs to entity 'Cache18'" in w
               for w in new["warnings"]), new["warnings"]

    stored = list_memories(memory_id=old["id"], project=project,
                           include_superseded=True)["memory"]
    assert stored["superseded_by"] is None


def test_list_memories_reports_scopes_and_per_scope_counts():
    project = "round18-scopes"
    remember("Round18 project scope fact", project=project)
    remember("Round18 global scope fact", scope="global")

    result = list_memories(project=project)
    assert result["ok"] is True
    assert set(result["scopes_searched"]) == {project, "global"}
    assert result["total_by_scope"][project] == 1
    assert result["total_by_scope"]["global"] == 1
    assert result["listed_by_scope"][project] == 1
    assert result["listed_by_scope"]["global"] == 1


def test_get_entity_reports_scope_conflicts():
    project = "round18-entity-scopes"
    entity = "Round18SharedEntity"
    remember("Round18 global entity fact", entity=entity, scope="global")
    remember("Round18 project entity fact", entity=entity, project=project)

    result = get_entity(entity, project=project)
    assert result["ok"] is True
    assert result["details"]["scopes_with_entity"] == {
        "global": "memory", project: "memory"}
    assert any("exists in several scopes" in w for w in result["warnings"])
    assert set(result["scopes_searched"]) == {project, "global"}
    assert result["memories_by_scope"][project] == 1
    assert result["memories_by_scope"]["global"] == 1


def test_relate_reuses_identical_context_memory():
    project = "round18-relate-context"
    first = relate("Round18A", "Round18B", "uses",
                   context="Round18A uses Round18B.", project=project)
    second = relate("Round18A", "Round18B", "uses",
                    context="Round18A uses Round18B.", project=project)

    assert first["ok"] is True and second["ok"] is True
    assert second["status"] == "updated"
    assert second["context_reused"] is True
    assert second["context_memory"] == first["context_memory"]


def test_export_can_exclude_expired_memories():
    project = "round18-export-expired"
    remember("Round18 expired export probe", project=project,
             expires_at="2020-01-01")

    without = export_memories(project=project, include_expired=False)
    assert without["ok"] is True
    assert without["count"] == 0
    assert any("include_expired=false" in w for w in without["warnings"])

    with_expired = export_memories(project=project, include_expired=True)
    assert with_expired["count"] == 1


def test_recall_slim_projection_omits_metadata():
    project = "round18-slim"
    remember("Round18 slim recall token", project=project,
             metadata={"source": "round18"})

    result = recall("Round18 slim recall token", project=project,
                    detail="slim")
    assert result["ok"] is True
    assert result["detail"] == "slim"
    assert result["results"]
    hit = result["results"][0]
    assert "content" in hit and "id" in hit
    assert "metadata" not in hit
    assert "_similarity" in hit


def test_forget_entity_names_every_scope_it_touched():
    project = "round18-forget-scopes"
    entity = "Round18ForgetEntity"
    remember("Round18 forget global fact", entity=entity, scope="global")
    remember("Round18 forget project fact", entity=entity, project=project)

    result = forget(entity=entity, soft=True)
    assert result["ok"] is True
    assert set(result["scopes_affected"]) == {project, "global"}
    assert any("touched multiple scopes" in w for w in result["warnings"])
