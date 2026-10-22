"""Regression tests for the audit fixes (round 4).

Every test here locks in a finding from the audit of the MCP surface:

- a write reports the scope it really used, so a later read or delete
  addresses the right scope;
- `supersede` and `supersede_content` never claim a replacement that did
  not happen;
- `dry_run` counts exactly what the real call removes;
- `project="global"` is the shared scope, not "every scope";
- a scope that only exists in the graph is listed and can be dropped;
- `get_entity` pages instead of truncating silently;
- an empty result names every filter that removed something;
- a broken metadata value in an import is reported per record.
"""

import json

from mcp_server.server import (
    drop_project,
    forget,
    get_entity,
    import_memories,
    list_entities,
    list_memories,
    list_projects,
    recall,
    relate,
    remember,
    remember_batch,
)


# --- remember reports the scope it really used ---------------------------

def test_remember_reports_the_scope_it_wrote_to():
    result = remember("Scope report: a project fact", entity="ScopeReport",
                      project="scope-report")

    assert result["scope"] == "project"
    assert result["project"] == "scope-report"


def test_remember_reports_the_shared_scope_when_scope_is_global():
    result = remember("Scope report: a global fact", entity="ScopeReport",
                      project="scope-report", scope="global")

    assert result["scope"] == "global"
    assert result["project"] == "global"


def test_remember_batch_reports_the_scope_per_fact():
    result = remember_batch(facts=[
        {"content": "Scope report: batch project fact", "entity": "ScopeReport"},
        {"content": "Scope report: batch global fact", "entity": "ScopeReport",
         "scope": "global"},
    ], project="scope-report")

    assert result["ok"] is True
    scopes = [entry["scope"] for entry in result["results"]]
    projects = [entry["project"] for entry in result["results"]]
    assert scopes == ["project", "global"]
    assert projects == ["scope-report", "global"]


# --- supersede never claims a replacement that did not happen ------------

def test_supersede_with_an_unknown_id_warns_instead_of_claiming_success():
    result = remember("Supersede report: an unreplaced fact",
                      entity="SupersedeReport", project="supersede-report",
                      supersede="ffffffffffffffff")

    assert result["superseded"] == []
    assert any("supersede skipped" in w for w in result["warnings"]), result


def test_supersede_content_without_a_match_warns():
    result = remember("Supersede report: another unreplaced fact",
                      entity="SupersedeReport", project="supersede-report",
                      supersede_content="nothing in this scope says this")

    assert result["superseded"] == []
    assert any("supersede_content skipped" in w for w in result["warnings"]), result


def test_supersede_with_a_known_id_still_replaces():
    old = remember("Supersede report: the old version", entity="SupersedeReport",
                   project="supersede-report")

    result = remember("Supersede report: the new version", entity="SupersedeReport",
                      project="supersede-report", supersede=old["id"])

    assert result["superseded"] == [old["id"]]


# --- dry_run counts what the real call removes ----------------------------

def test_forget_dry_run_sees_a_graph_node_of_any_scope():
    relate("DryRunNode", "DryRunTarget", "uses", project="dry-run-scope")

    preview = forget(entity="DryRunNode", dry_run=True)

    assert preview["ok"] is True, preview
    assert preview["counts"]["entities"] == 1
    assert preview["counts"]["relations"] == 1


def test_forget_dry_run_and_the_real_call_agree():
    relate("DryRunAgree", "DryRunAgreeTarget", "uses", project="dry-run-scope")

    preview = forget(entity="DryRunAgree", dry_run=True)
    real = forget(entity="DryRunAgree", project="dry-run-scope")

    assert preview["counts"]["entities"] == real["counts"]["entities"]
    assert preview["counts"]["relations"] == real["counts"]["relations"]


def test_forget_reports_the_relations_it_removes():
    relate("RelationCountA", "RelationCountB", "uses", project="relation-count")

    result = forget(entity="RelationCountA", project="relation-count")

    assert result["counts"]["relations"] == 1, result


# --- project="global" is the shared scope ---------------------------------

def test_list_memories_with_the_explicit_global_scope_stays_global():
    remember("Global scope probe: a project fact", entity="GlobalProbe",
             project="global-probe")
    remember("Global scope probe: a global fact", entity="GlobalProbe",
             scope="global")

    result = list_memories(project="global")

    assert result["ok"] is True
    assert {row["project"] for row in result["results"]} == {"global"}


def test_list_entities_with_the_explicit_global_scope_stays_global():
    remember("Global scope probe: a project entity", entity="GlobalEntity",
             project="global-probe")
    remember("Global scope probe: a global entity", entity="GlobalEntity",
             scope="global")
    relate("GlobalEdgeA", "GlobalEdgeB", "uses", project="global-probe")

    result = list_entities(project="global")

    assert {row["project"] for row in result["results"]} == {"global"}


def test_list_memories_without_a_project_still_lists_every_scope():
    remember("Global scope probe: a project fact", entity="GlobalProbe",
             project="global-probe")

    result = list_memories()

    assert "global-probe" in {row["project"] for row in result["results"]}


# --- a scope that only exists in the graph -------------------------------

def test_a_graph_only_project_is_listed_and_can_be_dropped():
    relate("GraphOnlyA", "GraphOnlyB", "uses", project="graph-only")

    listed = list_projects()
    assert any(row["project"] == "graph-only" for row in listed["results"]), listed

    dropped = drop_project(project="graph-only")

    assert dropped["ok"] is True
    assert dropped["counts"]["entities"] == 2
    assert dropped["counts"]["relations"] == 1

    listed_again = list_projects()
    assert not any(row["project"] == "graph-only" for row in listed_again["results"])


# --- get_entity pages instead of truncating silently ---------------------

def test_get_entity_reports_the_total_and_pages():
    for i in range(3):
        remember(f"Entity page: fact number {i}", entity="EntityPage",
                 project="entity-page")

    first = get_entity(entity="EntityPage", project="entity-page", limit=2)

    assert first["total"] == 3
    assert first["count"] == 2
    assert first["has_more"] is True
    assert first["next_offset"] == 2
    assert any("showing 2 of 3" in w for w in first["warnings"]), first

    second = get_entity(entity="EntityPage", project="entity-page", limit=2,
                        offset=first["next_offset"])

    assert second["count"] == 1
    assert second["has_more"] is False


# --- recall explains an empty result -------------------------------------

def test_recall_names_the_filter_that_removed_the_hits():
    remember("Recall report: a fresh but important fact", entity="RecallReport",
             project="recall-report", importance=1.0)

    result = recall("Recall report: a fresh but important fact",
                    project="recall-report", min_decay_score=0.99)

    assert result["count"] == 0
    assert any("min_decay_score" in w for w in result["warnings"]), result


def test_recall_reports_the_scopes_it_searched():
    remember("Recall report: a fact in a project", entity="RecallReport",
             project="recall-report")

    result = recall("Recall report: a fact in a project", project="recall-report")

    assert "scopes_searched" in result
    assert "recall-report" in result["scopes_searched"]


def test_recall_names_the_other_scopes_when_it_finds_nothing():
    remember("Recall report: a fact in a project", entity="RecallReport",
             project="recall-report")

    result = recall("something that is nowhere in the shared scope",
                    min_similarity=0.99)

    if result["count"] == 0:
        assert any("recall-report" in w for w in result["warnings"]), result


# --- import reports a broken metadata per record -------------------------

def test_import_reports_a_broken_metadata_per_record():
    payload = {
        "type": "memory_export",
        "version": "0.2",
        "records": [
            {"id": "broken-meta-1", "content": "Import probe: broken metadata",
             "metadata": "not json"},
            {"id": "good-meta-1", "content": "Import probe: good metadata",
             "metadata": {"a": 1}},
        ],
    }

    result = import_memories(data=json.dumps(payload), project="import-probe")

    assert result["ok"] is True
    assert result["imported"] == 1
    assert result["skipped"] == 1
    assert any("broken-meta-1" in e for e in result.get("errors", [])), result


# --- relation-context sentences can be hidden ----------------------------

def test_list_memories_can_hide_the_relation_context_sentences():
    relate("ContextHideA", "ContextHideB", "uses", project="context-hide",
           context="ContextHideA uses ContextHideB.")

    with_context = list_memories(project="context-hide")
    without_context = list_memories(project="context-hide",
                                    include_relation_context=False)

    assert with_context["count"] > without_context["count"]
    assert all(
        row["metadata"].get("relation_context") != "true"
        for row in without_context["results"]
    )
