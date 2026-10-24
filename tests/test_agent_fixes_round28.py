"""Regression tests for the round-28 live-audit fixes.

1. `remember_batch` rejects unknown per-fact keys individually instead of
   silently dropping them while the rest of the batch is stored.
2. An entity-scoped `recall` includes the relation-context sentences of that
   entity's edges when `include_relation_context` is on (and hides them when
   it is off).
3. `list_relations` warns about a graph project that does not exist instead
   of posing as an empty project.
4. `list_memories` failed calls answer `id: null`, even for an ambiguous id.
5. `find_paths` reports each relation once, under `relation_type`.
"""

from mcp_server.backup import export_memories, import_memories
from mcp_server.graph import (
    find_connections,
    find_paths,
    list_relations,
    relate,
)
from mcp_server.tools.memory import remember, remember_batch
from mcp_server.tools.read import list_memories, recall


def test_remember_batch_rejects_unknown_per_fact_keys():
    result = remember_batch(
        project="round28-batch",
        facts=[
            {"content": "Round28 clean batch fact", "entity": "R28Batch"},
            {"content": "Round28 malformed batch fact",
             "ttl": "2027-01-01", "priority": 3},
        ],
    )

    assert result["ok"] is True, result.get("error")
    assert result["succeeded"] == 1
    assert result["failed"] == 1
    good, bad = result["results"]
    assert good["status"] == "stored"
    assert bad["status"] == "error"
    assert bad["id"] is None
    assert "unknown argument(s)" in bad["error"]
    assert "'priority'" in bad["error"]
    assert "'ttl'" in bad["error"]
    assert "accepted arguments are:" in bad["error"]
    stored = list_memories(project="round28-batch", contains="Round28")
    assert [row["content"] for row in stored["results"]] == [
        "Round28 clean batch fact"]


def test_entity_recall_can_include_relation_context():
    project = "round28-context"
    relate(
        "R28Service", "R28Postgres", "USES",
        context="R28Service writes audit events into R28Postgres.",
        project=project,
    )

    without = recall(
        "writes audit events", project=project, entity="R28Service",
        include_relation_context=False, min_similarity=None)
    assert without["ok"] is True
    assert without["count"] == 0

    with_context = recall(
        "writes audit events", project=project, entity="R28Service",
        include_relation_context=True, min_similarity=None)
    assert with_context["ok"] is True
    assert with_context["count"] == 1
    assert with_context["results"][0]["content"] == (
        "R28Service writes audit events into R28Postgres.")


def test_list_relations_warns_about_a_missing_graph_project():
    result = list_relations(project="round28-missing-graph")

    assert result["ok"] is True
    assert result["count"] == 0
    assert any("does not exist" in warning for warning in result["warnings"])


def test_find_connections_names_a_missing_graph_project():
    result = find_connections(
        entity="R28Missing", project="round28-missing-graph")

    assert result["ok"] is False
    assert result["status"] == "not_found"
    assert "does not exist" in result["error"]


def test_list_memories_ambiguous_id_answers_id_null():
    stored = remember(content="Round28 ambiguous id probe", scope="global")
    assert stored["ok"] is True, stored.get("error")
    memory_id = stored["id"]
    exported = export_memories(project="global")
    imported = import_memories(
        data=exported["data"], format="json", project="round28-ambiguous")
    assert imported["ok"] is True, imported.get("error")

    result = list_memories(memory_id=memory_id)

    assert result["ok"] is False
    assert result["status"] == "error"
    assert "ambiguous memory id" in result["error"]
    assert result["id"] is None


def test_find_paths_reports_relation_type_once():
    project = "round28-paths"
    relate("R28A", "R28B", "USES", project=project)
    relate("R28B", "R28C", "USES", project=project)

    result = find_paths("R28A", "R28C", project=project)

    assert result["ok"] is True, result.get("error")
    assert result["count"] == 1
    relations = result["results"][0]["relations"]
    assert len(relations) == 2
    assert all("type" not in relation for relation in relations)
    assert all(relation["relation_type"] == "USES" for relation in relations)
