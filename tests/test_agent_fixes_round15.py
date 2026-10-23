"""Regression tests for the sixteenth round of fixes.

Covers:
- a noisy NLI hit between unrelated statements is never auto-resolved;
- `contradictions_report(resolve="newest", dry_run=true)` reports what it
  would invalidate without writing;
- a write that creates a new project scope says so, and `list_projects`
  makes every scope discoverable;
- `list_memories` / `recall` report cross-scope duplicates instead of hiding
  them silently, and `include_duplicates=true` shows every copy;
- `init_context()` names the other scopes when it only summarized global;
- `graph_boost` includes the relation-context sentence of a traversed edge.
"""

import mcp_server.tools.maintenance as maintenance
from mcp_server.server import (
    contradictions_report,
    init_context,
    list_memories,
    list_projects,
    recall,
    relate,
    remember,
)


def _fake_nli(*args, **kwargs):
    documents = kwargs.get("documents") or (args[1] if len(args) > 1 else [])
    return [0.95] * len(documents), None, []


def test_shares_subject_rejects_unrelated_statements():
    assert maintenance._shares_subject(
        "The deployment pipeline uses GitHub Actions",
        "The API version is 2.3.1",
    ) is False
    assert maintenance._shares_subject(
        "The API version is 2.3.1",
        "The API version is 3.0.0",
    ) is True


def test_contradictions_dry_run_reports_without_writing(monkeypatch):
    monkeypatch.setattr(maintenance, "nli_detailed", _fake_nli)
    project = "round15-dryrun"
    entity = "round15-dryrun-service"
    old = remember("Round15 service listens on port 8000", entity=entity,
                   project=project, dedupe="off")["id"]
    new = remember("Round15 service listens on port 9000", entity=entity,
                   project=project, dedupe="off")["id"]

    result = contradictions_report(project=project, entity=entity,
                                   resolve="newest", dry_run=True)

    assert result["ok"] is True
    assert result["resolved"] == 0
    assert result["would_resolve"] >= 1
    assert result["dry_run"] is True
    pair = next(p for p in result["results"]
                if {p["a"]["id"], p["b"]["id"]} == {old, new})
    assert pair["resolution"]["dry_run"] is True
    assert pair["resolution"]["superseded"] == old

    # Nothing was invalidated: both facts are still active.
    active = list_memories(project=project, include_global=False,
                           fields=["id"], limit=50)["results"]
    ids = {row["id"] for row in active}
    assert {old, new} <= ids


def test_new_scope_is_announced_and_listed():
    project = "round15-listed-scope"
    stored = remember("Round15 discovers a brand new scope", project=project)
    assert stored["ok"] is True
    assert stored["scope"] == "project"
    assert any("new project scope" in w for w in stored["warnings"])

    active_only = list_projects(include_empty=False)
    assert active_only["ok"] is True
    assert project in {row["project"] for row in active_only["results"]}

    listing = list_projects()
    assert listing["ok"] is True
    rows = {row["project"]: row for row in listing["results"]}
    assert "global" in rows
    assert rows[project]["memories"] >= 1
    assert rows[project]["has_table"] is True
    assert rows[project]["scope"] == "project"


def test_cross_scope_duplicates_are_reported_not_hidden():
    text = "Round15 cross-scope duplicate probe 4821"
    remember(text)
    remember(text, project="round15-dup")

    browsed = list_memories(project="*", contains="Round15 cross-scope", limit=50)
    assert browsed["ok"] is True
    assert len(browsed["results"]) == 1
    assert browsed.get("duplicates_hidden", 0) == 1
    assert any("duplicate" in w for w in browsed["warnings"])

    full = list_memories(project="*", contains="Round15 cross-scope", limit=50,
                         include_duplicates=True)
    assert len(full["results"]) == 2

    searched = recall("Round15 cross-scope duplicate probe", project="*",
                      limit=50)
    assert searched["ok"] is True
    assert searched.get("dropped_by_cross_scope_duplicate", 0) >= 1


def test_init_context_names_the_scopes_it_did_not_summarize():
    remember("Round15 context probe", project="round15-context")
    result = init_context(limit=5)
    assert result["ok"] is True
    assert any("other scopes exist" in w for w in result["warnings"])


def test_graph_boost_includes_relation_context():
    project = "round15-graph"
    from_entity = "Round15Source"
    to_entity = "Round15Target"
    relate(from_entity=from_entity, to_entity=to_entity,
           relation_type="causes", project=project,
           context="Round15Source causes Round15Target")

    result = recall("zebra galvanometer", project=project, limit=50,
                    graph_boost=True, graph_entity=from_entity)
    assert result["ok"] is True
    contents = [row["content"] for row in result["results"]]
    assert "Round15Source causes Round15Target" in contents
    context_rows = [row for row in result["results"]
                    if row.get("_graph_relation_context")]
    assert context_rows
