"""Regression tests for the twentieth round of fixes (agent audit follow-up).

Covers:
- `get_entity` reports the same total as the page it returns (cross-scope
  copies hidden, same-scope duplicates kept);
- `import_memories` uses the normal importance default instead of inventing
  1.0;
- `export_memories(include_deleted=...)` can separate soft-forgotten rows
  from superseded history;
- `remember_batch(skip_duplicates=false)` names the exact copy it stored
  again;
- a relation-context sentence is not a near-duplicate of the fact that
  describes the same edge;
- `contradictions_report(dry_run=true)` counts what a real resolve performs;
- `graph_boost` never pushes a query hit out of the page;
- `verify_integrity` names supersede chains;
- an empty filtered browse names the filters;
- `forget` names the `entity` argument;
- `relate_batch` reports the offending edge index.
"""

from mcp_server.db import get_conn
from mcp_server.graph import relate, relate_batch
from mcp_server.search import _search_memories_graph_boost
from mcp_server.server import (
    contradictions_report,
    export_memories,
    forget,
    get_entity,
    import_memories,
    list_memories,
    remember,
    remember_batch,
    verify_integrity,
)


def _set_updated_at(memory_id: str, interval: str) -> None:
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            f"UPDATE memories SET updated_at = NOW() - INTERVAL '{interval}' "
            f"WHERE id = %s",
            (memory_id,),
        )
        conn.commit()
    finally:
        conn.close()


def test_get_entity_total_matches_deduped_page():
    project = "round20-entity"
    entity = "Round20Entity"
    content = "Round20 shared fact about the entity"

    remember(content, entity=entity)
    remember(content, entity=entity, project=project)

    result = get_entity(entity=entity, project=project)

    assert result["ok"] is True
    assert result["total"] == 1
    assert len(result["memories"]) == 1
    assert result["memories"][0]["scope"] == "project"


def test_get_entity_keeps_same_scope_duplicates_in_total():
    project = "round20-duplicates"
    entity = "Round20Duplicates"
    content = "Round20 intentionally stored twice"

    batch = remember_batch(
        [{"content": content, "entity": entity},
         {"content": content, "entity": entity}],
        project=project, skip_duplicates=False,
    )
    assert batch["ok"] is True

    result = get_entity(entity=entity, project=project)

    assert result["total"] == 2
    assert len(result["memories"]) == 2


def test_import_uses_the_normal_importance_default():
    project = "round20-import"
    mid = "0f0f0f0f0f0f0f20"

    result = import_memories(data={
        "records": [{"id": mid, "content": "Round20 imported without importance"}]
    }, project=project)

    assert result["ok"] is True
    listed = list_memories(memory_id=mid, project=project)
    assert listed["ok"] is True
    assert listed["memory"]["importance"] == 0.5


def test_export_can_leave_soft_forgotten_rows_out():
    project = "round20-export"
    old_id = remember("Round20 export old fact", project=project)["id"]
    remember("Round20 export new fact", project=project, supersede=old_id)
    forgotten_id = remember("Round20 export forgotten fact",
                             project=project)["id"]
    forget(forgotten_id)

    history_only = export_memories(
        project=project, include_superseded=True, include_deleted=False)
    exported = {r["id"] for r in history_only["data"]["records"]}
    assert old_id in exported
    assert forgotten_id not in exported
    assert any("include_deleted=true" in w for w in history_only["warnings"])

    complete = export_memories(
        project=project, include_superseded=True, include_deleted=True)
    complete_ids = {r["id"] for r in complete["data"]["records"]}
    assert forgotten_id in complete_ids

    # include_deleted alone keeps forgotten rows without pretending they were
    # left out, while superseded history stays excluded.
    forgotten_only = export_memories(project=project, include_deleted=True)
    assert forgotten_id in {r["id"] for r in forgotten_only["data"]["records"]}
    assert not any("forgotten" in w and "excludes" in w
                   for w in forgotten_only["warnings"])

    # The default stays backwards compatible: include_superseded alone still
    # carries forgotten rows, so old export/import round trips keep working.
    legacy = export_memories(project=project, include_superseded=True)
    assert forgotten_id in {r["id"] for r in legacy["data"]["records"]}


def test_remember_batch_warns_when_skip_duplicates_stores_a_copy():
    project = "round20-batch"
    content = "Round20 exact copy in one batch"

    first = remember(content, project=project)
    result = remember_batch(
        [{"content": content}],
        project=project, skip_duplicates=False,
    )

    assert result["ok"] is True
    entry = result["results"][0]
    assert entry["id"] != first["id"]
    assert any("skip_duplicates=false" in w and first["id"] in w
               for w in entry["warnings"])


def test_relation_context_is_not_a_near_duplicate_of_its_fact():
    project = "round20-context"
    entity = "Round20Payments"
    relate(
        from_entity=entity, to_entity="Round20Kubernetes",
        relation_type="RUNS_ON",
        context="Round20 payments api runs on kubernetes",
        project=project,
    )

    result = remember(
        "Round20 payments api is running on kubernetes",
        entity=entity, project=project, dedupe="exact",
    )

    assert result["ok"] is True
    assert not any("near-duplicate" in w for w in result["warnings"])


def test_contradiction_dry_run_matches_the_real_resolve_count():
    project = "round20-contradictions"
    entity = "Round20Relay"
    a = remember("Round20 relay is not enabled", project=project,
                 entity=entity, dedupe="off")
    b = remember("Round20 relay is enabled", project=project,
                 entity=entity, dedupe="off")
    c = remember("Round20 relay is active", project=project,
                 entity=entity, dedupe="off")
    _set_updated_at(a["id"], "3 days")
    _set_updated_at(b["id"], "2 days")
    _set_updated_at(c["id"], "1 day")

    dry = contradictions_report(
        entity=entity, project=project, resolve="newest", dry_run=True)
    assert dry["would_resolve"] == 1
    skipped = [p["resolution"] for p in dry["results"]
               if p.get("resolution", {}).get("skipped")]
    assert len(skipped) == 1

    real = contradictions_report(
        entity=entity, project=project, resolve="newest")
    assert real["resolved"] == dry["would_resolve"]


def test_graph_boost_fills_only_the_slots_left_by_query_hits():
    project = "round20-graph"
    hub = "Round20Hub"
    neighbor = "Round20Neighbor"
    for index in range(5):
        remember(f"Round20 graph note {index}", project=project,
                 entity=neighbor)
    relate(from_entity=hub, to_entity=neighbor, relation_type="LINKS_TO",
           project=project)

    existing = [{
        "id": "round20-query-hit",
        "content": "Round20 weak but real query hit",
        "entity": hub,
        "_rrf_score": 0.01,
        "_relevance_score": 0.0,
        "_final_score": 0.0,
    }]
    results = _search_memories_graph_boost(existing, hub, 1, project)

    assert len(results) == 1
    assert results[0]["id"] == "round20-query-hit"


def test_verify_integrity_names_supersede_chains():
    project = "round20-chain"
    first = remember("Round20 chain fact one", project=project)["id"]
    second = remember("Round20 chain fact two", project=project,
                      supersede=first)["id"]
    remember("Round20 chain fact three", project=project,
             supersede=second)

    result = verify_integrity(project=project)

    assert result["chains"] >= 1
    assert any("chain" in w for w in result["warnings"])
    assert not any("no integrity problems found" in w
                   for w in result["warnings"])


def test_empty_filtered_browse_names_the_filter():
    project = "round20-empty"
    remember("Round20 present fact", project=project, entity="Round20Here")

    result = list_memories(project=project, entity="Round20Missing")

    assert result["total"] == 0
    assert any("entity" in w and "narrowed" in w for w in result["warnings"])


def test_forget_names_the_entity_argument():
    result = forget()

    assert result["ok"] is False
    assert "entity" in result["error"]
    assert "entity_name" not in result["error"]


def test_relate_batch_reports_the_offending_edge_index():
    result = relate_batch(edges=[
        {"from_entity": "Round20A", "to_entity": "Round20B",
         "relation_type": "LINK"},
        {"from_entity": "Round20A", "relation_type": "LINK"},
    ])

    assert result["ok"] is False
    assert "edge 1" in result["error"]
