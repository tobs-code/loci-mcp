"""Round-45 features: query forget, delete_entity, scoped batches, pruning.

WS6: `forget(query=...)` removes the semantically matching active rows of one
explicit scope and reports them in `bulk_results`. A hard delete no longer
deletes immediately: it marks `hard_deleted_at` and stays undoable until
`prune_memories` removes it after the grace period.

WS7: `delete_entity` removes a graph node (and its edges by default) without
touching the entity's memories unless asked.

WS8: `get_entities` accepts `{"entity": ..., "project": ...}` entries, so one
call can span scopes.

WS9: `prune_memories` no longer prunes every scope implicitly.

Bugs 1/2/5: `get_entity.memories_by_scope` is a real COUNT(*), graph
connections have a deterministic order, and `relate` without `context` reports
the stored context memory of an existing edge.
"""

from mcp_server.server import (
    delete_entity,
    drop_project,
    forget,
    get_entities,
    get_entity,
    list_entities,
    list_memories,
    prune_memories,
    relate,
    remember,
    update_memory,
)

WS6 = "round45-ws6"
WS6_HARD = "round45-ws6-hard"
WS7 = "round45-ws7"
WS8_A = "round45-ws8-a"
WS8_B = "round45-ws8-b"
WS9_A = "round45-ws9-a"
WS9_B = "round45-ws9-b"
BUG1 = "round45-bug1"
BUG2 = "round45-bug2"
BUG5 = "round45-bug5"


def _reset(*projects: str) -> None:
    for project in projects:
        drop_project(project)


# ---------------------------------------------------------------------------
# WS6: forget(query=...)
# ---------------------------------------------------------------------------

def test_forget_query_requires_an_explicit_scope():
    result = forget(query="anything")
    assert result["ok"] is False
    assert "project" in result["error"]


def test_forget_query_is_mutually_exclusive_with_other_selectors():
    by_id = forget(query="anything", memory_id="abc", project=WS6)
    assert by_id["ok"] is False
    assert "memory_id" in by_id["error"]

    by_contains = forget(query="anything", contains="x", project=WS6)
    assert by_contains["ok"] is False
    assert "mutually exclusive" in by_contains["error"]


def test_forget_query_dry_run_then_soft_forgets_the_matches():
    _reset(WS6)
    remember("WS6 deploy token is ZEBRA-ALPHA", project=WS6)
    remember("WS6 billing note is unrelated", project=WS6)

    dry = forget(query="deploy token", project=WS6, min_similarity=0.3,
                 dry_run=True)
    assert dry["ok"] is True
    assert dry["count"] >= 1
    assert all(item["status"] == "would_soft_forget"
               for item in dry["bulk_results"])
    assert all(item["content_preview"] for item in dry["bulk_results"])
    # Nothing was hidden by the dry run.
    assert list_memories(project=WS6, contains="ZEBRA-ALPHA")["count"] == 1

    real = forget(query="deploy token", project=WS6, min_similarity=0.3)
    assert real["ok"] is True
    assert real["status"] == "soft_forgotten"
    assert real["count"] == dry["count"]
    assert all(item["status"] == "soft_forgotten"
               for item in real["bulk_results"])
    assert list_memories(project=WS6, contains="ZEBRA-ALPHA")["count"] == 0
    assert list_memories(project=WS6, contains="billing note")["count"] == 1


def test_forget_query_without_matches_is_ok_with_a_warning():
    _reset(WS6)
    remember("WS6 canary fact", project=WS6)

    result = forget(query="quantum chromodynamics lattice", project=WS6,
                    min_similarity=0.99)

    assert result["ok"] is True
    assert result["count"] == 0
    assert result["status"] != "not_found"
    assert any("no memories matched" in w for w in result["warnings"])


def test_hard_delete_is_recoverable_until_pruned():
    _reset(WS6_HARD)
    stored = remember("WS6 hard delete probe", project=WS6_HARD)
    assert stored["ok"] is True

    deleted = forget(stored["id"], project=WS6_HARD, soft=False)
    assert deleted["ok"] is True
    assert deleted["status"] == "deleted"
    assert deleted["undo_until"]
    assert deleted["purge_after"]
    assert any("recoverable until" in w for w in deleted["warnings"])

    hidden = list_memories(project=WS6_HARD, memory_id=stored["id"])
    assert hidden["ok"] is False

    pending = prune_memories(project=WS6_HARD, dry_run=True)
    assert pending["ok"] is True
    assert pending["counts"]["pending_purge"] == 1
    assert pending["counts"]["memories"] == 0

    restored = update_memory(stored["id"], project=WS6_HARD, restore=True)
    assert restored["ok"] is True
    assert restored["status"] == "restored"
    assert list_memories(
        project=WS6_HARD, memory_id=stored["id"])["ok"] is True


def test_prune_physically_removes_a_hard_deleted_row_after_the_grace():
    _reset(WS6_HARD)
    stored = remember("WS6 purge probe", project=WS6_HARD)
    forget(stored["id"], project=WS6_HARD, soft=False)

    pruned = prune_memories(
        project=WS6_HARD, hard_delete_grace_days=0, dry_run=False)
    assert pruned["ok"] is True
    assert pruned["counts"]["memories"] >= 1
    assert pruned["scopes_pruned"] == [WS6_HARD]

    gone = list_memories(project=WS6_HARD, memory_id=stored["id"],
                         include_deleted=True)
    assert gone["ok"] is False


# ---------------------------------------------------------------------------
# WS7: delete_entity
# ---------------------------------------------------------------------------

def test_delete_entity_rejects_star_and_unknown_names():
    _reset(WS7)
    star = delete_entity("Anything", project="*")
    assert star["ok"] is False
    assert "*" in star["error"]

    missing = delete_entity("NoSuchEntity", project=WS7)
    assert missing["ok"] is False
    assert missing["status"] == "not_found"


def test_delete_entity_refuses_to_drop_edges_unless_allowed():
    _reset(WS7)
    remember("WS7 node fact", entity="WS7Node", project=WS7)
    relate("WS7Node", "WS7Peer", "KNOWS", project=WS7)

    refused = delete_entity("WS7Node", project=WS7, remove_relations=False)
    assert refused["ok"] is False
    assert refused["status"] == "conflict"
    assert refused["relations"] == 1
    assert "remove_relations=true" in refused["error"]

    dry = delete_entity("WS7Node", project=WS7, dry_run=True)
    assert dry["ok"] is True
    assert dry["counts"]["would_delete_entities"] == 1
    assert dry["counts"]["would_delete_relations"] == 1
    # The dry run changed nothing.
    assert "WS7Node" in [row["name"] for row in list_entities(
        project=WS7, include_global=False)["results"]]

    deleted = delete_entity("WS7Node", project=WS7)
    assert deleted["ok"] is True
    assert deleted["status"] == "deleted"
    assert deleted["counts"]["entities"] == 1
    assert deleted["counts"]["relations"] == 1
    assert "WS7Node" not in [row["name"] for row in list_entities(
        project=WS7, include_global=False, entity_type="graph")["results"]]
    # The memory survives a node-only delete.
    assert list_memories(project=WS7, contains="WS7 node fact")["count"] == 1


def test_delete_entity_can_soft_forget_the_entity_memories():
    _reset(WS7)
    remember("WS7 removable fact", entity="WS7Removable", project=WS7)

    deleted = delete_entity("WS7Removable", project=WS7,
                            remove_memories=True)
    assert deleted["ok"] is True
    assert deleted["counts"]["memories"] >= 1
    assert list_memories(
        project=WS7, contains="WS7 removable fact")["count"] == 0
    hidden = list_memories(project=WS7, contains="WS7 removable fact",
                           include_deleted=True)
    assert hidden["count"] >= 1


# ---------------------------------------------------------------------------
# WS8: cross-scope get_entities
# ---------------------------------------------------------------------------

def test_get_entities_accepts_per_entry_projects():
    _reset(WS8_A, WS8_B)
    remember("WS8 A fact", entity="WS8A", project=WS8_A)
    remember("WS8 B fact", entity="WS8B", project=WS8_B)

    result = get_entities(
        entities=["WS8A", {"entity": "WS8B", "project": WS8_B}],
        project=WS8_A,
    )

    assert result["ok"] is True
    assert result["status"] == "ok"
    assert result["count"] == 2
    assert result["succeeded"] == 2
    assert result["failed"] == 0
    assert result["scopes"] == sorted([WS8_A, WS8_B])
    by_name = {entry["entity"]: entry for entry in result["results"]}
    assert by_name["WS8A"]["project"] == WS8_A
    assert by_name["WS8B"]["project"] == WS8_B
    assert by_name["WS8A"]["scopes_searched"]
    assert by_name["WS8B"]["scopes_searched"]


def test_get_entities_reports_a_bad_entry_without_aborting_the_call():
    _reset(WS8_A)
    remember("WS8 A fact", entity="WS8A", project=WS8_A)

    result = get_entities(
        entities=[
            {"name": "WS8A", "project": WS8_A, "bogus": 1},
            "WS8A",
        ],
        project=WS8_A,
    )

    assert result["ok"] is True
    assert result["status"] == "partial"
    assert result["failed"] == 1
    bad = result["results"][0]
    assert bad["ok"] is False
    assert bad["status"] == "error"
    assert "bogus" in bad["error"]
    assert "accepted arguments are" in bad["error"]
    assert result["results"][1]["ok"] is True


def test_get_entities_dedupes_by_name_and_scope():
    _reset(WS8_A)
    remember("WS8 dedupe fact", entity="WS8Dup", project=WS8_A)

    result = get_entities(
        entities=["WS8Dup", {"entity": "WS8Dup", "project": WS8_A}],
        project=WS8_A,
    )

    assert result["ok"] is True
    assert result["count"] == 1


def test_get_entities_rejects_star_per_entry():
    _reset(WS8_A)
    remember("WS8 star fact", entity="WS8Star", project=WS8_A)

    result = get_entities(
        entities=[{"entity": "WS8Star", "project": "*"}, "WS8Star"],
        project=WS8_A,
    )

    assert result["ok"] is True
    assert result["status"] == "partial"
    assert result["results"][0]["ok"] is False
    assert "*" in result["results"][0]["error"]


# ---------------------------------------------------------------------------
# WS9: prune_memories scope selection
# ---------------------------------------------------------------------------

def test_prune_memories_requires_exactly_one_scope_selector():
    neither = prune_memories()
    assert neither["ok"] is False
    assert "pass project" in neither["error"]

    both = prune_memories(project=WS9_A, all_scopes=True)
    assert both["ok"] is False
    assert "mutually exclusive" in both["error"]


def test_prune_memories_global_only_touches_the_shared_table():
    _reset(WS9_A)
    stored = remember("WS9 global prune probe")
    forget(stored["id"], soft=False)

    result = prune_memories(project="global", hard_delete_grace_days=0,
                            dry_run=True)

    assert result["ok"] is True
    assert result["project"] == "global"
    assert result["all_scopes"] is False
    assert set(result["counts"]["by_scope"]) <= {"global"}
    assert result["counts"]["by_scope"]["global"] >= 1


def test_prune_memories_all_scopes_covers_every_table():
    _reset(WS9_A, WS9_B)
    project_row = remember("WS9 project prune probe", project=WS9_A)
    forget(project_row["id"], project=WS9_A, soft=False)
    global_row = remember("WS9 global prune probe")
    forget(global_row["id"], soft=False)

    result = prune_memories(all_scopes=True, hard_delete_grace_days=0,
                            dry_run=True)

    assert result["ok"] is True
    assert result["all_scopes"] is True
    assert "global" in result["counts"]["by_scope"]
    assert WS9_A in result["counts"]["by_scope"]


def test_prune_memories_project_only_touches_that_scope():
    _reset(WS9_A, WS9_B)
    row = remember("WS9 scoped prune probe", project=WS9_A)
    forget(row["id"], project=WS9_A, soft=False)
    # Create the other scope so an explicit project can be pruned.
    remember("WS9 other scope fact", project=WS9_B)

    result = prune_memories(project=WS9_B, hard_delete_grace_days=0,
                            dry_run=True)

    assert result["ok"] is True
    assert WS9_A not in result["counts"]["by_scope"]


# ---------------------------------------------------------------------------
# Bug 1: memories_by_scope must not depend on limit
# ---------------------------------------------------------------------------

def test_get_entity_memories_by_scope_is_independent_of_limit():
    _reset(BUG1)
    for i in range(5):
        remember(f"BUG1 counted fact {i}", entity="BUG1Entity",
                 project=BUG1)

    for limit in (1, 2, 4):
        result = get_entity("BUG1Entity", project=BUG1, limit=limit)
        assert result["ok"] is True
        assert result["memories_by_scope"][BUG1] == 5, limit


# ---------------------------------------------------------------------------
# Bug 2: graph connections have a deterministic order
# ---------------------------------------------------------------------------

def test_get_entity_connections_are_ordered_by_weight_then_name():
    _reset(BUG2)
    relate("BUG2Hub", "BUG2Zeta", "LINKS", weight=0.2, project=BUG2)
    relate("BUG2Hub", "BUG2Beta", "LINKS", weight=0.9, project=BUG2)
    relate("BUG2Hub", "BUG2Alpha", "LINKS", weight=0.9, project=BUG2)

    first = get_entity("BUG2Hub", project=BUG2)
    second = get_entity("BUG2Hub", project=BUG2)

    names = [edge["entity"]
             for edge in first["details"]["connections"]]
    assert names == ["BUG2Alpha", "BUG2Beta", "BUG2Zeta"]
    assert names == [edge["entity"]
                     for edge in second["details"]["connections"]]


# ---------------------------------------------------------------------------
# Bug 5: relate without context reports the stored sentence
# ---------------------------------------------------------------------------

def test_relate_without_context_reports_the_existing_context_memory():
    _reset(BUG5)
    created = relate("BUG5A", "BUG5B", "LINKS", project=BUG5,
                     context="BUG5 explains why they are linked")
    assert created["ok"] is True
    memory_id = created["context_memory"]
    assert memory_id

    again = relate("BUG5A", "BUG5B", "LINKS", project=BUG5)

    assert again["ok"] is True
    assert again["context_memory"] == memory_id
    assert again["context_reused"] is True
