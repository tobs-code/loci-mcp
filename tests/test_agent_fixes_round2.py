"""Regression tests for the second round of agent-usability fixes.

Each test protects a concrete dead end an agent hit during the tool sweep:
a silently re-activated superseded fact, an unrecoverable soft delete, a
full-text hit that outranked real relevance, hidden pagination, synthetic
entities in the browse view, an export that hid its own exclusions, and a
contradiction report that could not be acted on.
"""

from mcp_server.server import (
    contradictions_report,
    export_memories,
    forget,
    init_context,
    list_entities,
    list_memories,
    recall,
    relate,
    remember,
    restore_memory,
)


# --------------------------------------------------------------------------
# Dedupe must not silently re-activate a superseded fact
# --------------------------------------------------------------------------

def test_remember_warns_when_it_re_adds_a_superseded_fact():
    old_id = remember("The deploy target is blue")["id"]
    remember("The deploy target is green", supersede=old_id)

    result = remember("The deploy target is blue")

    assert result["status"] == "stored"
    assert any("superseded memory" in w and old_id in w
               for w in result["warnings"])


def test_remember_warns_on_a_semantic_match_with_a_superseded_fact():
    old_id = remember("The deploy target is the blue environment")["id"]
    # The replacing fact is unrelated on purpose: a one-word variant of the
    # old fact is above the semantic dedupe floor (0.85) and would be caught
    # as a duplicate before the superseded-match warning could fire.
    remember("The vault password rotates monthly", supersede=old_id)

    result = remember("The deploy target is the blue environment",
                      dedupe="semantic")

    assert any("superseded memory" in w and old_id in w
               for w in result["warnings"])


# --------------------------------------------------------------------------
# A soft delete must be reversible
# --------------------------------------------------------------------------

def test_restore_memory_brings_a_soft_forgotten_fact_back():
    memory_id = remember("The vault password is hunter2")["id"]
    forget(memory_id)
    assert recall("vault password")["results"] == []

    listed = list_memories(memory_id=memory_id, include_deleted=True)
    assert listed["count"] == 1
    assert listed["results"][0]["deleted_at"] is not None

    restored = restore_memory(memory_id)
    assert restored["ok"] is True
    assert restored["status"] == "restored"
    assert any(r["id"] == memory_id for r in recall("vault password")["results"])

    default_list = list_memories(memory_id=memory_id)
    assert default_list["count"] == 1
    assert "deleted_at" not in default_list["results"][0]


def test_restore_memory_rejects_an_active_or_hard_deleted_fact():
    active_id = remember("This fact was never forgotten")["id"]
    assert restore_memory(active_id)["status"] == "not_found"

    purged_id = remember("This fact is gone for good")["id"]
    forget(purged_id, soft=False)
    assert restore_memory(purged_id)["status"] == "not_found"


# --------------------------------------------------------------------------
# Pagination and direct id lookup
# --------------------------------------------------------------------------

def test_recall_offset_pages_through_the_ranked_results():
    for i in range(5):
        remember(f"Paging probe {i}: the cache node number {i}")

    full = recall("paging probe cache node", limit=10, min_similarity=None)
    assert full["count"] == 5

    page = recall("paging probe cache node", limit=2, offset=1,
                  min_similarity=None)

    assert page["offset"] == 1
    assert page["count"] == 2
    ids = [r["id"] for r in page["results"]]
    assert ids == [r["id"] for r in full["results"][1:3]]
    assert page["has_more"] is True


def test_list_memories_can_look_up_a_single_id():
    wanted = remember("Id lookup probe: the queue worker is idle")["id"]
    remember("Id lookup probe: the scheduler is active")

    result = list_memories(memory_id=wanted)

    assert result["count"] == 1
    assert result["results"][0]["id"] == wanted


# --------------------------------------------------------------------------
# Synthetic relation-context entities must not clutter the browse view
# --------------------------------------------------------------------------

def test_list_entities_hides_relation_context_entities():
    relate("BrowseOne", "BrowseTwo", "KNOWS", context="They met in Berlin")

    names = [r["name"] for r in list_entities()["results"]]

    assert "BrowseOne_KNOWS_BrowseTwo" not in names
    assert "BrowseOne" in names
    assert "BrowseTwo" in names


def test_list_entities_keeps_a_real_entity_with_a_context_memory():
    remember("A real fact about MixedContext", entity="MixedContext")
    relate("MixedContext", "OtherContext", "KNOWS",
           context="A synthetic edge context")

    names = [r["name"] for r in list_entities()["results"]]

    assert "MixedContext" in names
    assert "MixedContext_KNOWS_OtherContext" not in names


# --------------------------------------------------------------------------
# Export must say what it left out
# --------------------------------------------------------------------------

def test_export_warns_about_superseded_and_forgotten_rows():
    old_id = remember("Export probe: the old port is 8000")["id"]
    remember("Export probe: the new port is 9000", supersede=old_id)
    forgotten_id = remember("Export probe: this secret is forgotten")["id"]
    forget(forgotten_id)

    result = export_memories()

    assert result["ok"] is True
    warnings = " ".join(result["warnings"])
    assert "superseded" in warnings
    assert "forgotten" in warnings
    assert "include_superseded=true" in warnings

    complete = export_memories(include_superseded=True)
    assert not any("excludes" in w for w in complete["warnings"])
    exported = {r["id"] for r in complete["data"]["records"]}
    assert old_id in exported
    assert forgotten_id in exported


# --------------------------------------------------------------------------
# Contradictions can be resolved, not only reported
# --------------------------------------------------------------------------

def test_contradictions_report_can_supersede_the_older_fact():
    older = remember("Alice likes tea", entity="Alice")["id"]
    newer = remember("Alice does not like tea", entity="Alice")["id"]

    report = contradictions_report(entity="Alice", resolve="newest")

    assert report["count"] == 1
    assert report["resolved"] == 1
    pair = report["results"][0]
    assert pair["resolution"] == {"superseded": older, "kept": newer}

    history = recall("Alice tea", include_superseded=True, limit=10)["results"]
    old = next(r for r in history if r["id"] == older)
    assert old["superseded_by"] == newer

    active = recall("Alice tea", limit=10)["results"]
    assert all(r["id"] != older for r in active)


# --------------------------------------------------------------------------
# Full-text coverage and the renamed similarity counter
# --------------------------------------------------------------------------

def test_fulltext_coverage_dampens_a_single_word_match():
    # This document matches only the word "backend" but repeats it, which
    # gives it a high raw ts_rank. The 3-word query is only 1/3 covered.
    weak = remember("backend " * 12)["id"]
    strong = remember("database storage backend")["id"]

    result = recall("database storage backend", limit=10, min_similarity=None)
    hits = {r["id"]: r for r in result["results"]}

    assert hits[weak]["_text_score"] < hits[strong]["_text_score"]
    assert hits[weak]["_relevance_score"] < hits[strong]["_relevance_score"]


def test_recall_names_the_candidates_it_dropped():
    remember("Filter probe: the primary cache lives on redis cluster alpha")

    result = recall("a phrase that matches nothing at all", limit=5)

    assert "dropped_by_similarity" in result
    assert "filtered_by_similarity" not in result


# --------------------------------------------------------------------------
# The all-projects bootstrap names the global scope
# --------------------------------------------------------------------------

def test_init_context_all_projects_lists_the_global_scope():
    remember("Bootstrap scope probe", project="scope-probe")

    result = init_context(project="*")

    assert "global" in result["projects"]
    assert "scope-probe" in result["projects"]
