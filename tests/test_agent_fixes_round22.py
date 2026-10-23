"""Regression tests for the twenty-second round of fixes (agent audit).

Covers:
- `supersede_last` accepts an attribute update that adds detail words
  (containment instead of Jaccard) but still refuses another attribute that
  shares the sentence template;
- a same-entity near-duplicate about another subject is labelled as such and
  does not advise a blind supersede;
- `forget` reports the project it acted in;
- `update_memory`/`update_batch` answer `conflict` (not `not_found`) for a
  row that exists but is superseded/forgotten/expired;
- `get_entity` can hide relation-context sentences;
- `unrelate` removes graph nodes that it left without an edge or a memory;
- the dry-run counts of `forget(entity='*')` and `drop_project` agree about
  already-forgotten rows;
- `contradictions_report(resolve="newest")` uses the creation time, so a
  `touch` does not decide which fact survives.
"""

import mcp_server.tools.memory as memory
from mcp_server.server import (
    contradictions_report,
    drop_project,
    forget,
    get_entity,
    list_entities,
    list_memories,
    prune_memories,
    recall_batch,
    relate,
    remember,
    unrelate,
    update_batch,
    update_memory,
)


def _tokens(*texts):
    return [memory._topic_tokens(text, "Refund worker") for text in texts]


def test_topic_guard_accepts_an_attribute_update_with_extra_detail():
    old = ("Refund worker retries failed jobs three times with exponential "
           "backoff.")
    new = ("Refund worker retries failed jobs up to five times with a 30 "
           "second timeout.")

    assert memory._shares_subject(new, old, "Refund worker", 0.9638) is True


def test_topic_guard_refuses_a_different_attribute():
    old = "The Round19Svc runs on port 8766 using postgresql 16 in staging"
    new = ("The Round19Svc runs a nightly backup job using postgresql 16 in "
           "staging")

    assert memory._shares_subject(new, old, "Round19Svc", 0.7915) is False


def test_same_entity_near_duplicate_about_another_subject_is_not_superseded():
    near_dup = {
        "id": "abc123",
        "content": "Alpha service primary region is eu-central-1",
        "entity": "Alpha",
        "similarity": 0.83,
    }

    warning = memory._near_duplicate_warning(
        near_dup, "Alpha", new_content="Alpha service legacy region is us-east-1")

    assert "different subject" in warning
    assert "supersede=abc123 to replace it" not in warning


def test_forget_reports_the_project_it_acted_in():
    stored = remember("forget project probe", project="round22-forget")
    result = forget(stored["id"], project="round22-forget")

    assert result["ok"] is True
    assert result["project"] == "round22-forget"


def test_update_memory_reports_a_hidden_row_as_conflict():
    old = remember("round22 old deployment fact", project="round22-conflict")
    new = remember("round22 new deployment fact", project="round22-conflict",
                   supersede=old["id"])

    superseded = update_memory(old["id"], importance=0.2,
                              project="round22-conflict")
    forgotten = remember("round22 forgettable fact", project="round22-conflict")
    forget(forgotten["id"], project="round22-conflict")
    soft = update_memory(forgotten["id"], touch=True,
                         project="round22-conflict")

    assert superseded["status"] == "conflict"
    assert soft["status"] == "conflict"
    assert new["ok"] is True


def test_update_batch_reports_a_hidden_row_as_conflict():
    stored = remember("round22 batch forgotten fact",
                      project="round22-batch-conflict")
    forget(stored["id"], project="round22-batch-conflict")

    result = update_batch(
        [{"memory_id": stored["id"], "touch": True}],
        project="round22-batch-conflict")

    assert result["results"][0]["status"] == "conflict"


def test_get_entity_can_hide_relation_context():
    relate("round22-a", "round22-b", "DEPENDS_ON",
           context="round22-a depends on round22-b",
           project="round22-entity")
    remember("round22-a has a health endpoint", entity="round22-a",
             project="round22-entity")

    with_context = get_entity("round22-a", project="round22-entity")
    without_context = get_entity("round22-a", project="round22-entity",
                                 include_relation_context=False)

    assert with_context["relation_context_count"] == 1
    assert without_context["relation_context_count"] == 0
    assert without_context["total"] == 1


def test_unrelate_drops_the_graph_nodes_it_orphaned():
    relate("round22-x", "round22-y", "USES", project="round22-orphan")

    result = unrelate("round22-x", "round22-y", "USES",
                      project="round22-orphan")
    listed = list_entities(project="round22-orphan")

    assert result["ok"] is True
    assert set(result["orphan_nodes_removed"]) == {"round22-x", "round22-y"}
    assert listed["results"] == []


def test_dry_run_counts_agree_about_already_forgotten_rows():
    first = remember("round22 count one", project="round22-counts")
    remember("round22 count two", project="round22-counts")
    forget(first["id"], project="round22-counts")

    emptied = forget(entity="*", project="round22-counts", dry_run=True)
    dropped = drop_project("round22-counts", dry_run=True)

    assert emptied["counts"]["memories"] == 1
    assert emptied["counts"]["active_memories"] == 1
    assert emptied["counts"]["already_forgotten"] == 1
    assert dropped["counts"]["memories"] == 2
    assert dropped["counts"]["active_memories"] == 1
    assert dropped["counts"]["already_forgotten"] == 1


def test_resolve_newest_ignores_a_touch():
    old = remember("The round22 cache is enabled", entity="Round22Cache",
                   project="round22-resolve")
    new = remember("The round22 cache is not enabled", entity="Round22Cache",
                   project="round22-resolve")

    # A decay refresh must not promote the older fact to "newest".
    update_memory(old["id"], touch=True, project="round22-resolve")
    result = contradictions_report(entity="Round22Cache",
                                   project="round22-resolve",
                                   resolve="newest",
                                   resolve_min_confidence=0.8)
    history = list_memories(memory_id=old["id"], project="round22-resolve",
                            include_superseded=True)["memory"]

    assert result["resolved"] == 1
    assert history["superseded_by"] == new["id"]


def test_recall_batch_returns_one_ranked_list_per_query():
    remember("round22 deploy uses helm", project="round22-batch-search")
    remember("round22 logs live in loki", project="round22-batch-search")

    result = recall_batch(
        ["deploy helm", "logs loki"], project="round22-batch-search",
        detail="slim")

    assert result["ok"] is True
    assert result["count"] == 2
    assert [entry["query"] for entry in result["results"]] == [
        "deploy helm", "logs loki"]
    assert all(entry["ok"] for entry in result["results"])


def test_recall_batch_rejects_empty_queries():
    result = recall_batch(["deploy", "  "])

    assert result["ok"] is False
    assert "query 1 is empty" in result["error"]


def test_prune_memories_only_removes_inactive_rows():
    expired = remember("round22 expired fact",
                       project="round22-prune",
                       expires_at="2020-01-01T00:00:00Z")
    forgotten = remember("round22 forgotten fact", project="round22-prune")
    active = remember("round22 active fact", project="round22-prune")
    forget(forgotten["id"], project="round22-prune")

    preview = prune_memories(project="round22-prune", older_than_days=0)
    deleted = prune_memories(project="round22-prune", older_than_days=0,
                             dry_run=False)

    assert preview["dry_run"] is True
    assert preview["counts"]["memories"] == 2
    assert deleted["counts"]["memories"] == 2
    assert list_memories(memory_id=active["id"],
                         project="round22-prune")["ok"] is True
    assert list_memories(memory_id=expired["id"], project="round22-prune",
                         include_expired=True)["ok"] is False
    assert list_memories(memory_id=forgotten["id"], project="round22-prune",
                         include_deleted=True)["ok"] is False
