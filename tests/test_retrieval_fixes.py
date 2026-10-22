"""Regression tests for the retrieval and usability fixes.

Every test in here protects a bug that made the server harder to use for an
agent: silent empty result sets, a bootstrap that hid fresh facts, scores
that could not be compared, and helpers that destroyed unrelated memories.
"""

import json

from mcp_server.server import (
    export_memories,
    forget,
    init_context,
    list_memories,
    recall,
    remember,
    remember_batch,
    stats_resource,
)


# --------------------------------------------------------------------------
# A full-text hit must survive a weak vector score
# --------------------------------------------------------------------------

def test_fulltext_hit_survives_a_weak_vector_similarity():
    # A long fact with one rare identifier: the embedding model dilutes a
    # single-token query over so many words that the cosine score collapses.
    memory_id = remember(
        "ZEBRAFISH_TOKEN rotates every quarter during the annual platform "
        "maintenance window while the service landscape team reviews the "
        "rollout checklist of the distributed caching layer"
    )["id"]

    loose = recall("ZEBRAFISH_TOKEN", min_similarity=None, limit=5)
    hit = next(r for r in loose["results"] if r["id"] == memory_id)
    # Precondition: the vector retriever alone would drop this hit.
    assert hit["_similarity"] < 0.3

    strict = recall("ZEBRAFISH_TOKEN", limit=5)

    assert any(r["id"] == memory_id for r in strict["results"])


def test_recall_reports_the_candidates_it_filtered():
    remember("Filter probe: the primary cache lives on redis cluster alpha")

    result = recall("a phrase that matches nothing at all", limit=5)

    assert result["ok"] is True
    assert "candidates" in result
    assert "dropped_by_similarity" in result


def test_recall_warns_when_the_filter_removed_everything():
    remember("Empty probe: the deploy target is a kubernetes cluster in eu")

    result = recall("a phrase that matches nothing at all", limit=5,
                    min_similarity=0.99)

    assert result["ok"] is True
    assert result["count"] == 0
    assert any("min_similarity" in w for w in result["warnings"])


def test_recall_says_when_nothing_matches_at_all():
    result = recall("a phrase that matches nothing at all", limit=5)

    assert result["ok"] is True
    assert result["count"] == 0
    assert result["candidates"] == 0
    assert any("nothing in this scope" in w for w in result["warnings"])


# --------------------------------------------------------------------------
# The session bootstrap shows what changed last
# --------------------------------------------------------------------------

def test_init_context_returns_the_most_recent_memories():
    project = "recency"
    old_id = remember("Recency probe: the old fact about the cache layer",
                      project=project)["id"]
    # Read it often, so it becomes the most *used* memory of the scope.
    for _ in range(5):
        recall("old fact cache layer", project=project, limit=1)
    new_id = remember("Recency probe: the brand new fact about the queue worker",
                      project=project)["id"]

    result = init_context(project=project, limit=5)

    ids = [m["id"] for m in result["recent_memories"]]
    assert ids[0] == new_id
    assert old_id in ids


def test_init_context_honours_the_limit():
    project = "bootstrap-limit"
    for i in range(5):
        remember(f"Bootstrap limit probe {i}: fact number {i}", project=project)

    result = init_context(project=project, limit=2)

    assert result["count"] == 2
    assert len(result["recent_memories"]) == 2


# --------------------------------------------------------------------------
# Scores stay comparable
# --------------------------------------------------------------------------

def test_all_published_scores_stay_within_zero_and_one():
    remember("Score probe: the primary database is postgres with pgvector",
             entity="Scores")
    remember("Score probe: coffee tastes better with oat milk", entity="Scores")

    result = recall("primary database postgres pgvector", limit=5,
                    include_decay=True)

    for hit in result["results"]:
        for key in ("_similarity", "_relevance_score", "_final_score"):
            if hit.get(key) is None:
                continue
            assert 0.0 <= hit[key] <= 1.0, (key, hit[key])


def test_relevance_score_separates_a_strong_hit_from_a_weak_one():
    remember("Spread probe: the primary database is postgres with pgvector")
    remember("Spread probe: coffee tastes better with oat milk")

    result = recall("primary database postgres pgvector", limit=5,
                    min_similarity=None)

    scores = [r["_relevance_score"] for r in result["results"]]
    assert len(scores) >= 2
    assert max(scores) - min(scores) > 0.1
    assert "primary database" in result["results"][0]["content"]


# --------------------------------------------------------------------------
# supersede_last does not destroy unrelated facts
# --------------------------------------------------------------------------

def test_decay_filter_reports_dropped_hits_on_non_empty_results():
    # The decay filter used to be silent while hits survived: an agent saw
    # fewer results without learning that min_decay_score removed candidates.
    remember("Decay drop probe: fresh and important side of the topic",
             importance=1.0)
    remember("Decay drop probe: barely weighted low-importance side note",
             importance=0.1)

    result = recall("Decay drop probe", limit=5, include_decay=True,
                    min_similarity=None, min_decay_score=0.7)

    assert result["ok"] is True
    assert len(result["results"]) >= 1
    assert result.get("dropped_by_decay", 0) >= 1


def test_supersede_last_refuses_an_unrelated_replacement():
    remember("the deployment pipeline runs on friday", entity="Guard")
    unrelated = remember("coffee tastes better with oat milk",
                         entity="Guard")["id"]

    result = remember("the release checklist lives in RELEASE.md",
                      entity="Guard", supersede_last=True)

    assert result["ok"] is True
    assert result["superseded"] == []
    assert any("supersede_last skipped" in w for w in result["warnings"])
    hits = recall("coffee oat milk", limit=5)["results"]
    assert any(h["id"] == unrelated and h["superseded_by"] is None
               for h in hits)


def test_supersede_last_replaces_the_most_similar_fact_not_the_newest():
    similar = remember("the build runs on friday", entity="Build4")["id"]
    # A newer, unrelated fact of the same entity used to BLOCK the
    # replacement: only the newest memory was measured against the new one.
    remember("coffee tastes better with oat milk", entity="Build4")

    result = remember("the build runs on saturday", entity="Build4",
                      supersede_last=True)

    assert result["ok"] is True
    assert result["superseded"] == [similar]


def test_supersede_last_still_replaces_a_related_fact():
    first = remember("the build runs on friday", entity="Build2")["id"]
    second = remember("the build runs on saturday", entity="Build2")["id"]

    result = remember("the build runs on sunday", entity="Build2",
                      supersede_last=True)

    assert result["ok"] is True
    # Both memory ids are about the same topic, so either may win the
    # most-similar race - but exactly one of them must be replaced.
    assert len(result["superseded"]) == 1
    assert result["superseded"][0] in (first, second)
    history = recall("build runs on", include_superseded=True, limit=10)["results"]
    assert {r["id"] for r in history} == {first, second, result["id"]}


# --------------------------------------------------------------------------
# Duplicates explain themselves
# --------------------------------------------------------------------------

def test_duplicate_warning_names_the_dedupe_mode():
    remember("Dedupe mode probe: the invoice is paid monthly")

    exact = remember("Dedupe mode probe: the invoice is paid monthly")
    semantic = remember("Dedupe mode probe: the invoice is paid monthly.",
                        dedupe="semantic")

    assert exact["status"] == "duplicate"
    assert exact["duplicate_mode"] == "exact"
    assert "exact" in exact["warnings"][0]
    assert semantic["status"] == "duplicate"
    assert semantic["duplicate_mode"] == "semantic"
    assert "semantic" in semantic["warnings"][0]


# --------------------------------------------------------------------------
# One name for the same thing
# --------------------------------------------------------------------------

def test_export_writes_both_type_and_mem_type():
    remember("Export naming probe", mem_type="decision")

    payload = export_memories()["data"]

    record = payload["records"][0]
    assert record["type"] == "decision"
    assert record["mem_type"] == "decision"


def test_memories_report_their_project():
    remember("Project naming probe", project="naming")

    rows = list_memories(project="naming", limit=5)["results"]

    assert rows
    assert all("project" in row for row in rows)
    assert rows[0]["project"] == "naming"


def test_the_project_copy_wins_over_the_global_copy():
    remember("Scope dedupe probe: the same content in both scopes", project="dup")
    remember("Scope dedupe probe: the same content in both scopes", scope="global")

    listed = list_memories(limit=10)["results"]
    matching = [r for r in listed
                if r["content"].startswith("Scope dedupe probe")]

    assert len(matching) == 1
    assert matching[0]["project"] == "dup"

    recalled = recall("Scope dedupe probe", project="dup", limit=10)["results"]
    matching = [r for r in recalled
                if r["content"].startswith("Scope dedupe probe")]
    assert len(matching) == 1
    assert matching[0]["project"] == "dup"


def test_stats_resource_lists_the_projects_sorted():
    remember("Stats probe a", project="zeta")
    remember("Stats probe b", project="alpha")

    stats = json.loads(stats_resource())

    assert stats["projects"] == sorted(stats["projects"])
    assert "alpha" in stats["projects"]
    assert "zeta" in stats["projects"]


# --------------------------------------------------------------------------
# Browsing by literal text
# --------------------------------------------------------------------------

def test_list_memories_can_filter_on_the_literal_content():
    remember("Literal probe: the token is ZEBRAFISH_TOKEN")
    remember("Literal probe: coffee tastes better with oat milk")

    rows = list_memories(contains="ZEBRAFISH_TOKEN", limit=5)["results"]

    assert len(rows) == 1
    assert "ZEBRAFISH_TOKEN" in rows[0]["content"]


def test_list_memories_escapes_wildcards_in_the_content_filter():
    remember("Wildcard probe: the release is 100% done")

    rows = list_memories(contains="100%", limit=5)["results"]

    assert len(rows) == 1


def test_batch_duplicate_reports_the_dedupe_mode():
    remember("Batch mode probe: the report is weekly")

    result = remember_batch([{"content": "Batch mode probe: the report is weekly"}],
                            dedupe="semantic")

    entry = result["results"][0]
    assert entry["status"] == "duplicate"
    assert entry["duplicate_mode"] == "exact"


# --------------------------------------------------------------------------
# Removing things: preview first, purge a whole scope
# --------------------------------------------------------------------------

def test_forget_can_preview_what_it_would_remove():
    memory_id = remember("Dry run probe: the invoice is paid monthly")["id"]

    result = forget(memory_id=memory_id, dry_run=True)

    assert result["ok"] is True
    assert result["status"] == "ok"
    assert result["counts"]["memories"] == 1
    assert result["dry_run"] is True
    assert "Would remove" in result["message"]
    # Nothing was removed.
    hits = recall("invoice paid monthly", limit=5)["results"]
    assert any(h["id"] == memory_id for h in hits)


def test_forget_can_preview_a_whole_project():
    remember("Dry run scope probe: the invoice is paid monthly",
             project="dry-run-scope")

    result = forget(entity="*", project="dry-run-scope", dry_run=True)

    assert result["ok"] is True
    assert result["status"] == "ok"
    assert result["counts"]["memories"] == 1
    assert list_memories(project="dry-run-scope", include_global=False,
                         limit=10)["results"] != []


def test_forget_can_purge_a_whole_project():
    remember("Purge probe: the invoice is paid monthly", project="purge-me")
    remember("Purge probe: the report is weekly", project="purge-me")
    keep = remember("Purge probe: the global policy stays",
                    scope="global")["id"]

    result = forget(entity="*", project="purge-me", soft=False)

    assert result["ok"] is True
    assert result["counts"]["memories"] == 2
    assert list_memories(project="purge-me", include_global=False,
                         limit=10)["results"] == []
    # The global scope is untouched.
    hits = recall("global policy stays", limit=5)["results"]
    assert any(h["id"] == keep for h in hits)


def test_forget_refuses_to_purge_without_a_project():
    result = forget(entity="*")

    assert result["ok"] is False
    assert "project" in result["error"]
