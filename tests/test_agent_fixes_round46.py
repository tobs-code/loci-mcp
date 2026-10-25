"""Round-46 correctness fixes.

Bug A: `forget(query=...)` is a semantic selector. It must keep only rows with
a real vector score at or above `min_similarity`; a keyword-only full-text hit
is not a semantic match and must not be removed.

Bug B: `prune_memories` must never purge a hard-deleted row through the
expired/forgotten branches. Only the hard-delete grace branch may remove it.
"""

from mcp_server.app import app
from mcp_server.db import get_conn
from mcp_server.server import forget, prune_memories, remember
from mcp_server.tools.memory import _mem_table

BUG_A = "round46-bug-a"
BUG_B = "round46-bug-b"

QUERY = "Kubernetes deployment rollout"


def _drop_embedding(project: str, memory_id: str) -> None:
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            f"UPDATE {_mem_table(project)} SET embedding = NULL WHERE id = %s",
            (memory_id,))
        conn.commit()
    finally:
        conn.close()


def test_forget_query_keeps_only_vector_matches_above_min_similarity():
    pipeline = remember(
        "The deployment pipeline runs on Kubernetes with a blue-green "
        "rollout strategy.", project=BUG_A)
    history = remember(
        "Kubernetes was originally open-sourced by Google in 2014.",
        project=BUG_A)
    rollback = remember(
        "A Kubernetes deployment rollout needs a documented rollback plan.",
        project=BUG_A)
    _drop_embedding(BUG_A, history["id"])

    result = forget(query=QUERY, project=BUG_A, min_similarity=0.45,
                    dry_run=True)
    assert result["ok"] is True
    listed = {entry["id"] for entry in result["bulk_results"]}
    assert pipeline["id"] in listed
    assert rollback["id"] in listed
    assert history["id"] not in listed
    for entry in result["bulk_results"]:
        assert entry["similarity"] is not None
        assert entry["similarity"] >= 0.45


def test_forget_query_never_selects_a_keyword_only_hit():
    history = remember(
        "Kubernetes was originally open-sourced by Google in 2014.",
        project=BUG_A)
    remember(
        "A Kubernetes deployment rollout needs a documented rollback plan.",
        project=BUG_A)
    _drop_embedding(BUG_A, history["id"])

    result = forget(query=QUERY, project=BUG_A, min_similarity=0.0,
                    dry_run=True)
    assert result["ok"] is True
    assert history["id"] not in {
        entry["id"] for entry in result["bulk_results"]}
    for entry in result["bulk_results"]:
        assert entry["similarity"] is not None


def test_prune_memories_does_not_purge_a_hard_deleted_row_before_grace():
    stored = remember("Round46 grace probe", project=BUG_B)
    forget(stored["id"], project=BUG_B, soft=False)

    pending = prune_memories(project=BUG_B, dry_run=True, older_than_days=0,
                             hard_delete_grace_days=30)
    assert pending["ok"] is True
    assert pending["counts"]["memories"] == 0
    assert pending["counts"]["pending_purge"] == 1
    assert pending["counts"]["by_scope"] == {}


def test_prune_memories_purges_a_hard_deleted_row_after_grace():
    stored = remember("Round46 expired grace probe", project=BUG_B)
    forget(stored["id"], project=BUG_B, soft=False)

    due = prune_memories(project=BUG_B, dry_run=True, older_than_days=30,
                         hard_delete_grace_days=0)
    assert due["ok"] is True
    assert due["counts"]["memories"] == 1
    assert due["counts"]["pending_purge"] == 0


def test_prune_memories_still_removes_soft_forgotten_rows():
    stored = remember("Round46 soft forgotten probe", project=BUG_B)
    forget(stored["id"], project=BUG_B)

    pruned = prune_memories(project=BUG_B, dry_run=True, older_than_days=0,
                            hard_delete_grace_days=30)
    assert pruned["ok"] is True
    assert pruned["counts"]["memories"] == 1
    assert pruned["counts"]["pending_purge"] == 0


def test_forget_query_default_min_similarity_is_080():
    schema = app._tool_manager.get_tool("forget").parameters
    prop = schema["properties"]["min_similarity"]

    assert prop["default"] == 0.80
    assert "0.80" in prop["description"]
