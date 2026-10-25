"""Round-46 correctness fixes.

Bug A: `forget(query=...)` is a semantic selector. It must keep only rows with
a real vector score at or above `min_similarity`; a keyword-only full-text hit
is not a semantic match and must not be removed.
"""

from mcp_server.db import get_conn
from mcp_server.server import forget, remember
from mcp_server.tools.memory import _mem_table

BUG_A = "round46-bug-a"

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
