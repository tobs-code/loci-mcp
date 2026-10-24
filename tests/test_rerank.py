"""Unit tests for the optional cross-encoder reranking stage.

The full retrieval tests disable reranking (the fake inference server has no
/rerank endpoint), so these tests exercise the ordering, score and fallback
logic directly with a stubbed client.
"""

from mcp_server import search

import pytest


def _rows():
    return [
        {"id": "decoy", "content": "Deployment window is Sunday",
         "_relevance_score": 0.42, "_rrf_score": 0.03},
        {"id": "answer", "content": "Deployment needs two approvals",
         "_relevance_score": 0.40, "_rrf_score": 0.02},
        {"id": "tail", "content": "Logs are kept for 30 days",
         "_relevance_score": 0.10, "_rrf_score": 0.01},
    ]


def test_rerank_reorders_ambiguous_block(monkeypatch):
    monkeypatch.setattr(search, "RERANK_ENABLED", True)
    monkeypatch.setattr(search, "RERANK_ALWAYS", True)
    monkeypatch.setattr(search, "rerank",
                        lambda query, docs: [0.0, 5.0, 1.0])

    rows = _rows()
    stats = {}
    search._rerank_rows(rows, "how do we deploy?", stats, False)

    assert [row["id"] for row in rows] == ["answer", "tail", "decoy"]
    assert rows[0]["_rerank_score"] > rows[1]["_rerank_score"]
    assert rows[0]["_relevance_score"] > rows[1]["_relevance_score"]
    assert stats["reranked"] == 3


def test_rerank_skips_a_confident_first_stage(monkeypatch):
    monkeypatch.setattr(search, "RERANK_ENABLED", True)
    monkeypatch.setattr(search, "RERANK_ALWAYS", False)

    def fail(*args, **kwargs):
        raise AssertionError("reranker must not be called")

    monkeypatch.setattr(search, "rerank", fail)
    rows = [
        {"id": "answer", "content": "one", "_relevance_score": 0.95},
        {"id": "decoy", "content": "two", "_relevance_score": 0.50},
    ]
    search._rerank_rows(rows, "query", {}, False)

    assert [row["id"] for row in rows] == ["answer", "decoy"]


def test_rerank_failure_keeps_the_first_stage_ranking(monkeypatch):
    monkeypatch.setattr(search, "RERANK_ENABLED", True)
    monkeypatch.setattr(search, "RERANK_ALWAYS", True)

    def fail(*args, **kwargs):
        raise RuntimeError("inference down")

    monkeypatch.setattr(search, "rerank", fail)
    rows = _rows()
    before = [row["id"] for row in rows]
    stats = {}
    search._rerank_rows(rows, "how do we deploy?", stats, False)

    assert [row["id"] for row in rows] == before
    assert "inference down" in stats["rerank_error"]


def test_rerank_rescoring_keeps_the_project_scope_boost(monkeypatch):
    monkeypatch.setattr(search, "RERANK_ENABLED", True)
    monkeypatch.setattr(search, "RERANK_ALWAYS", True)
    monkeypatch.setattr(search, "rerank", lambda query, docs: [0.0, 1.0])

    rows = [
        {"id": "global", "content": "one", "_project": "global",
         "_relevance_score": 0.40},
        {"id": "project", "content": "two", "_project": "api",
         "_relevance_score": 0.40},
    ]
    search._rerank_rows(rows, "query", {}, False)

    assert rows[0]["id"] == "project"
    assert rows[0]["_scope_boost"] > 0
    assert rows[0]["_relevance_score"] > rows[0]["_rerank_score"]


def test_rerank_sigmoid_is_applied_exactly_once():
    """The inference service returns raw logits, so the client sigmoids once.

    Applying the model's default sigmoid in the service AND again here
    squashed every score into 0.5-0.73: an unrelated pair looked like
    "medium confidence" and survived `min_relevance=0.5`.
    """
    neutral, unrelated, relevant = search._sigmoid_scores([0.0, -5.0, 5.0])

    assert neutral == pytest.approx(0.5)
    assert unrelated < 0.01
    assert relevant > 0.99


def test_rerank_keeps_unrelated_candidates_low(monkeypatch):
    monkeypatch.setattr(search, "RERANK_ENABLED", True)
    monkeypatch.setattr(search, "RERANK_ALWAYS", True)
    monkeypatch.setattr(search, "rerank",
                        lambda query, docs: [-5.0, -4.0])

    rows = [
        {"id": "db", "content": "The database runs PostgreSQL 16.",
         "_relevance_score": 0.30, "_rrf_score": 0.02},
        {"id": "deploy", "content": "Deployments happen on Sunday.",
         "_relevance_score": 0.29, "_rrf_score": 0.01},
    ]
    search._rerank_rows(rows, "quantum entanglement in qubits", {}, False)

    assert max(row["_relevance_score"] for row in rows) < 0.05
