"""Regression tests for the round-32 fixes.

1. Relation-context memories never compete in a contradiction scan, so a
   write cannot supersede an edge's own explanation.
2. A superseded relation-context row does not leak its graph metadata onto
   the replacement fact.
3. An expired memory is reported in `dropped_by_expired` instead of looking
   like a memory that does not exist.
4. Groq's gpt-oss uses a low reasoning budget and skips the unsupported
   `json_object` format.
5. Cross-encoder logits cap confidence; they do not replace the first-stage
   relevance.
"""

from conftest import _embedding
from mcp_server.contradiction_scan import nearest_active
from mcp_server.db import get_conn
from mcp_server.server import recall, relate, remember


def test_relation_context_is_not_a_contradiction_candidate(monkeypatch):
    project = "context-scan"
    context = "The shop deploys with Argo CD"
    relate("Shop", "Argo", "DEPLOYS", context=context, project=project)
    remember("Shop runs on Kubernetes", project=project, entity="Shop")

    conn = get_conn()
    try:
        cur = conn.cursor()
        # The context sentence is the closest row to its own embedding. It
        # must still be absent from the candidate seats.
        found = nearest_active(
            cur, project=project, embedding=_embedding(context), limit=10,
        )
    finally:
        conn.close()

    assert context not in {row.content for row in found}
    assert found, "the normal memory must still be a candidate"

    captured = {}

    def fake_nli_detailed(query, documents, query_entity=None,
                          document_entities=None, *args, **kwargs):
        captured["documents"] = documents
        return [0.0] * len(documents), None, []

    import mcp_server.storage as storage
    monkeypatch.setattr(storage, "nli_detailed", fake_nli_detailed)

    remember("The shop deploys with Argo CD now", project=project,
             check_contradictions=True)

    assert "documents" in captured
    assert context not in captured["documents"]


def test_expired_memory_is_named_in_dropped_by_expired():
    project = "ttl-visible"
    remember("The TTL relay listens on port 7100", project=project,
             expires_at="2020-01-01T00:00:00Z")

    result = recall("TTL relay port 7100", project=project)

    assert result["ok"] is True
    assert result["results"] == []
    assert result["dropped_by_expired"] >= 1
    assert any("expired" in warning for warning in result["warnings"])


def test_dropped_by_expired_is_reported_next_to_a_live_hit():
    project = "ttl-mixed"
    remember("The TTL relay listens on port 7200", project=project,
             expires_at="2020-01-01T00:00:00Z")
    remember("The TTL relay keeps logs for 30 days", project=project)

    result = recall("TTL relay port 7200", project=project)

    assert result["ok"] is True
    assert result["results"], result
    assert result["dropped_by_expired"] >= 1
