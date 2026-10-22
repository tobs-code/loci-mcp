"""Fix round: NLI false positives across entities.

1. The NLI prompt no longer presupposes that both facts describe the same
   subject: differing attribute values (ports, hosts, ...) of two DIFFERENT
   subjects are neutral. The old prompt judged "staging API on port 9443"
   vs "inference service on localhost:8001" as a 0.95 contradiction.
2. The /nli contract carries optional entity labels, and both contradiction
   paths send them, so the model knows which subject each fact belongs to.
3. `contradictions_report(resolve="newest")` only auto-resolves pairs of
   the SAME entity (or both untagged): cross-entity pairs stay active and
   are flagged with `cross_entity: true` plus a warning.
"""
from inference.llm import nli_user_message
from mcp_server import inference_client
from mcp_server.server import (
    contradictions_report,
    recall,
    remember,
)


# --------------------------------------------------------------------------
# The NLI prompt receives and uses the entity context
# --------------------------------------------------------------------------

def test_nli_user_message_annotates_entities_and_marks_a_mismatch():
    message = nli_user_message(
        "The API staging domain is api-staging.example.com on port 9443",
        "The inference service runs on localhost:8001",
        query_entity="API", document_entity="Inference",
    )

    assert "[entity: API]" in message
    assert "[entity: Inference]" in message
    assert "different subjects" in message


def test_nli_user_message_without_entity_labels_is_the_plain_facts():
    message = nli_user_message("Fact A text", "Fact B text")

    assert message == "Fact A: Fact A text\nFact B: Fact B text"


def test_nli_user_message_same_entity_adds_no_mismatch_warning():
    message = nli_user_message("A text", "B text",
                               query_entity="API", document_entity="API")

    assert "different subjects" not in message


def test_nli_detailed_forwards_the_entity_labels(monkeypatch):
    captured = {}

    class _Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"contradiction_scores": [0.0]}

    class _Session:
        def post(self, url, json=None, timeout=None):
            captured.update(json)
            return _Response()

    monkeypatch.setattr(inference_client, "_get_session", lambda: _Session())

    scores, partial = inference_client.nli_detailed(
        "Fact A", ["Fact B"],
        query_entity="API", document_entities=["PostgreSQL"],
    )

    assert scores == [0.0]
    assert partial is None
    assert captured == {
        "query": "Fact A",
        "documents": ["Fact B"],
        "query_entity": "API",
        "document_entities": ["PostgreSQL"],
    }


def test_check_contradictions_sends_the_fact_entities(monkeypatch):
    import mcp_server.storage as storage

    captured = {}

    def fake_nli(query, documents, query_entity=None, document_entities=None,
                 *args, **kwargs):
        captured.update(
            query=query, documents=documents,
            query_entity=query_entity, document_entities=document_entities,
        )
        return [0.0] * len(documents)

    monkeypatch.setattr(storage, "nli", fake_nli)

    remember("The resolve relay answers on port 5100", entity="Relay")
    remember("The resolve relay answers on port 5200", entity="Relay",
             check_contradictions=True)

    assert captured["query_entity"] == "Relay"
    assert captured["document_entities"] == ["Relay"]


def test_report_sends_the_fact_entities_to_nli(monkeypatch):
    remember("The report relay answers on port 5300", entity="Relay")
    remember("The report relay answers on port 5400", entity="Relay")

    captured = {}

    def fake_nli_detailed(query, documents, query_entity=None,
                          document_entities=None, *args, **kwargs):
        captured.update(
            query=query, documents=documents,
            query_entity=query_entity, document_entities=document_entities,
        )
        return [0.0] * len(documents), None

    import mcp_server.tools.maintenance as maintenance
    monkeypatch.setattr(maintenance, "nli_detailed", fake_nli_detailed)

    contradictions_report(entity="Relay")

    assert captured["query_entity"] == "Relay"
    assert captured["document_entities"] == ["Relay"]


# --------------------------------------------------------------------------
# resolve="newest" is safe: no recency-based winner across entities
# --------------------------------------------------------------------------

def test_cross_entity_pairs_are_flagged_and_never_auto_resolved():
    older = remember("The cross relay answers on port 6100 in staging",
                     entity="API")["id"]
    newer = remember("The cross relay does not answer on port 6100 in "
                     "staging", entity="Inference")["id"]

    report = contradictions_report(resolve="newest")

    pair = report["results"][0]
    assert {pair["a"]["id"], pair["b"]["id"]} == {older, newer}
    assert pair["cross_entity"] is True
    assert pair.get("resolution") is None
    assert report["resolved"] == 0
    assert any("resolve skipped" in w for w in report["warnings"])

    # Both facts stay active: the report may not pick a winner it cannot
    # judge (the newer one is NOT necessarily the correct one here).
    active = {r["id"] for r in
              recall("cross relay port 6100", limit=10)["results"]}
    assert {older, newer} <= active


def test_untagged_pairs_are_still_resolved_by_newest():
    older = remember("The untagged relay redeploys at midnight on friday")
    newer = remember("The untagged relay does not redeploy at midnight on "
                     "friday")

    report = contradictions_report(resolve="newest")

    pair = report["results"][0]
    assert pair["cross_entity"] is False
    assert report["resolved"] == 1
    assert pair["resolution"] == {
        "superseded": older["id"], "kept": newer["id"],
    }


def test_report_pairs_carry_the_entity_names():
    remember("The named relay answers on port 6300", entity="API")
    remember("The named relay does not answer on port 6300", entity="API")

    report = contradictions_report(entity="API")

    pair = report["results"][0]
    assert pair["a"]["entity"] == "API"
    assert pair["b"]["entity"] == "API"
    assert pair["cross_entity"] is False
