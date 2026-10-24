"""Tests for the agent-usability fixes from the live tool audit.

Covers:
- remember/remember_batch/contradictions_report survive an NLI outage and
  say so instead of presenting zeros as a clean scan.
- inference_client.nli rejects an error payload instead of reading zeros.
- a PARTIAL NLI failure on a write is never read as "no contradiction"
  (the write-side check uses nli_detailed and raises on a partial payload).
- exact-hash short-circuit visible in the duplicate message (semantic dedupe).
- list_memories exposes a machine-readable `forgotten` flag and hides the
  '__forgotten__' sentinel from superseded_by.
- get_entity returns memories newest-update first.
- update_memory names which arguments would make the update meaningful.
"""

import pytest

from mcp_server.tools.maintenance import contradictions_report
from mcp_server.tools.memory import (
    remember,
    remember_batch,
    update_memory,
)
from mcp_server.tools.read import get_entity, list_memories


def _raise_nli(*args, **kwargs):
    raise RuntimeError("NLI unavailable: provider down")


def test_remember_survives_nli_outage_and_warns(monkeypatch):
    remember(content="Seed fact about the gateway.",
             entity="Gateway", project="nli-outage")
    monkeypatch.setattr("mcp_server.storage.nli_detailed", _raise_nli)
    result = remember(
        content="The gateway deploys to staging on Fridays.",
        entity="Gateway",
        project="nli-outage",
        check_contradictions=True,
    )
    assert result["ok"] is True
    assert result["status"] == "stored"
    warnings = " ".join(result["warnings"])
    assert "contradiction check skipped" in warnings
    assert "NLI unavailable" in warnings
    assert result["contradictions"]["skipped"].startswith("NLI unavailable:")


def test_remember_batch_survives_nli_outage_and_warns(monkeypatch):
    remember(content="Seed fact about the batch gateway.",
             entity="Gateway", project="nli-outage-batch")
    monkeypatch.setattr("mcp_server.storage.nli_detailed", _raise_nli)
    result = remember_batch(
        facts=[{
            "content": "The gateway deploys to staging on Fridays.",
            "entity": "Gateway",
        }],
        project="nli-outage-batch",
        check_contradictions=True,
    )
    assert result["ok"] is True
    entry = result["results"][0]
    assert entry["status"] == "stored"
    assert entry["contradictions"]["skipped"].startswith("NLI unavailable:")
    assert any("contradiction check skipped" in w
               for w in entry.get("warnings", []))


def test_contradictions_report_aggregates_partial_nli_warnings(monkeypatch):
    # A per-fact NLI call produces one partial_error each ("1 of N"); the
    # report used to repeat the nearly identical lines. Now it aggregates.
    entity = "PartialEntity"
    remember(content=f"{entity} batches exports on the first host.",
             entity=entity, project="nli-partial-report")
    remember(content=f"{entity} batches exports on the second host.",
             entity=entity, project="nli-partial-report")
    remember(content=f"{entity} batches exports on the third host.",
             entity=entity, project="nli-partial-report")
    monkeypatch.setattr(
        "mcp_server.tools.maintenance.nli_detailed",
        lambda q, d, *a, **kw: (
            [0.0] * len(d),
            f"NLI unavailable for 1 of {len(d)} document(s): bad JSON",
            [0]),
    )
    result = contradictions_report(entity=entity,
                                   project="nli-partial-report")
    assert result["ok"] is True
    partial_lines = [w for w in result["warnings"]
                     if w.startswith("scan partially incomplete")]
    assert len(partial_lines) == 1, partial_lines
    assert "in total" in partial_lines[0]
    # The per-call format ("1 of 2") is gone: only the aggregated totals name
    # the documents, so partial warnings never repeat per fact.
    assert "1 of" not in partial_lines[0]


def test_contradictions_report_survives_nli_outage(monkeypatch):
    # Two facts on one entity: the vector part finds candidate pairs, the
    # NLI leg raises - the tool must return an envelope with a warning.
    entity = "OutageEntity"
    remember(content=f"{entity} stores reports on port 9001.",
             entity=entity, project="nli-outage-report")
    remember(content=f"{entity} stores reports on port 9002.",
             entity=entity, project="nli-outage-report")
    monkeypatch.setattr("mcp_server.tools.maintenance.nli_detailed", _raise_nli)
    result = contradictions_report(entity=entity, project="nli-outage-report")
    assert result["ok"] is True
    warnings = " ".join(result["warnings"])
    assert "scan incomplete" in warnings
    assert "not a clean contradiction scan" in warnings


def test_inference_client_nli_raises_on_error_payload(monkeypatch):
    import mcp_server.inference_client as ic

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"contradiction_scores": [0.0],
                    "error": "NLI unavailable: provider down"}

    class FakeSession:
        def post(self, *a, **kw):
            return FakeResponse()

    monkeypatch.setattr(ic, "_get_session", lambda: FakeSession())
    with pytest.raises(RuntimeError, match="NLI unavailable"):
        ic.nli("A", ["B"])


def test_semantic_dedupe_duplicate_message_names_the_hash_shortcut():
    stored = remember(content="Hash shortcut note for semantic dedupe.",
                      project="dedupe-msg")
    assert stored["ok"] is True
    duplicate = remember(content="Hash shortcut note for semantic dedupe.",
                         project="dedupe-msg", dedupe="semantic")
    assert duplicate["status"] == "duplicate"
    assert "the content hash short-circuits the semantic check" in \
        duplicate["warnings"][0]


def test_list_memories_marks_forgotten_rows_and_hides_the_sentinel():
    stored = remember(content="Forgotten marker fact.", project="forgotten-flag")
    from mcp_server.tools.memory import forget  # deferred: module-level OK too
    forsaken = forget(memory_id=stored["id"], project="forgotten-flag")
    assert forsaken["status"] == "soft_forgotten"

    listed = list_memories(project="forgotten-flag", contains="Forgotten marker",
                           include_deleted=True)
    row = listed["results"][0]
    assert row["deleted_at"] is not None
    assert row["forgotten"] is True
    assert row["superseded_by"] is None


def test_get_entity_sorts_memories_newest_update_first():
    remember(content="Old fact about sort entity.", entity="SortEntity",
             project="entity-order", importance=0.5)
    second = remember(content="Newer fact about sort entity.",
                      entity="SortEntity", project="entity-order")
    result = update_memory(memory_id=second["id"], project="entity-order",
                           importance=0.7)
    assert result["ok"] is True

    entity = get_entity(entity="SortEntity", project="entity-order")
    memories = [m["content"] for m in entity["memories"]]
    # The memory updated last (the new one) must come first.
    assert memories[0] == "Newer fact about sort entity."
    # Both rows are there and the order is a true updated_at descending.
    assert len(memories) == 2


def test_update_memory_says_what_is_missing():
    stored = remember(content="Empty-update message fact.",
                      project="update-msg")
    result = update_memory(memory_id=stored["id"], project="update-msg")
    assert result["ok"] is False
    assert "nothing to update: pass content, importance, metadata," in \
        result["error"]


def test_update_memory_answers_not_found_before_the_validation():
    # An unknown id must read as "the memory does not exist", not as "you
    # forgot to pass something": the existence check runs first now.
    result = update_memory(memory_id="zzunknownid00")

    assert result["ok"] is False
    assert result["status"] == "not_found"
    assert "NOT FOUND" in result["error"]


def test_update_memory_answers_not_found_with_fields_passsed():
    result = update_memory(memory_id="zzunknownid01", importance=0.5)

    assert result["ok"] is False
    assert result["status"] == "not_found"
    assert "NOT FOUND" in result["error"]


def test_nli_detailed_returns_partial_scores_and_keeps_the_warning(monkeypatch):
    import mcp_server.inference_client as ic

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"contradiction_scores": [0.95, 0.0],
                    "partial_error": "NLI unavailable for 1 of 2 document(s): bad JSON",
                    "failed": 1}

    class FakeSession:
        def post(self, *args, **kwargs):
            return FakeResponse()

    monkeypatch.setattr(ic, "_get_session", lambda: FakeSession())
    scores, partial, failed = ic.nli_detailed("A", ["B", "C"])
    assert scores == [0.95, 0.0]
    assert "1 of 2" in partial
    assert failed == []
    # The plain wrapper keeps its list contract for existing callers.
    assert ic.nli("A", ["B", "C"]) == [0.95, 0.0]


def test_server_health_reports_the_running_code():
    from mcp_server.tools.maintenance import server_health

    result = server_health()
    assert result["ok"] is True
    assert result["version"]
    assert len(result["code_fingerprint"]) == 12
    assert result["started_at"]


def test_list_memories_accepts_forgotten_in_fields():
    from mcp_server.tools.memory import forget

    stored = remember(content="Forgotten fields probe.",
                      project="forgotten-fields")
    active = list_memories(project="forgotten-fields",
                           contains="Forgotten fields probe.",
                           fields=["id", "forgotten"])
    assert active["results"][0]["forgotten"] is False

    forget(memory_id=stored["id"], project="forgotten-fields")
    deleted = list_memories(project="forgotten-fields",
                            contains="Forgotten fields probe.",
                            include_deleted=True,
                            fields=["id", "forgotten"])
    assert deleted["results"][0]["forgotten"] is True
