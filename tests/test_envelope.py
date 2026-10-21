"""Contract tests for the uniform tool envelope.

Every tool answers with the same shape so an agent never has to guess:
  {"ok": bool, "status": str, "id": str|None, "error": str|None,
   "warnings": list}
plus, for tools that return a collection:
  {"count": int, "results": [...]}
"""

from mcp_server.server import forget, remember, remember_batch, update_memory


def test_remember_answers_with_the_envelope():
    result = remember("Envelope fact about a purple bicycle")

    assert result["ok"] is True
    assert result["status"] == "stored"
    assert result["error"] is None
    assert result["warnings"] == []
    assert len(result["id"]) == 16


def test_remember_reports_duplicates_as_a_status_not_a_string():
    first = remember("Envelope duplicate probe")

    result = remember("Envelope duplicate probe")

    assert result["ok"] is True
    assert result["status"] == "duplicate"
    assert result["id"] == first["id"]
    assert result["error"] is None


def test_remember_reports_invalid_input_as_a_failed_envelope():
    result = remember("x", mem_type="bogus")

    assert result["ok"] is False
    assert result["status"] == "error"
    assert result["id"] is None
    assert "Invalid mem_type" in result["error"]


def test_remember_batch_answers_with_one_envelope_and_per_fact_results():
    result = remember_batch([
        {"content": "Envelope batch fact one"},
        {"content": ""},
        {"content": "Envelope batch fact two"},
    ])

    assert result["ok"] is True
    assert result["count"] == 3
    assert result["error"] is None
    assert [r["status"] for r in result["results"]] == ["stored", "error", "stored"]
    assert result["results"][1]["error"] == "content is required"
    assert result["results"][0]["id"]


def test_remember_batch_reports_a_failed_batch_as_a_failed_envelope():
    result = remember_batch([{"content": ""}])

    assert result["ok"] is False
    assert result["status"] == "error"
    assert result["error"]


def test_update_memory_answers_with_the_envelope():
    memory_id = remember("Envelope update target")["id"]

    result = update_memory(memory_id, importance=0.5)

    assert result["ok"] is True
    assert result["status"] == "updated"
    assert result["id"] == memory_id


def test_update_memory_reports_an_unknown_id_as_not_found():
    result = update_memory("does-not-exist", importance=0.2)

    assert result["ok"] is False
    assert result["status"] == "not_found"
    assert result["error"] == "NOT FOUND: does-not-exist"


def test_forget_answers_with_the_envelope_and_itemized_counts():
    memory_id = remember("Envelope forget target")["id"]

    result = forget(memory_id)

    assert result["ok"] is True
    assert result["status"] == "soft_forgotten"
    assert result["id"] == memory_id
    assert result["counts"]["memories"] == 1
    assert result["counts"]["entities"] == 0
    assert result["counts"]["relations"] == 0


def test_forget_reports_an_unknown_id_as_not_found():
    result = forget("missing-id")

    assert result["ok"] is False
    assert result["status"] == "not_found"
    assert result["error"] == "NOT FOUND: missing-id"
