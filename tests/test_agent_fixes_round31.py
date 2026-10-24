"""Regression tests for the round-31 live-audit fixes.

1. `relate` refuses entity names with the reserved characters `$`/`` ` ``
   (and blank names) like every other graph entry point.
2. Every `remember_batch` and `update_batch` entry carries a `warnings` list,
   also for errors and duplicates, so callers need no key check.
3. An invalid `mem_type` answers the same message in `remember`,
   `remember_batch` and `update_memory`.
"""

from mcp_server.graph import relate
from mcp_server.tools.memory import (
    remember,
    remember_batch,
    update_batch,
    update_memory,
)


def test_relate_refuses_reserved_and_blank_entity_names():
    project = "round31-graph"

    for bad in ("bad$name", "bad`name", "   "):
        result = relate(bad, "R31Target", "USES", project=project)
        assert result["ok"] is False
        assert result["id"] is None
        assert "entity" in result["error"].lower()


def test_remember_batch_every_entry_has_a_warnings_list():
    result = remember_batch(
        project="round31-batch",
        facts=[
            {"content": "R31 stored fact"},
            {"content": ""},
            {"content": "R31 stored fact"},
            {"content": "R31 bad type", "mem_type": "bogus"},
        ],
    )

    assert result["ok"] is True, result.get("error")
    entries = result["results"]
    assert [entry["status"] for entry in entries] == [
        "stored", "error", "duplicate", "error"]
    for entry in entries:
        assert isinstance(entry.get("warnings"), list)
    assert "Invalid mem_type. Choose from:" in entries[3]["error"]


def test_update_batch_every_entry_has_a_warnings_list():
    stored = remember("R31 update target", project="round31-update")
    result = update_batch(
        project="round31-update",
        updates=[
            {"memory_id": stored["id"], "importance": 0.9},
            {"memory_id": "doesnotexist", "importance": 0.9},
            {"memory_id": stored["id"]},
        ],
    )

    assert result["ok"] is True, result.get("error")
    entries = result["results"]
    assert entries[0]["status"] == "updated"
    assert entries[1]["status"] == "not_found"
    assert entries[2]["status"] == "error"
    for entry in entries:
        assert isinstance(entry.get("warnings"), list)


def test_invalid_mem_type_answers_the_same_message_everywhere():
    single = remember("R31 type single", mem_type="bogus")
    batch = remember_batch(
        [{"content": "R31 type batch", "mem_type": "bogus"}])
    updated = update_memory(
        remember("R31 type update")["id"], mem_type="bogus")

    for result in (single, updated):
        assert result["ok"] is False
        assert "Invalid mem_type. Choose from:" in result["error"]
    assert "Invalid mem_type. Choose from:" in batch["results"][0]["error"]
