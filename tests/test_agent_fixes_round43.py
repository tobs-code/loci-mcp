"""Round-43 features: temporal validity, provenance/confidence and tags.

Three data-layer additions that every read path has to honour:
- `valid_from`/`valid_to` give a fact a validity window; `as_of` reads it at a
  point in time and reports `dropped_by_validity` when the only match is out of
  the window.
- `source`/`agent_id`/`confidence` record where a fact came from and how sure
  the writer was.
- `tags` are first-class (normalized, filtered with AND/OR) instead of a
  convention inside `metadata`.
"""

import json

from mcp_server.server import (
    export_memories,
    forget,
    import_memories,
    list_memories,
    recall,
    remember,
    remember_batch,
    update_batch,
    update_memory,
)

PROJECT = "round43-features"


def _fields(memory_id: str, project: str = PROJECT) -> dict:
    result = list_memories(
        project=project, memory_id=memory_id,
        fields=["id", "content", "tags", "source", "agent_id", "confidence",
                "valid_from", "valid_to"])
    assert result["ok"] is True, result
    return result["memory"]


# ---------------------------------------------------------------------------
# Temporal validity
# ---------------------------------------------------------------------------

def test_as_of_reads_the_fact_at_a_point_in_time():
    stored = remember("Round43 validity probe: the deploy window is 02:00",
                      project=PROJECT,
                      valid_from="2020-01-01T00:00:00+00:00",
                      valid_to="2022-01-01T00:00:00+00:00")
    assert stored["ok"] is True

    inside = recall("Round43 validity probe deploy window", project=PROJECT,
                    as_of="2021-06-01T00:00:00+00:00")
    assert [r["id"] for r in inside["results"]] == [stored["id"]]

    after = recall("Round43 validity probe deploy window", project=PROJECT,
                   as_of="2023-06-01T00:00:00+00:00")
    assert after["results"] == []
    assert after["dropped_by_validity"] >= 1


def test_as_of_before_valid_from_does_not_see_the_fact():
    remember("Round43 backdate probe: the region is eu-west", project=PROJECT,
             valid_from="2024-01-01T00:00:00+00:00")

    before = recall("Round43 backdate probe region", project=PROJECT,
                    as_of="2023-01-01T00:00:00+00:00")

    assert before["results"] == []
    assert before["dropped_by_validity"] >= 1
    assert "dropped_by_validity" not in recall(
        "Round43 backdate probe region", project=PROJECT)


def test_supersede_closes_the_validity_window_of_the_old_row():
    old = remember("Round43 supersede validity: the port is 8000",
                   project=PROJECT, valid_from="2020-01-01T00:00:00+00:00")
    assert old["ok"]

    new = remember("Round43 supersede validity: the port is 9000",
                   project=PROJECT, supersede=old["id"],
                   valid_from="2024-01-01T00:00:00+00:00")
    assert new["ok"] and new["status"] == "stored"

    row = list_memories(project=PROJECT, memory_id=old["id"],
                        include_superseded=True)["memory"]
    assert row["valid_to"] is not None


def test_clear_valid_to_removes_the_end_of_the_window():
    stored = remember("Round43 clear validity: the contract ends in March",
                      project=PROJECT,
                      valid_from="2020-01-01T00:00:00+00:00",
                      valid_to="2021-01-01T00:00:00+00:00")
    assert stored["ok"]

    cleared = update_memory(stored["id"], project=PROJECT, clear_valid_to=True)
    assert cleared["ok"] is True

    row = _fields(stored["id"])
    assert row["valid_to"] is None
    assert row["valid_from"] is not None


def test_invalid_validity_window_is_an_argument_error():
    result = remember("Round43 broken validity", project=PROJECT,
                      valid_from="2022-01-01T00:00:00+00:00",
                      valid_to="2021-01-01T00:00:00+00:00")

    assert result["ok"] is False
    assert "valid_to must not be earlier than valid_from" in result["error"]

    broken = remember("Round43 broken timestamp", project=PROJECT,
                      valid_from="not-a-date")
    assert broken["ok"] is False
    assert "valid_from" in broken["error"]


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def test_provenance_round_trips_through_writes_and_reads():
    stored = remember("Round43 provenance probe: latency is 40ms",
                      project=PROJECT, source=" benchmark ",
                      agent_id="agent-7", confidence=0.9)
    assert stored["ok"] is True

    row = _fields(stored["id"])
    assert row["source"] == "benchmark"
    assert row["agent_id"] == "agent-7"
    assert row["confidence"] == 0.9


def test_confidence_out_of_range_is_refused():
    result = remember("Round43 bad confidence", project=PROJECT,
                      confidence=1.5)

    assert result["ok"] is False
    assert "confidence" in result["error"]


def test_provenance_filters_narrow_recall():
    remember("Round43 source filter probe: cache is warm", project=PROJECT,
             source="runbook", confidence=0.95)
    remember("Round43 source filter probe: cache is cold", project=PROJECT,
             source="guess", confidence=0.2)

    by_source = recall("Round43 source filter probe cache", project=PROJECT,
                       source="runbook")
    assert [r["content"] for r in by_source["results"]] == [
        "Round43 source filter probe: cache is warm"]

    by_confidence = recall("Round43 source filter probe cache", project=PROJECT,
                           min_confidence=0.9)
    assert [r["content"] for r in by_confidence["results"]] == [
        "Round43 source filter probe: cache is warm"]


def test_update_memory_can_clear_provenance():
    stored = remember("Round43 clear provenance probe", project=PROJECT,
                      source="somewhere", agent_id="agent-1", confidence=0.4)
    assert stored["ok"]

    updated = update_memory(stored["id"], project=PROJECT, source="",
                            agent_id="", confidence=0.8)
    assert updated["ok"] is True

    row = _fields(stored["id"])
    assert row["source"] is None
    assert row["agent_id"] is None
    assert row["confidence"] == 0.8


# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------

def test_tags_are_normalized_and_deduplicated():
    stored = remember("Round43 tag normalize probe", project=PROJECT,
                      tags=[" DB ", "db", "Prod"])
    assert stored["ok"] is True

    row = _fields(stored["id"])
    assert row["tags"] == ["db", "prod"]


def test_tag_filters_use_and_or_semantics():
    remember("Round43 tag filter probe: alpha", project=PROJECT,
             tags=["db", "prod"])
    remember("Round43 tag filter probe: beta", project=PROJECT, tags=["db"])
    remember("Round43 tag filter probe: gamma", project=PROJECT,
             tags=["staging"])

    both = recall("Round43 tag filter probe", project=PROJECT,
                  tags=["db", "prod"])
    assert [r["content"] for r in both["results"]] == [
        "Round43 tag filter probe: alpha"]

    either = recall("Round43 tag filter probe", project=PROJECT,
                    tags_any=["prod", "staging"])
    assert sorted(r["content"] for r in either["results"]) == [
        "Round43 tag filter probe: alpha",
        "Round43 tag filter probe: gamma"]


def test_tags_and_tags_any_are_mutually_exclusive():
    result = recall("Round43 tag filter probe", project=PROJECT,
                    tags=["db"], tags_any=["prod"])

    assert result["ok"] is False
    assert "tags and tags_any are mutually exclusive" in result["error"]


def test_update_memory_merges_and_replaces_tags():
    stored = remember("Round43 tag update probe", project=PROJECT,
                      tags=["db"])
    assert stored["ok"]

    merged = update_memory(stored["id"], project=PROJECT, tags=["prod"],
                           tags_mode="merge")
    assert merged["ok"] is True
    assert _fields(stored["id"])["tags"] == ["db", "prod"]

    replaced = update_memory(stored["id"], project=PROJECT, tags=["only"],
                             tags_mode="replace")
    assert replaced["ok"] is True
    assert _fields(stored["id"])["tags"] == ["only"]

    cleared = update_memory(stored["id"], project=PROJECT, clear_tags=True)
    assert cleared["ok"] is True
    assert _fields(stored["id"])["tags"] == []


def test_too_many_tags_is_an_argument_error():
    result = remember("Round43 too many tags", project=PROJECT,
                      tags=[f"tag-{i}" for i in range(33)])

    assert result["ok"] is False
    assert "at most 32 tags" in result["error"]


def test_batch_writes_carry_provenance_and_tags():
    result = remember_batch(
        facts=[
            {"content": "Round43 batch feature one", "tags": ["db"],
             "source": "batch", "confidence": 0.7},
            {"content": "Round43 batch feature two", "tags": ["prod"]},
        ], project=PROJECT)
    assert result["ok"] is True

    rows = list_memories(project=PROJECT, contains="Round43 batch feature")
    by_content = {r["content"]: r for r in rows["results"]}
    assert by_content["Round43 batch feature one"]["tags"] == ["db"]
    assert by_content["Round43 batch feature one"]["source"] == "batch"
    assert by_content["Round43 batch feature one"]["confidence"] == 0.7

    updated = update_batch(updates=[
        {"memory_id": by_content["Round43 batch feature two"]["id"],
         "tags": ["release"], "tags_mode": "merge"},
    ], project=PROJECT)
    assert updated["ok"] is True
    row = _fields(by_content["Round43 batch feature two"]["id"])
    assert row["tags"] == ["prod", "release"]


# ---------------------------------------------------------------------------
# Export / import (Bug 4 + the new fields)
# ---------------------------------------------------------------------------

def test_export_uses_one_mem_type_key_and_version_0_4():
    forget(entity="*", project=PROJECT, soft=False, dry_run=False)
    remember("Round43 export single name", project=PROJECT,
             mem_type="decision", tags=["backup"], source="doc",
             confidence=0.8, valid_from="2021-01-01T00:00:00+00:00")

    payload = export_memories(project=PROJECT)["data"]

    assert payload["version"] == "0.4"
    record = payload["records"][0]
    assert record["mem_type"] == "decision"
    assert "type" not in record
    assert record["tags"] == ["backup"]
    assert record["source"] == "doc"
    assert record["confidence"] == 0.8
    assert record["valid_from"].startswith("2021-01-01")


def test_jsonl_export_import_preserves_the_new_fields():
    forget(entity="*", project="round43-export", soft=False, dry_run=False)
    forget(entity="*", project="round43-import", soft=False, dry_run=False)
    stored = remember("Round43 jsonl roundtrip probe", project="round43-export",
                      tags=["db", "prod"], source="runbook",
                      agent_id="agent-3", confidence=0.75,
                      valid_from="2020-05-01T00:00:00+00:00")
    assert stored["ok"]

    exported = export_memories(project="round43-export", format="jsonl")
    assert exported["ok"] is True

    restored = import_memories(exported["data"], format="jsonl",
                               project="round43-import")
    assert restored["ok"] is True
    assert restored["imported"] == 1

    row = _fields(stored["id"], project="round43-import")
    assert row["tags"] == ["db", "prod"]
    assert row["source"] == "runbook"
    assert row["agent_id"] == "agent-3"
    assert row["confidence"] == 0.75
    assert str(row["valid_from"]).startswith("2020-05-01")


def test_export_filters_narrow_the_backup():
    forget(entity="*", project=PROJECT, soft=False, dry_run=False)
    remember("Round43 export filter one", project=PROJECT, tags=["keep"])
    remember("Round43 export filter two", project=PROJECT, tags=["drop"])

    payload = export_memories(project=PROJECT, tags=["keep"])["data"]

    assert payload["count"] == 1
    assert payload["records"][0]["content"] == "Round43 export filter one"
