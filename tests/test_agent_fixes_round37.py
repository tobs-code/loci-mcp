"""Round-37 fixes: `forget` by metadata filter and content substring."""
from mcp_server.server import forget, list_memories, remember


PROJECT = "round37-forget-selectors"


def _store(content, entity=None, metadata=None):
    result = remember(content, project=PROJECT, entity=entity,
                      metadata=metadata, dedupe="off")
    assert result["ok"] is True, result
    return result["id"]


def test_forget_by_metadata_filter_removes_only_matching_rows():
    keep = _store("Round37 keep this fact", metadata={"run": "keep"})
    drop_a = _store("Round37 drop this fact", metadata={"run": "drop"})
    drop_b = _store("Round37 drop the other fact", metadata={"run": "drop"})

    result = forget(project=PROJECT, metadata_filter={"run": "drop"})

    assert result["ok"] is True
    assert result["status"] == "soft_forgotten"
    assert result["counts"]["memories"] == 2
    assert result["count"] == 2
    assert result["scopes_affected"] == [PROJECT]
    assert {r["id"] for r in result["bulk_results"]} == {drop_a, drop_b}
    assert all(r["content_preview"].startswith("Round37 drop")
               for r in result["bulk_results"])
    assert all(r["reason"] == "metadata_filter"
               for r in result["bulk_results"])

    active = list_memories(project=PROJECT, limit=50)
    assert [row["id"] for row in active["results"]] == [keep]


def test_forget_by_contains_is_literal_and_case_insensitive():
    _store("Round37 Zebra crossing report")
    _store("Round37 plain report")

    result = forget(project=PROJECT, contains="zEbRa")

    assert result["ok"] is True
    assert result["counts"]["memories"] == 1
    assert result["bulk_results"][0]["reason"] == "contains"
    assert "Zebra" in result["bulk_results"][0]["content_preview"]


def test_forget_selector_dry_run_changes_nothing():
    memory_id = _store("Round37 dry run candidate", metadata={"dry": "yes"})

    preview = forget(project=PROJECT, metadata_filter={"dry": "yes"},
                     dry_run=True)

    assert preview["ok"] is True
    assert preview["status"] == "ok"
    assert preview["dry_run"] is True
    assert preview["bulk_results"][0]["status"] == "would_soft_forget"
    assert list_memories(memory_id=memory_id)["results"][0]["id"] == memory_id


def test_forget_selector_hard_delete():
    memory_id = _store("Round37 hard delete me", metadata={"hard": "yes"})

    result = forget(project=PROJECT, metadata_filter={"hard": "yes"},
                    soft=False)

    assert result["status"] == "deleted"
    assert result["bulk_results"][0]["status"] == "deleted"
    missing = list_memories(memory_id=memory_id)
    assert missing["ok"] is False
    assert missing["status"] == "not_found"


def test_forget_selector_requires_explicit_project():
    result = forget(contains="Round37")

    assert result["ok"] is False
    assert "project" in result["error"]


def test_forget_selector_rejects_mixed_selectors():
    memory_id = _store("Round37 exclusive fact", metadata={"x": 1})

    with_id = forget(memory_id=memory_id, project=PROJECT,
                     metadata_filter={"x": 1})
    assert with_id["ok"] is False

    both = forget(project=PROJECT, metadata_filter={"x": 1},
                  contains="Round37")
    assert both["ok"] is False

    empty = forget(project=PROJECT, metadata_filter={})
    assert empty["ok"] is False
    blank = forget(project=PROJECT, contains="   ")
    assert blank["ok"] is False


def test_forget_selector_without_match_answers_ok_with_warning():
    result = forget(project=PROJECT, contains="no-such-round37-token")

    assert result["ok"] is True
    assert result["counts"]["memories"] == 0
    assert result["bulk_results"] == []
    assert any("no memories matched" in w for w in result["warnings"])


def test_forget_selector_global_scope_needs_reserved_name():
    content = "Round37 global selector fact"
    remember(content, scope="global", metadata={"round37": "global"},
             dedupe="off")

    result = forget(project="global", metadata_filter={"round37": "global"})

    assert result["ok"] is True
    assert result["scopes_affected"] == ["global"]
    assert result["counts"]["memories"] >= 1
