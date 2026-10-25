"""Round-39 feature: `merge_memories` folds several facts into one."""
import mcp_server.server as server
from mcp_server.server import list_memories, merge_memories, remember


PROJECT = "round39-merge"


def _fact(content, **kwargs):
    result = remember(content, project=PROJECT, dedupe="off", **kwargs)
    assert result["ok"] is True, result
    return result["id"]


def test_merge_supersedes_every_source():
    first = _fact("Round39 API timeout is 30 seconds",
                  metadata={"source": "runbook"})
    second = _fact("Round39 API retries three times",
                   metadata={"owner": "platform"})

    result = merge_memories([first, second],
                            "Round39 API times out after 30 seconds and "
                            "retries three times", project=PROJECT)

    assert result["ok"] is True
    assert result["status"] == "merged"
    assert result["id"]
    assert result["merged_ids"] == [first, second]
    assert result["superseded"] == [first, second]

    merged = list_memories(memory_id=result["id"])
    assert merged["ok"] is True
    assert merged["results"][0]["metadata"]["source"] == "runbook"
    assert merged["results"][0]["metadata"]["owner"] == "platform"

    history = list_memories(project=PROJECT, include_superseded=True,
                            limit=50)
    superseded = {row["id"]: row["superseded_by"] for row in history["results"]
                  if row["id"] in (first, second)}
    assert superseded == {first: result["id"], second: result["id"]}


def test_merge_metadata_replace_and_explicit_wins():
    first = _fact("Round39 first detail", metadata={"a": 1, "b": 1})
    second = _fact("Round39 second detail", metadata={"b": 2})

    merged = merge_memories([first, second], "Round39 combined detail",
                            project=PROJECT, metadata={"c": 3},
                            metadata_mode="merge")
    meta = list_memories(memory_id=merged["id"])["results"][0]["metadata"]
    assert meta == {"a": 1, "b": 2, "c": 3}

    replaced = merge_memories([_fact("Round39 replace source one",
                                     metadata={"x": 1}),
                               _fact("Round39 replace source two",
                                     metadata={"y": 2})],
                              "Round39 replaced detail",
                              project=PROJECT, metadata={"c": 3},
                              metadata_mode="replace")
    meta = list_memories(memory_id=replaced["id"])["results"][0]["metadata"]
    assert meta == {"c": 3}


def test_merge_dry_run_writes_nothing():
    first = _fact("Round39 dry source one")
    second = _fact("Round39 dry source two")

    preview = merge_memories([first, second], "Round39 dry merged fact",
                             project=PROJECT, dry_run=True)

    assert preview["ok"] is True
    assert preview["dry_run"] is True
    assert preview["id"] is None
    assert preview["would_supersede"] == [first, second]
    assert preview["would_create"]["content"] == "Round39 dry merged fact"
    rows = list_memories(project=PROJECT, limit=50)
    assert {row["id"] for row in rows["results"]} >= {first, second}


def test_merge_without_project_uses_the_source_scope():
    first = _fact("Round39 scoped source one")
    second = _fact("Round39 scoped source two")

    result = merge_memories([first, second], "Round39 scoped merged fact")

    assert result["ok"] is True, result
    assert result["project"] == PROJECT
    row = list_memories(memory_id=result["id"])["results"][0]
    assert row["project"] == PROJECT


def test_merge_rejects_mixed_scopes_without_project():
    project_id = _fact("Round39 project source")
    global_id = remember("Round39 global source", scope="global",
                         dedupe="off")["id"]

    result = merge_memories([project_id, global_id],
                            "Round39 impossible merge")

    assert result["ok"] is False
    assert result["status"] == "error"
    assert "several scopes" in result["error"]
    assert set(result["scopes"]) == {PROJECT, "global"}


def test_merge_rolls_back_on_inactive_source():
    first = _fact("Round39 active source")
    second = _fact("Round39 forgotten source")
    assert merge_memories  # keep the import explicit
    from mcp_server.server import forget
    forget(memory_id=second, project=PROJECT)

    result = merge_memories([first, second], "Round39 should not exist",
                            project=PROJECT)

    assert result["ok"] is False
    assert result["status"] == "conflict"
    assert result["invalid"] == {second: "soft_forgotten"}
    assert list_memories(memory_id=first)["ok"] is True


def test_merge_refuses_duplicate_merged_content():
    existing = _fact("Round39 duplicate target text")
    first = _fact("Round39 duplicate source one")
    second = _fact("Round39 duplicate source two")

    result = merge_memories([first, second], "Round39 duplicate target text",
                            project=PROJECT)

    assert result["ok"] is False
    assert result["status"] == "duplicate"
    assert result["duplicate_of"] == existing
    assert list_memories(memory_id=first)["ok"] is True


def test_merge_validates_arguments_and_registration():
    assert merge_memories(["only-one"], "content")["ok"] is False
    assert merge_memories(["a", "b"], "   ")["ok"] is False
    assert merge_memories(["a", "a"], "content")["ok"] is False
    assert merge_memories in server._ALL_TOOLS
    assert "merge_memories" not in server._CORE_TOOL_NAMES
