"""Round-36 fixes: batch entity lookup.

`get_entities` answers one entry per name with the exact `get_entity`
semantics (details, memories, cross-scope dedupe, paging) and never aborts
the batch for an unknown name.
"""
import mcp_server.server as server
from mcp_server.server import get_entities, get_entity, remember


def test_get_entities_mixes_found_and_missing_names():
    remember("Round36 Alpha owns the deploy pipeline", entity="Round36Alpha")
    remember("Round36 Beta runs the nightly backup", entity="Round36Beta")

    result = get_entities(
        ["Round36Alpha", "Round36Missing", "Round36Beta"])

    assert result["ok"] is True
    assert result["status"] == "partial"
    assert result["count"] == 3
    assert result["succeeded"] == 2
    assert result["failed"] == 1
    alpha, missing, beta = result["results"]
    assert alpha["ok"] is True
    assert alpha["entity"] == "Round36Alpha"
    assert alpha["details"]["type"] == "memory-only"
    assert alpha["memories"][0]["content"].startswith("Round36 Alpha")
    assert missing["ok"] is False
    assert missing["status"] == "not_found"
    assert missing["error"]
    assert beta["ok"] is True
    assert any("Round36Missing" in w for w in result["warnings"])


def test_get_entities_removes_duplicate_names_and_matches_get_entity():
    remember("Round36 shared fact about the worker pool", entity="Round36Svc")

    result = get_entities(["Round36Svc", " Round36Svc "])

    assert result["count"] == 1
    entry = result["results"][0]
    single = get_entity("Round36Svc")
    assert entry["total"] == single["total"]
    assert entry["memories"] == single["memories"]
    assert entry["details"]["name"] == "Round36Svc"


def test_get_entities_keeps_cross_scope_dedupe_and_hidden_count():
    project = "round36-entities"
    content = "Round36 duplicate: the cache has four shards"
    remember(content, project=project, entity="Round36Cache")
    remember(content, scope="global", entity="Round36Cache")

    entry = get_entities(["Round36Cache"], project=project)["results"][0]

    assert entry["duplicates_hidden"] == 1
    assert entry["total"] == 1
    assert any("hidden copy" in w for w in entry["warnings"])


def test_get_entities_pages_per_entity():
    for index in range(3):
        remember(f"Round36 paged fact number {index}", entity="Round36Page")

    entry = get_entities(["Round36Page"], limit_per_entity=2,
                         offset_per_entity=1)["results"][0]

    assert entry["total"] == 3
    assert len(entry["memories"]) == 2
    assert entry["has_more"] is False


def test_get_entities_validates_names_and_scope():
    blank = get_entities(["Round36Ok", "   "])
    assert blank["ok"] is False
    assert "blank" in blank["error"]

    star = get_entities(["Round36Ok"], project="*")
    assert star["ok"] is False
    assert "project='*'" in star["error"]

    too_many = get_entities([f"entity-{i}" for i in range(26)])
    assert too_many["ok"] is False
    assert "at most 25" in too_many["error"]


def test_get_entities_is_full_toolset_only():
    assert get_entities in server._ALL_TOOLS
    assert "get_entities" not in server._CORE_TOOL_NAMES
