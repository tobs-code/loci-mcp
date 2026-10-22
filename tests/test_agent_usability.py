"""Contract tests for the remaining agent-usability improvements:
dedupe modes, supersede by content/last, unified entity parameter,
typed metadata."""

from mcp_server.server import (
    find_connections,
    forget,
    get_entity,
    recall,
    remember,
    remember_batch,
)


def test_dedupe_semantic_rejects_a_paraphrase():
    first = remember("Dedupe probe: the invoice is paid monthly")["id"]

    result = remember("Dedupe probe: the invoice is paid monthly.",
                      dedupe="semantic")

    assert result["status"] == "duplicate"
    assert result["id"] == first


def test_dedupe_exact_is_the_default_and_stores_paraphrases():
    remember("Dedupe default probe: the invoice is paid monthly")

    result = remember("Dedupe default probe: the invoice is paid monthly.")

    assert result["status"] == "stored"


def test_dedupe_off_stores_identical_content_again():
    remember("Dedupe off probe: identical content")

    result = remember("Dedupe off probe: identical content", dedupe="off")

    assert result["status"] == "stored"


def test_dedupe_rejects_an_unknown_mode():
    result = remember("Dedupe mode probe", dedupe="sideways")

    assert result["ok"] is False
    assert "dedupe" in result["error"]


def test_supersede_content_replaces_the_fact_with_the_same_text():
    old_id = remember("Supersede probe: the API key rotates weekly")["id"]

    result = remember("Supersede probe: the API key rotates daily",
                      supersede_content="Supersede probe: the API key rotates weekly")

    assert result["ok"] is True
    assert result["superseded"] == [old_id]
    history = recall("API key rotates", include_superseded=True, limit=10)["results"]
    old = next(r for r in history if r["id"] == old_id)
    assert old["superseded_by"] == result["id"]


def test_supersede_last_replaces_the_most_similar_fact_of_the_entity():
    first = remember("Supersede last probe: the build runs on friday",
                     entity="Build")["id"]
    second = remember("Supersede last probe: the build runs on saturday",
                      entity="Build")["id"]

    result = remember("Supersede last probe: the build runs on sunday",
                      entity="Build", supersede_last=True)

    assert result["superseded"] == [second]
    history = recall("build runs on", include_superseded=True, limit=10)["results"]
    assert {r["id"] for r in history} == {first, second, result["id"]}


def test_find_connections_accepts_the_entity_parameter():
    from mcp_server.server import relate

    relate("Alias", "Target", "KNOWS")

    result = find_connections(entity="Alias")

    assert result["ok"] is True
    assert result["results"][0]["entity"] == "Target"


def test_forget_accepts_the_entity_parameter():
    remember("Alias forget probe", entity="AliasTarget")

    result = forget(entity="AliasTarget")

    assert result["ok"] is True
    assert result["status"] == "soft_forgotten"


def test_get_entity_accepts_the_entity_parameter():
    remember("Alias get probe", entity="AliasGet")

    result = get_entity(entity="AliasGet")

    assert result["ok"] is True
    assert result["entity"] == "AliasGet"


def test_metadata_keeps_its_value_types():
    remember("Typed metadata probe", metadata={"flag": True, "count": 42,
                                                "note": "text"})

    hits = recall("Typed metadata probe", limit=5)["results"]
    hit = hits[0]
    assert hit["metadata"]["flag"] is True
    assert hit["metadata"]["count"] == 42
    assert hit["metadata"]["note"] == "text"


def test_remember_batch_honours_the_dedupe_mode():
    remember("Batch dedupe probe: the report is weekly")

    result = remember_batch([
        {"content": "Batch dedupe probe: the report is weekly"},
    ], dedupe="semantic")

    assert result["results"][0]["status"] == "duplicate"
