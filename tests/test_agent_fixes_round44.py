"""Round-44 features: `pinned` (always-include) memories + Bug 3.

WS3 adds a `pinned` flag: a pinned fact is never dropped by a relevance,
similarity or decay floor. `recall(include_pinned=True)` backfills a pinned
fact onto the page, `compile_context` leads with the pins, and
`list_memories(order_by="pinned")` lists them first. The flag round-trips
through `remember`/`remember_batch`, `update_memory`/`update_batch` and the
backup.

Bug 3: the exact-match path used to overwrite `_relevance_score` with a
constant (`0.95`/`0.85`), so two exact hits of very different similarity
reported the same score. The literal signal now only sets `_exact_match` and
leaves the genuine vector+FTS blend in place.
"""

from mcp_server.server import (
    compile_context,
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

PROJECT = "round44-pinned"


def _fields(memory_id: str, project: str = PROJECT) -> dict:
    result = list_memories(
        project=project, memory_id=memory_id,
        fields=["id", "content", "pinned", "entity"])
    assert result["ok"] is True, result
    return result["memory"]


# ---------------------------------------------------------------------------
# Storing and editing the flag
# ---------------------------------------------------------------------------

def test_remember_stores_pinned_and_default_is_false():
    forget(entity="*", project=PROJECT, soft=False, dry_run=False)
    pinned = remember("Round44 pin probe: the canary token is ALPHA",
                      project=PROJECT, pinned=True)
    plain = remember("Round44 plain probe: the canary token is BETA",
                     project=PROJECT)
    assert pinned["ok"] and plain["ok"]

    assert _fields(pinned["id"])["pinned"] is True
    assert _fields(plain["id"])["pinned"] is False


def test_batch_writes_carry_pinned_per_fact():
    forget(entity="*", project="round44-batch", soft=False, dry_run=False)
    result = remember_batch(
        facts=[
            {"content": "Round44 batch pinned fact", "pinned": True},
            {"content": "Round44 batch plain fact"},
        ], project="round44-batch")
    assert result["ok"] is True

    rows = list_memories(project="round44-batch", contains="Round44 batch")
    by_content = {r["content"]: r for r in rows["results"]}
    assert by_content["Round44 batch pinned fact"]["pinned"] is True
    assert by_content["Round44 batch plain fact"]["pinned"] is False


def test_update_memory_toggles_the_flag():
    forget(entity="*", project=PROJECT, soft=False, dry_run=False)
    stored = remember("Round44 toggle probe", project=PROJECT)
    assert stored["ok"]

    on = update_memory(stored["id"], project=PROJECT, pinned=True)
    assert on["ok"] is True
    assert _fields(stored["id"])["pinned"] is True

    off = update_memory(stored["id"], project=PROJECT, pinned=False)
    assert off["ok"] is True
    assert _fields(stored["id"])["pinned"] is False


def test_update_batch_carries_pinned():
    forget(entity="*", project="round44-batch", soft=False, dry_run=False)
    stored = remember("Round44 batch update probe", project="round44-batch")
    assert stored["ok"]

    result = update_batch(
        updates=[{"memory_id": stored["id"], "pinned": True}],
        project="round44-batch")
    assert result["ok"] is True
    assert _fields(stored["id"], project="round44-batch")["pinned"] is True


# ---------------------------------------------------------------------------
# recall: the pin survives the relevance floors
# ---------------------------------------------------------------------------

def test_recall_include_pinned_backfills_an_unrelated_fact():
    forget(entity="*", project=PROJECT, soft=False, dry_run=False)
    pin = remember("Round44 pin: the build server is BUILD-7",
                   project=PROJECT, pinned=True)
    remember("Round44 ranked: the deploy command is DEPLOY-NOW",
             project=PROJECT)
    assert pin["ok"]

    ranked = recall("Round44 ranked deploy command", project=PROJECT)
    assert pin["id"] not in [r["id"] for r in ranked["results"]]

    with_pins = recall("Round44 ranked deploy command", project=PROJECT,
                       include_pinned=True)
    ids = [r["id"] for r in with_pins["results"]]
    assert pin["id"] in ids
    assert with_pins["pinned_included"] >= 1
    pin_hit = next(r for r in with_pins["results"] if r["id"] == pin["id"])
    assert pin_hit["_pinned"] is True
    # A pin sits after the ranked hits.
    assert ids[-1] == pin["id"]


def test_recall_pins_still_respect_structural_filters():
    forget(entity="*", project=PROJECT, soft=False, dry_run=False)
    remember("Round44 structural pin: the queue is Q-1", project=PROJECT,
             entity="Round44-EntityA", pinned=True)
    remember("Round44 structural other: the queue is Q-2", project=PROJECT,
             entity="Round44-EntityB")

    scoped = recall("Round44 structural queue", project=PROJECT,
                    entity="Round44-EntityB", include_pinned=True)
    assert all(r.get("entity") != "Round44-EntityA"
               for r in scoped["results"])
    assert scoped.get("pinned_included", 0) == 0


def test_recall_pin_displaces_the_lowest_ranked_hit_when_page_is_full():
    forget(entity="*", project=PROJECT, soft=False, dry_run=False)
    pin = remember("Round44 displace pin: the region is eu-central",
                   project=PROJECT, pinned=True)
    remember("Round44 displace one: the region is eu-west", project=PROJECT)
    remember("Round44 displace two: the region is us-east", project=PROJECT)

    page = recall("Round44 displace region", project=PROJECT, limit=1,
                  include_pinned=True)
    assert [r["id"] for r in page["results"]] == [pin["id"]]
    assert page["pinned_included"] == 1


def test_recall_only_superseded_ignores_pins():
    forget(entity="*", project=PROJECT, soft=False, dry_run=False)
    pin = remember("Round44 superseded pin: the flag is on",
                   project=PROJECT, pinned=True)
    assert pin["ok"]
    remember("Round44 superseded pin: the flag is off", project=PROJECT,
             supersede=pin["id"])

    history = recall("Round44 superseded pin flag", project=PROJECT,
                     only_superseded=True, include_pinned=True)
    assert all(not r.get("_pinned") for r in history["results"])
    assert history.get("pinned_included", 0) == 0


def test_list_memories_order_by_pinned_lists_pins_first():
    forget(entity="*", project="round44-list", soft=False, dry_run=False)
    plain = remember("Round44 list plain", project="round44-list")
    pin = remember("Round44 list pinned", project="round44-list", pinned=True)
    assert plain["ok"] and pin["ok"]

    listed = list_memories(project="round44-list", order_by="pinned",
                           contains="Round44 list")
    assert [r["id"] for r in listed["results"]][0] == pin["id"]


def test_list_memories_rejects_unknown_order():
    result = list_memories(project=PROJECT, order_by="pinnedx")
    assert result["ok"] is False
    assert "order_by" in result["error"]


# ---------------------------------------------------------------------------
# compile_context: pins lead the block
# ---------------------------------------------------------------------------

def test_compile_context_leads_with_pins_and_reports_them():
    forget(entity="*", project="round44-compile", soft=False, dry_run=False)
    pin = remember("Round44 compile pin: the runbook is RB-9",
                   project="round44-compile", pinned=True)
    remember("Round44 compile ranked: the runbook is RB-1",
             project="round44-compile")
    assert pin["ok"]

    compiled = compile_context("Round44 compile runbook", project="round44-compile")
    assert compiled["ok"] is True
    assert compiled["items"][0]["id"] == pin["id"]
    assert compiled["items"][0]["pinned"] is True
    assert compiled["pinned_included"] >= 1


def test_compile_context_can_disable_pins():
    forget(entity="*", project="round44-compile", soft=False, dry_run=False)
    pin = remember("Zeppelin backup window is 03:00 daily",
                   project="round44-compile", pinned=True)
    remember("Round44 compile off: the secret is S-2",
             project="round44-compile")

    compiled = compile_context("Round44 compile off secret",
                               project="round44-compile", include_pinned=False)
    assert pin["id"] not in [i["id"] for i in compiled["items"]]
    assert compiled["pinned_included"] == 0


# ---------------------------------------------------------------------------
# Backup round-trip
# ---------------------------------------------------------------------------

def test_export_import_preserves_pinned():
    forget(entity="*", project="round44-export", soft=False, dry_run=False)
    forget(entity="*", project="round44-import", soft=False, dry_run=False)
    stored = remember("Round44 export pin probe", project="round44-export",
                      pinned=True)
    assert stored["ok"]

    payload = export_memories(project="round44-export")["data"]
    assert payload["records"][0]["pinned"] is True

    restored = import_memories(payload, project="round44-import")
    assert restored["ok"] is True
    assert _fields(stored["id"], project="round44-import")["pinned"] is True


# ---------------------------------------------------------------------------
# Bug 3: the exact-match path must not flatten `_relevance_score`
# ---------------------------------------------------------------------------

def test_exact_matches_keep_their_real_relevance():
    forget(entity="*", project="round44-exact", soft=False, dry_run=False)
    # Both rows contain the literal identifier, so both are exact hits; their
    # vector similarity to the query differs (one carries extra unrelated
    # words). Before the fix both reported the constant floor.
    remember("ZEBRAFISH_TOKEN", project="round44-exact")
    remember("ZEBRAFISH_TOKEN unrelated padded words here",
             project="round44-exact")

    hits = recall("ZEBRAFISH_TOKEN", project="round44-exact")["results"]
    by_content = {r["content"]: r for r in hits}
    bare = by_content.get("ZEBRAFISH_TOKEN")
    padded = by_content.get("ZEBRAFISH_TOKEN unrelated padded words here")
    assert bare is not None and padded is not None
    assert bare.get("_exact_match") in ("content", "substring")
    assert padded.get("_exact_match") in ("content", "substring")
    # The identifier-bearing row is the better match; the padded one must not
    # be forced up to the same constant.
    assert bare["_relevance_score"] > padded["_relevance_score"]


# ---------------------------------------------------------------------------
# WS5: compile_context diversity and entity cap
# ---------------------------------------------------------------------------

def test_diversity_is_on_by_default_and_collapses_near_duplicates():
    forget(entity="*", project="round44-div", soft=False, dry_run=False)
    # Identical content with exact dedupe disabled: the rows must survive the
    # write, then the DEFAULT diversity step collapses the copies.
    for _ in range(5):
        remember("Round44 div probe: the value is 42", project="round44-div",
                 dedupe="off")
    remember("Round44 div probe sibling: the value is 7",
             project="round44-div")

    compiled = compile_context("Round44 div probe value",
                               project="round44-div")
    assert compiled["ok"] is True
    assert compiled["diversity"] == 0.3
    assert compiled["dropped_by_diversity"] >= 1
    assert compiled["count"] < 6


def test_diversity_zero_is_a_strict_noop():
    forget(entity="*", project="round44-div", soft=False, dry_run=False)
    for _ in range(4):
        remember("Round44 div off probe: the value is 42",
                 project="round44-div", dedupe="off")

    compiled = compile_context("Round44 div off probe value",
                               project="round44-div", diversity=0.0)
    assert compiled["ok"] is True
    assert compiled["dropped_by_diversity"] == 0
    assert compiled["count"] == 4


def test_max_per_entity_caps_one_entity_but_not_another():
    forget(entity="*", project="round44-cap", soft=False, dry_run=False)
    for index in range(3):
        remember(f"Round44 cap probe alpha {index}", project="round44-cap",
                 entity="Round44-CapA")
    remember("Round44 cap probe beta", project="round44-cap",
             entity="Round44-CapB")

    compiled = compile_context("Round44 cap probe", project="round44-cap",
                               max_per_entity=1)
    assert compiled["ok"] is True
    assert compiled["dropped_by_entity_cap"] >= 1
    entities = [item["entity"] for item in compiled["items"]]
    assert entities.count("Round44-CapA") <= 1


def test_context_pins_are_never_evicted_by_the_entity_cap():
    forget(entity="*", project="round44-cap", soft=False, dry_run=False)
    for index in range(3):
        remember(f"Round44 cap pin probe {index}", project="round44-cap",
                 entity="Round44-PinA", pinned=(index == 0))
    remember("Round44 cap pin probe plain", project="round44-cap",
             entity="Round44-PinA")

    compiled = compile_context("Round44 cap pin probe", project="round44-cap",
                               max_per_entity=1)
    ids = [item["id"] for item in compiled["items"]]
    assert compiled["pinned_included"] >= 1
    assert any(item["pinned"] for item in compiled["items"])
    assert ids  # the pin survived the cap


def test_context_rejects_bad_diversity_and_cap():
    bad_div = compile_context("Round44 validation", project=PROJECT,
                              diversity=1.5)
    assert bad_div["ok"] is False
    assert "diversity" in bad_div["error"]

    bad_cap = compile_context("Round44 validation", project=PROJECT,
                              max_per_entity=0)
    assert bad_cap["ok"] is False
    assert "max_per_entity" in bad_cap["error"]
