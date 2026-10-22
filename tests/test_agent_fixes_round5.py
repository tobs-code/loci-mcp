"""Round 5 of the agent-usability fixes.

1. `check_contradictions` no longer needs an entity: without one the check
   runs across all entities (the rate-limit bug of the usability review),
   and with one, leftover candidate seats go to neighbours of *other*
   entities so entity tagging cannot hide a conflict.
2. A blank/whitespace `project` is refused instead of silently writing to
   the global scope; data-driven call sites (batch, import) report the
   affected item individually.
3. `contradictions_report` also works without an entity: scope-wide scan.
4. An FTS-only hit that covers less than FTS_MIN_COVERAGE (now 0.3) of the
   query is dropped: the "pricing/per" one-word noise case is gone.
"""
from mcp_server.server import (
    contradictions_report,
    export_memories,
    import_memories,
    recall,
    remember,
    remember_batch,
)


# --------------------------------------------------------------------------
# Fix 1: contradiction check without entity
# --------------------------------------------------------------------------

def test_check_contradictions_without_entity_finds_the_conflict():
    # Two entity-less facts about the same topic with a contradicting
    # attribute. Previously this answered `found: false` silently, because
    # the candidate search asked `entity = %s` with a NULL entity.
    remember("The cache size is 1000 megabytes on the aws cluster")

    result = remember(
        "The cache size is not 1000 megabytes on the aws cluster",
        check_contradictions=True,
    )

    payload = result["contradictions"]
    assert payload["found"] is True
    assert len(payload["superseded"]) == 1
    assert payload["superseded"][0]["similarity"] == 0.95
    assert any("superseded 1 contradicting memory/facts" in w
               for w in result["warnings"])
    # The old fact is gone from the active set.
    active = recall("cache size megabytes", limit=10)["results"]
    assert len(active) == 1


def test_check_contradictions_without_entity_reports_the_cross_entity_mode():
    # Nothing contradicts here: the check ran anyway, and the answer tells
    # the agent that it compared beyond the missing entity tag.
    result = remember("A brand new fact with no clashes at all",
                      check_contradictions=True)

    assert result["contradictions"]["found"] is False
    assert result["contradictions"]["cross_entity"] is True
    assert "skipped" not in result["contradictions"]
    assert any("no entity set" in w and "across all entities" in w
               for w in result["warnings"])


def test_check_contradictions_with_entity_catches_other_entities_too():
    # The conflicting memory belongs to a different entity. The old
    # `WHERE entity = %s` pre-filter never saw it; now the leftover
    # candidate seats are filled with cross-entity neighbours.
    remember("Bob deploys to production every morning", entity="Bob")

    result = remember(
        "Bob does not deploy to production every morning",
        entity="Alice",
        check_contradictions=True,
    )

    payload = result["contradictions"]
    assert payload["found"] is True
    assert payload["superseded"][0]["similarity"] == 0.95
    active = recall("deploys production morning", limit=10)["results"]
    assert len(active) == 1


def test_check_contradictions_in_batch_works_without_entity():
    result = remember_batch(
        facts=[
            {"content": "The export runs at noon on sundays"},
            {"content": "The export does not run at noon on sundays"},
            {"content": "A third neutral fact with distinct words"},
        ],
        check_contradictions=True,
    )

    entries = result["results"]
    assert all(e["status"] == "stored" for e in entries)
    conflicting = entries[1]
    assert conflicting["contradictions"]["found"] is True
    neutral = entries[2]
    assert neutral["contradictions"]["cross_entity"] is True
    assert "skipped" not in neutral["contradictions"]


# --------------------------------------------------------------------------
# Fix 2: blank project is a caller mistake, not the global scope
# --------------------------------------------------------------------------

def test_remember_refuses_a_blank_project_with_a_named_argument():
    result = remember("A fact with a blank project name", project="   ")

    assert result["ok"] is False
    assert "project" in result["error"]
    assert "empty" in result["error"]
    # Nothing was stored anywhere.
    assert recall("a fact with a blank project", project="*", limit=10)["results"] == []


def test_remember_batch_reports_a_blank_fact_project_individually():
    result = remember_batch(facts=[
        {"content": "First fact with clean distinct words"},
        {"content": "Second fact with clean distinct words", "project": "  "},
    ])

    assert result["status"] == "partial"
    entries = result["results"]
    assert entries[0]["status"] == "stored"
    assert entries[1]["status"] == "error"
    assert "project" in entries[1]["error"]


def test_import_reports_a_record_with_a_blank_project_and_keeps_the_rest():
    remember("Good record for the import probe", project="import-probe")
    export = export_memories(project="import-probe", include_graph=False)
    records = export["data"]["records"]
    assert records
    broken = dict(records[0])
    broken["project"] = "   "

    result = import_memories(
        data={"type": "memory_export", "version": "0.2", "records": [broken]},
        format="json",
    )

    assert result["ok"] is True
    assert result["imported"] == 0
    assert result["skipped"] == 1
    assert any("project" in e for e in result.get("errors", []))


# --------------------------------------------------------------------------
# Fix 3: contradictions_report without entity scans the whole scope
# --------------------------------------------------------------------------

def test_report_without_entity_finds_cross_entity_conflicts():
    first = remember("The nightly batch processed 42 orders on monday")
    second = remember("The nightly batch did not process 42 orders on monday")

    report = contradictions_report()

    assert report["ok"] is True
    assert report["count"] == 1
    pair = report["results"][0]
    assert {pair["a"]["id"], pair["b"]["id"]} == {first["id"], second["id"]}
    assert pair["confidence"] > 0.7


def test_report_without_entity_and_resolve_newest_supersedes_the_older():
    older = remember("The staging database reloads at midnight on saturday")
    newer = remember("The staging database does not reload at midnight on saturday")

    report = contradictions_report(resolve="newest")

    assert report["resolved"] == 1
    pair = report["results"][0]
    pair = report["results"][0]
    assert pair["resolution"] == {"superseded": older["id"], "kept": newer["id"]}
    active = recall("staging database reloads", limit=10)["results"]
    assert [r["id"] for r in active] == [newer["id"]]


def test_report_without_entity_in_a_project_articulates_emptiness():
    remember("Deploy note for the empty report probe", project="report-probe")
    remember("Deploy note for the empty report probe", project="report-probe",
             entity="DeployOps")

    report = contradictions_report(entity=None, project="report-probe")

    assert report["ok"] is True
    # Two non-contradicting facts: the answer still says why it is empty.
    assert any("nothing to compare" in w or "no contradictions" in w
               or "no candidate pairs" in w for w in report["warnings"])


# --------------------------------------------------------------------------
# Fix 4: one-word FTS-only matches of a four-word query are dropped
# --------------------------------------------------------------------------

def test_single_function_word_fts_match_is_dropped():
    # "per" is NOT a stopword for the english dictionary, so this document
    # matched the 4-lexeme query through that one function word. With
    # FTS_MIN_COVERAGE 0.3, the 1/4 = 0.25 coverage is below the floor.
    pricing = remember("The pricing tier for enterprise customers is $500 per month")

    result = recall("requests per minute limit", limit=10)

    assert all(r["id"] != pricing["id"] for r in result["results"])
    assert result.get("dropped_by_coverage", 0) >= 1


def test_distinctive_word_fts_match_survives_the_floor():
    # One distinctive lexeme of a short query must keep coming back: only
    # the sliver case was supposed to go.
    probe = remember("Cache probe: the primary sessions live on redis cluster alpha")

    result = recall("redis cluster alpha", limit=10, min_similarity=None)

    assert any(r["id"] == probe["id"] for r in result["results"])
