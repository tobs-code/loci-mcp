"""Duplicate and near-duplicate policy lives in one module."""

from mcp_server.duplicates import (
    dedupe_across_scopes,
    dedupe_similar_rows,
    is_cross_scope_duplicate,
    same_topic,
    shared_content_word,
)


def _row(memory_id, content_hash, project="global", **extra):
    row = {"id": memory_id, "content_hash": content_hash,
           "_project": project, "_table": ("memories" if project == "global"
                                           else f"mem_{project}")}
    row.update(extra)
    return row


def test_same_topic_accepts_an_attribute_update():
    old = "Refund worker retries failed jobs three times with backoff."
    new = ("Refund worker retries failed jobs up to five times with a 30 "
           "second timeout.")

    assert same_topic(new, old, "Refund worker", 0.96) is True


def test_same_topic_refuses_another_attribute_of_the_same_entity():
    port = "Payments service listens on port 8001 for requests."
    backup = "Payments service runs a nightly backup job at 02:00."

    assert same_topic(port, backup, "Payments service", 0.96) is False


def test_shared_content_word_is_the_cheap_safety_net():
    assert shared_content_word("API uses Postgres", "Postgres is at v16")
    assert not shared_content_word("API uses Postgres", "Redis caches sessions")


def test_project_copy_wins_over_global_and_history_survives():
    rows = [
        _row("p", "h1", "api"),
        _row("g", "h1", "global"),
        _row("h", "h1", "global", superseded_by="newer"),
    ]

    kept, hidden = dedupe_across_scopes(rows)

    assert {r["id"] for r in kept} == {"p", "h"}
    assert hidden == 1


def test_copies_inside_one_scope_stay_visible():
    rows = [_row("a", "h1", "api"), _row("b", "h1", "api")]

    kept, hidden = dedupe_across_scopes(rows)

    assert len(kept) == 2
    assert hidden == 0


def test_include_duplicates_disables_the_winner_rule():
    rows = [_row("p", "h1", "api"), _row("g", "h1", "global")]

    kept, hidden = dedupe_across_scopes(rows, include_duplicates=True)

    assert len(kept) == 2
    assert hidden == 0


def test_stream_duplicate_needs_an_active_row():
    seen: dict[str, str] = {}
    active = _row("p", "h1", "api")
    history = _row("h", "h1", "global", deleted_at="2020-01-01")

    assert is_cross_scope_duplicate(active, "mem_api", seen) is False
    assert is_cross_scope_duplicate(_row("g", "h1"), "memories", seen) is True
    assert is_cross_scope_duplicate(history, "memories", seen) is False


def test_bootstrap_wording_dedupe_keeps_the_first_variant():
    rows = [{"content": "Deploy on Mondays"},
            {"content": "deploy on monday"},
            {"content": "Restart the cache"}]

    kept = dedupe_similar_rows(rows)

    assert [r["content"] for r in kept] == [
        "Deploy on Mondays", "Restart the cache"]
