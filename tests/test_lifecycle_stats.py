"""A TTL is a lifecycle state, not a soft delete.

An expired memory is invisible to every active read. The aggregate counts
must agree: a row whose `expires_at` has passed is not an active memory.
This is the characterization suite for the lifecycle module.
"""

import json

from mcp_server.server import (
    drop_project,
    forget,
    init_context,
    list_projects,
    remember,
    stats_resource,
)


PROJECT = "lifecycle-expired-counts"


def _seed():
    remember("Lifecycle active probe", project=PROJECT)
    remember("Lifecycle expired probe", project=PROJECT,
             expires_at="2020-01-01T00:00:00Z")


def _project_row(project=PROJECT):
    return next(r for r in list_projects()["results"] if r["project"] == project)


def test_init_context_does_not_count_expired_rows_as_active():
    _seed()

    result = init_context(project=PROJECT, include_global=False)
    stats = next(r for r in result["project_stats"] if r["project"] == PROJECT)

    assert result["active_memories"] == 1
    assert result["total_memories"] == 2
    assert stats["memories"] == 1
    assert stats["total_memories"] == 2


def test_list_projects_does_not_count_expired_rows_as_active():
    _seed()

    row = _project_row()

    assert row["memories"] == 1
    assert row["total_memories"] == 2


def test_stats_resource_does_not_count_expired_rows_as_active():
    _seed()

    stats = json.loads(stats_resource())

    assert stats["active_memories"] == 1
    assert stats["total_memories"] == 2


def test_forget_entity_star_preview_does_not_count_expired_rows_as_active():
    _seed()

    preview = forget(entity="*", project=PROJECT, dry_run=True)

    assert preview["ok"] is True
    assert preview["counts"]["active_memories"] == 1
    assert preview["counts"]["already_forgotten"] == 0


def test_drop_project_preview_does_not_count_expired_rows_as_active():
    _seed()

    preview = drop_project(PROJECT, dry_run=True)

    assert preview["ok"] is True
    assert preview["counts"]["active_memories"] == 1
    assert preview["counts"]["memories"] == 2
