"""Startup failures must reach `server_health`, not only the container log."""

import pytest

from mcp_server.startup_status import (
    clear_startup_warnings,
    record_warning,
    startup_warnings,
)


@pytest.fixture(autouse=True)
def _isolate():
    clear_startup_warnings()
    yield
    clear_startup_warnings()


def test_records_a_warning():
    record_warning("full-text index unavailable")

    assert startup_warnings() == ["full-text index unavailable"]


def test_deduplicates_and_preserves_order():
    record_warning("first")
    record_warning("second")
    record_warning("first")

    assert startup_warnings() == ["first", "second"]


def test_ignores_blank_messages():
    record_warning("")
    record_warning("   ")

    assert startup_warnings() == []


def test_snapshot_is_a_copy():
    record_warning("first")

    snapshot = startup_warnings()
    snapshot.append("mutated")

    assert startup_warnings() == ["first"]


def test_clear_resets_the_list():
    record_warning("first")
    clear_startup_warnings()

    assert startup_warnings() == []
