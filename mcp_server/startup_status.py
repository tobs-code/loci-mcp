"""Non-fatal startup failures that a caller must be able to see.

Migrations that fail during startup used to disappear into a log line: the
server kept running with a stale full-text index or an unmigrated table, and
`server_health` still reported "ok". The warnings recorded here are returned by
`server_health`, so a degraded startup is visible to an agent instead of only in
the container log.
"""

from __future__ import annotations

import threading

_LOCK = threading.Lock()
_WARNINGS: list[str] = []


def record_warning(message: str) -> None:
    """Remember one startup failure (deduplicated, order preserved)."""
    text = str(message).strip()
    if not text:
        return
    with _LOCK:
        if text not in _WARNINGS:
            _WARNINGS.append(text)


def startup_warnings() -> list[str]:
    """A snapshot of the recorded startup failures."""
    with _LOCK:
        return list(_WARNINGS)


def clear_startup_warnings() -> None:
    """Reset the list (tests use this to isolate a startup scenario)."""
    with _LOCK:
        _WARNINGS.clear()
