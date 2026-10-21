"""Uniform response envelope for every MCP tool.

An agent should never have to guess the shape of an answer. Every tool
returns a dict with the same core keys:

    {"ok": bool, "status": str, "id": str|None, "error": str|None,
     "warnings": list}

Tools that return a collection add "count" and "results". Tools that
carry extra payload add their own keys (e.g. "counts" for forget).

The envelope is a deep, stable interface: the tool internals may change
freely, the envelope may not.
"""

from __future__ import annotations

from typing import Any

STATUS_STORED = "stored"
STATUS_DUPLICATE = "duplicate"
STATUS_UPDATED = "updated"
STATUS_DELETED = "deleted"
STATUS_SOFT_FORGOTTEN = "soft_forgotten"
STATUS_NOT_FOUND = "not_found"
STATUS_ERROR = "error"
STATUS_PARTIAL = "partial"
STATUS_OK = "ok"


def ok(status: str, id: str | None = None, **extra: Any) -> dict:
    """Successful envelope."""
    env: dict[str, Any] = {
        "ok": True,
        "status": status,
        "id": id,
        "error": None,
        "warnings": [],
    }
    env.update(extra)
    return env


def fail(status: str, error: str, id: str | None = None, **extra: Any) -> dict:
    """Failed envelope."""
    env: dict[str, Any] = {
        "ok": False,
        "status": status,
        "id": id,
        "error": error,
        "warnings": [],
    }
    env.update(extra)
    return env


def collection(status: str, results: list[dict], **extra: Any) -> dict:
    """Envelope for tools that return a list of items."""
    env: dict[str, Any] = {
        "ok": True,
        "status": status,
        "id": None,
        "error": None,
        "warnings": [],
        "count": len(results),
        "results": results,
    }
    env.update(extra)
    return env
