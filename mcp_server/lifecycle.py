"""The single definition of a memory's lifecycle state.

A row is *active* only while it is not superseded, not soft-forgotten and
not expired. Every read, write guard, count and export asks this module, so
a change to the rule can never make `recall` and the statistics disagree
again.
"""

from datetime import datetime, timezone
from typing import Any

from mcp_server.config import SENTINEL_FORGOTTEN

# The SQL form used by every "is this row usable?" query. Kept as a constant
# because it is embedded in larger f-strings; prefer `active_sql()` when an
# alias is needed.
ACTIVE_ROW_SQL = (
    "superseded_by IS NULL AND deleted_at IS NULL "
    "AND (expires_at IS NULL OR expires_at > NOW())"
)

# Relation-context sentences explain a graph edge; they are not facts about
# the world and must never compete with a real memory in a contradiction scan.
NOT_RELATION_CONTEXT_SQL = (
    "COALESCE(metadata->>'relation_context', 'false') <> 'true'"
)


def active_sql(alias: str | None = None) -> str:
    """The active-row predicate, optionally qualified with a table alias."""
    prefix = f"{alias}." if alias else ""
    return (
        f"{prefix}superseded_by IS NULL AND {prefix}deleted_at IS NULL "
        f"AND ({prefix}expires_at IS NULL OR {prefix}expires_at > NOW())"
    )


def superseded_marker_sql(alias: str | None = None) -> str:
    """Match a real supersede marker, never the forgotten sentinel."""
    prefix = f"{alias}." if alias else ""
    return (
        f"{prefix}superseded_by IS NOT NULL "
        f"AND {prefix}superseded_by <> '{SENTINEL_FORGOTTEN}'"
    )


def visibility_sql(include_deleted: bool = False,
                   include_superseded: bool = False,
                   include_expired: bool = False) -> str:
    """SQL filter for the orthogonal visibility flags.

    Active rows are always visible; `include_superseded` adds history,
    `include_deleted` adds forgotten rows, and `include_expired` adds rows
    whose TTL has passed. Without `include_expired`, an expired row stays
    hidden even when `include_deleted` or `include_superseded` is set: a TTL
    does not turn a row into history or make it a forgotten one.
    """
    if include_deleted and include_superseded:
        base = "TRUE"
    elif include_deleted:
        base = "(deleted_at IS NOT NULL OR superseded_by IS NULL)"
    elif include_superseded:
        base = "deleted_at IS NULL"
    elif not include_expired:
        return ACTIVE_ROW_SQL
    else:
        base = "superseded_by IS NULL AND deleted_at IS NULL"
    if include_expired:
        return base
    return f"({base}) AND (expires_at IS NULL OR expires_at > NOW())"


def parse_expires_at(value: Any) -> datetime:
    """Parse a TTL value, rejecting anything that is not ISO 8601.

    Returns a timezone-aware datetime; a value without an offset is treated
    as UTC, matching how PostgreSQL interprets the stored timestamptz.
    """
    try:
        stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "expires_at must be an ISO 8601 timestamp "
            "(e.g. '2027-01-15' or '2027-01-15T12:00:00Z')"
        ) from exc
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp


def is_expired(value: Any) -> bool:
    """True when a stored TTL is in the past; never raises."""
    if value is None:
        return False
    if isinstance(value, datetime):
        stamp = value
    elif isinstance(value, str):
        try:
            stamp = parse_expires_at(value)
        except ValueError:
            return False
    else:
        return False
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp <= datetime.now(timezone.utc)


def is_visible(row: dict, include_deleted: bool = False,
               include_superseded: bool = False,
               include_expired: bool = False) -> bool:
    """Python mirror of `visibility_sql` for one loaded row."""
    expired = is_expired(row.get("expires_at"))
    if expired and not include_expired:
        return False
    deleted = row.get("deleted_at") is not None
    superseded_by = row.get("superseded_by")
    superseded = (superseded_by is not None
                  and superseded_by != SENTINEL_FORGOTTEN)
    if include_deleted and include_superseded:
        return True
    if include_deleted:
        return deleted or not superseded
    if include_superseded:
        return not deleted
    return not deleted and not superseded


def is_active(row: dict) -> bool:
    """True when the row is neither superseded, forgotten nor expired."""
    return is_visible(row)


def counts(cur, table: str) -> dict[str, int]:
    """Lifecycle counts for one memory table.

    `active` uses the same predicate as `recall`; `superseded` excludes the
    forgotten sentinel. A row may be both superseded and forgotten, so the
    four numbers do not have to add up.
    """
    cur.execute(
        f"SELECT COUNT(*), "
        f"COUNT(*) FILTER (WHERE {active_sql()}), "
        f"COUNT(*) FILTER (WHERE {superseded_marker_sql()}), "
        f"COUNT(*) FILTER (WHERE deleted_at IS NOT NULL) "
        f"FROM {table}"
    )
    total, active, superseded, forgotten = cur.fetchone()
    return {
        "total": int(total),
        "active": int(active),
        "superseded": int(superseded),
        "forgotten": int(forgotten),
    }
