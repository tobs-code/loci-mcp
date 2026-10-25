"""Optional SQL filters shared by every retrieval and maintenance path.

The temporal (`as_of`), provenance (`source`, `agent_id`, `min_confidence`) and
tag filters are plain column conditions, so they must be built exactly the same
way in the hybrid search, the browse, the graph-boost expansion and the export.
Keeping them in one module also keeps `search.py` from growing a second copy.

Nothing here touches the database: the callers own the cursor and the table
list. The helpers only validate and normalize caller input, and return SQL
fragments plus the parameters in the same order.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

MAX_TEXT_LEN = 200
MAX_TAGS = 32
MAX_TAG_LEN = 64


def parse_iso_timestamp(value: Any, field: str = "as_of") -> datetime | None:
    """Parse an ISO-8601 date/datetime, or explain what is wrong.

    `None` and an empty string mean "no filter", so an optional argument can be
    passed through unchanged. Anything else must parse; a caller who typed a
    broken timestamp gets an argument error instead of a silently unfiltered
    (or silently empty) result.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(
            tzinfo=timezone.utc)
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an ISO 8601 timestamp")
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        stamp = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(
            f"{field} is not a valid ISO 8601 timestamp: {value!r}"
        ) from exc
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp


def normalize_optional_text(value: Any, field: str,
                            max_len: int = MAX_TEXT_LEN) -> str | None:
    """Trim an optional free-text field; empty becomes None."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    text = value.strip()
    if not text:
        return None
    if len(text) > max_len:
        raise ValueError(f"{field} must be at most {max_len} characters")
    return text


def validate_confidence(value: Any, field: str = "confidence") -> float | None:
    """Validate an optional 0..1 confidence value."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number between 0.0 and 1.0")
    number = float(value)
    if number < 0.0 or number > 1.0:
        raise ValueError(f"{field} must be between 0.0 and 1.0")
    return number


def normalize_tags(value: Any, field: str = "tags") -> list[str] | None:
    """Normalize a tag list to trimmed, lower-case, unique values.

    A comma-separated string is accepted as well, so a client that cannot send
    an array still gets a usable filter. Tag matching is case-insensitive by
    construction: stored tags are normalized the same way.
    """
    if value is None:
        return None
    if isinstance(value, str):
        raw = value.split(",")
    elif isinstance(value, (list, tuple)):
        raw = list(value)
    else:
        raise ValueError(f"{field} must be a list of strings")
    tags: list[str] = []
    seen: set[str] = set()
    for entry in raw:
        if not isinstance(entry, str):
            raise ValueError(f"{field} must be a list of strings")
        tag = entry.strip().lower()
        if not tag:
            continue
        if len(tag) > MAX_TAG_LEN:
            raise ValueError(
                f"each tag must be at most {MAX_TAG_LEN} characters")
        if tag not in seen:
            seen.add(tag)
            tags.append(tag)
    if raw and not tags:
        raise ValueError(f"{field} must not be empty")
    if len(tags) > MAX_TAGS:
        raise ValueError(f"{field} accepts at most {MAX_TAGS} tags")
    return tags


def validity_condition(as_of: Any,
                       prefix: str = "") -> tuple[str, list[Any]]:
    """SQL for "was this memory valid at `as_of`", plus its parameters.

    A row is valid in `[valid_from, valid_to)`; an open `valid_to` means it
    still is. The same fragment serves the filter and the inverted
    `dropped_by_validity` count, so the two can never drift apart.
    """
    stamp = parse_iso_timestamp(as_of, "as_of")
    if stamp is None:
        return "", []
    column = f"{prefix}.valid_from" if prefix else "valid_from"
    end = f"{prefix}.valid_to" if prefix else "valid_to"
    return (f"({column} <= %s AND ({end} IS NULL OR {end} > %s))",
            [stamp, stamp])


def feature_conditions(
    *,
    as_of: Any = None,
    tags: Any = None,
    tags_any: Any = None,
    source: Any = None,
    agent_id: Any = None,
    min_confidence: Any = None,
    prefix: str = "",
) -> tuple[list[str], list[Any]]:
    """Build the optional feature filters shared by every retrieval path.

    `tags` requires every tag (AND), `tags_any` at least one (OR); they are
    mutually exclusive. `prefix` qualifies the columns for a query that aliases
    the memory table.
    """
    if tags is not None and tags_any is not None:
        raise ValueError("tags and tags_any are mutually exclusive")
    conditions: list[str] = []
    params: list[Any] = []
    validity, validity_params = validity_condition(as_of, prefix)
    if validity:
        conditions.append(validity)
        params.extend(validity_params)
    column = f"{prefix}.tags" if prefix else "tags"
    normalized = normalize_tags(tags, "tags")
    if normalized:
        conditions.append(f"{column} @> %s::text[]")
        params.append(normalized)
    normalized_any = normalize_tags(tags_any, "tags_any")
    if normalized_any:
        conditions.append(f"{column} && %s::text[]")
        params.append(normalized_any)
    source_value = normalize_optional_text(source, "source")
    if source_value is not None:
        source_column = f"{prefix}.source" if prefix else "source"
        conditions.append(f"{source_column} = %s")
        params.append(source_value)
    agent_value = normalize_optional_text(agent_id, "agent_id")
    if agent_value is not None:
        agent_column = f"{prefix}.agent_id" if prefix else "agent_id"
        conditions.append(f"{agent_column} = %s")
        params.append(agent_value)
    confidence = validate_confidence(min_confidence, "min_confidence")
    if confidence is not None:
        confidence_column = f"{prefix}.confidence" if prefix else "confidence"
        conditions.append(
            f"({confidence_column} IS NOT NULL AND {confidence_column} >= %s)")
        params.append(confidence)
    return conditions, params
