import json
import logging
import re
from datetime import datetime, timezone
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

from mcp_server.config import (
    DEDUPE_MODES,
    MEMORY_TYPE_HELP,
    MEMORY_TYPES,
    NEAR_DUPLICATE_THRESHOLD,
    SENTINEL_FORGOTTEN,
    SUPERSEDE_SIMILARITY_MIN,
    VALID_SCOPES,
)
from mcp_server.db import get_conn
from mcp_server.envelope import (
    STATUS_CONFLICT,
    STATUS_DELETED,
    STATUS_DUPLICATE,
    STATUS_ERROR,
    STATUS_NOT_FOUND,
    STATUS_OK,
    STATUS_PARTIAL,
    STATUS_RESTORED,
    STATUS_SOFT_FORGOTTEN,
    STATUS_STORED,
    STATUS_UPDATED,
    fail,
    ok,
)
from mcp_server.graph import _relate_edges, _validate_weight
from mcp_server.inference_client import embed, nli_configured
from mcp_server.scopes import (
    _all_mem_tables,
    _ensure_project_table,
    _mem_table,
    _memory_scopes,
    _norm_entity,
    _norm_project,
    _project_exists,
    _safe_ident,
    _scope_key,
    _scope_label,
    _scope_of_table,
    _table_exists,
)
from mcp_server.scoring import _content_hash
from mcp_server.storage import (
    ACTIVE_ROW_SQL,
    clear_dangling_supersede,
    _by_content_hash,
    _check_contradictions,
    _deduplicate,
    _find_by_content,
    _find_by_id,
    _find_last_for_entity,
    _find_last_in_scope,
    _find_near_duplicate,
    _find_superseded_match,
    _scope_of_id,
    _store_memory,
    _supersede,
)

logger = logging.getLogger(__name__)


class BatchFact(BaseModel):
    """One fact of a batch. Only `content` is required."""

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="before")
    @classmethod
    def _allow_missing_content(cls, data: Any) -> Any:
        """A fact without `content` must not abort the whole batch.

        The schema still marks `content` as required, so a client that
        follows it never hits this path. But an agent that builds the
        facts programmatically must not lose the whole batch because one
        entry is malformed: the runtime validation reports that fact
        individually ("content is required") and stores the rest.
        """
        if isinstance(data, dict) and "content" not in data:
            return {**data, "content": None}
        return data

    content: Any = Field(json_schema_extra={"type": "string"},
                          description="The fact to store (required, non-empty).")
    mem_type: Any = Field(default="fact", json_schema_extra={
        "type": "string",
        "enum": ["fact", "decision", "learning", "preference", "pattern",
                 "incident", "runbook", "context", "observation", "convention",
                 "command", "note"],
    }, description=MEMORY_TYPE_HELP)
    entity: Any = Field(default=None, json_schema_extra={"type": ["string", "null"]},
                         description="Entity this memory belongs to; links it in the graph.")
    importance: Any = Field(default=None, json_schema_extra={"type": ["number", "null"]},
                            description="Importance in 0.0-1.0 (default 0.5); scales decay weighting.")
    metadata: Any = Field(default=None, json_schema_extra={"type": ["object", "null"]},
                          description="Free-form JSON object stored with the memory.")
    scope: Any = Field(default=None, json_schema_extra={
        "type": ["string", "null"], "enum": ["global", "project", None],
    }, description="Where to store: 'project' (needs a project) or 'global'.")
    project: Any = Field(default=None, json_schema_extra={"type": ["string", "null"]},
                         description="Overrides the batch project for this fact.")
    supersede: Any = Field(default=None, json_schema_extra={"type": ["string", "null"]},
                           description="Memory id this fact replaces.")
    supersede_content: Any = Field(default=None, json_schema_extra={"type": ["string", "null"]},
                                   description="Exact content of the memory this fact "
                                               "replaces. Matches the target scope first, "
                                               "then the global scope (a global match hides "
                                               "that fact in every scope and is reported as "
                                               "a cross-scope supersede).")
    supersede_last: Any = Field(default=None, json_schema_extra={"type": ["boolean", "null"]},
                                description="Replace the memory that best "
                                            "matches the new fact (most "
                                            "similar, not the newest) - of the "
                                            "entity, or across the whole scope "
                                            "when no entity is set. Only when the "
                                            "new fact is about the same "
                                            "topic (similarity >= 0.5); otherwise the "
                                            "fact is stored additionally.")
    auto_supersede_near_duplicate: Any = Field(
        default=None,
        json_schema_extra={"type": ["boolean", "null"]},
        description="Replace an active near-duplicate (similarity >= 0.75) "
                    "instead of storing the reworded copy as a fork. "
                    "Alias for on_near_duplicate='supersede'.")
    on_near_duplicate: Any = Field(
        default=None,
        json_schema_extra={"type": ["string", "null"],
                           "enum": ["warn", "supersede", "reject", None]},
        description="'warn' stores and warns, 'supersede' replaces the "
                    "near-duplicate, 'reject' stores nothing.")
    expires_at: Any = Field(default=None, json_schema_extra={
        "type": ["string", "null"]},
        description="Optional ISO 8601 timestamp when this memory stops "
                    "being active (TTL).")
    fuzzy_threshold: Any = Field(default=None, json_schema_extra={"type": ["number", "null"]},
                                 description="Similarity threshold 0.0-1.0 for semantic dedup.")
    dedupe: Any = Field(default=None, json_schema_extra={
        "type": ["string", "null"], "enum": ["exact", "semantic", "off", None],
    }, description="Duplicate detection: 'exact' (content hash), 'semantic' (vector similarity) or 'off'.")


def _sanitize_batch_fact(value: Any) -> Any:
    """Accept a malformed entry without losing the whole batch.

    A fact that is not an object at all must not abort the batch with a
    schema error: it is reported individually ("fact must be an object")
    and the other facts are still stored.
    """
    if isinstance(value, (dict, BatchFact)):
        return value
    return {"content": None, "_not_an_object": True}


BatchFactInput = Annotated[BatchFact, BeforeValidator(_sanitize_batch_fact)]


def _same_subject_entity(near_dup: dict, entity: str | None) -> bool:
    """Whether the near-duplicate match and the new fact share an entity.

    Entity tagging is the only reliable subject signal available before an
    NLI call: two memories about staging and production are often nearly
    identical in wording but describe different systems. An untagged write
    and an untagged match count as the same subject (the historical
    behaviour); one tagged and one untagged does not.
    """
    return _norm_str(near_dup.get("entity")) == _norm_str(entity)


_WORD_RE = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")
_CAMEL_WORD_RE = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+")
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "for", "from",
    "has", "have", "in", "is", "it", "its", "of", "on", "or", "that", "the",
    "this", "to", "was", "were", "with", "now", "still", "also",
    "gets", "get", "new", "old",
    "der", "die", "das", "ein", "eine", "einer", "und", "oder", "ist", "sind",
    "auf", "fuer", "für", "mit", "von", "zu", "im", "in", "den", "dem",
    "des", "wird", "werden", "hat", "haben",
}

# Attribute *values* change when a fact is updated, so they must not count as
# the shared subject: "runs on friday" -> "runs on saturday" is the same
# topic, "runs on port 8766" -> "runs a nightly backup" is not.
_VALUE_WORDS = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "mon", "tue", "wed", "thu", "fri", "sat", "sun",
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
    "daily", "weekly", "monthly", "yearly", "annual", "annually",
    "hourly", "today", "tomorrow", "yesterday", "tonight", "utc", "gmt",
} | _STOPWORDS

_NEGATIONS = {
    "not", "no", "never", "without", "none", "cannot", "disabled", "false",
    "nicht", "kein", "keine", "keinen", "ohne", "nie", "niemals",
}


def _entity_tokens(entity: str | None) -> set[str]:
    """Content-free tokens of an entity name (CamelCase is split).

    The entity name is not a subject term: two facts about the same service
    always share it, even when one describes its port and the other its
    backup job. Removing it is what makes the topic check meaningful.
    """
    if not entity:
        return set()
    tokens: set[str] = set()
    for chunk in re.split(r"[^A-Za-z0-9]+", str(entity)):
        if not chunk:
            continue
        # The content tokenizer keeps "Round19Svc" as one token, so the
        # camel-split parts alone would not remove it from the topic words.
        tokens.add(chunk.lower())
        for match in _CAMEL_WORD_RE.findall(chunk):
            tokens.add(match.lower())
    return tokens


def _topic_tokens(text: str, entity: str | None) -> set[str]:
    """Subject words of a fact: no stopwords, values or the entity name."""
    tokens = {token.lower() for token in _WORD_RE.findall((text or "").lower())}
    ignored = _VALUE_WORDS | _entity_tokens(entity)
    return {token for token in tokens
            if token not in ignored and not token.isdigit() and len(token) > 1}


def _shares_subject(new_content: str, old_content: str,
                    entity: str | None,
                    similarity: float | None = None) -> bool:
    """Whether two facts talk about the same non-entity subject.

    `supersede_last` used to accept any pair above a similarity floor. Two
    sentences about the same service share their template ("the QA service
    runs ..."), so a port fact could replace a backup-job fact. This compares
    the remaining *attribute* words after removing the entity, stopwords and
    attribute values (numbers, days, intervals). A port update keeps the
    words "port"/"runs"; a backup job does not.

    Jaccard overlap (not just "one shared word") keeps a generic verb such as
    "runs" from proving a topic match on its own. Too-short content falls back
    to a very high similarity so a legitimate one-word update is not blocked.
    """
    new_tokens = _topic_tokens(new_content, entity)
    old_tokens = _topic_tokens(old_content, entity)
    if not new_tokens or not old_tokens:
        return similarity is not None and similarity >= 0.92
    overlap = len(new_tokens & old_tokens)
    return overlap / len(new_tokens | old_tokens) > 0.5


def _content_conflict(new_content: str, old_content: str) -> bool:
    """Cheap signal that two similar texts differ in a number or a negation.

    Such a pair is usually an attribute update or an outright contradiction,
    not a reworded copy. The near-duplicate warning labels it accordingly so
    an agent does not blindly accept `on_near_duplicate='reject'` advice for a
    fact that is actually meant to replace the old value.

    This is deliberately not an NLI call: writes stay cheap. It only decides
    how the warning reads; `check_contradictions=true` remains the real test.
    """
    new_numbers = set(_NUMBER_RE.findall((new_content or "").lower()))
    old_numbers = set(_NUMBER_RE.findall((old_content or "").lower()))
    if new_numbers and old_numbers and new_numbers != old_numbers:
        return True
    new_words = set(_WORD_RE.findall((new_content or "").lower()))
    old_words = set(_WORD_RE.findall((old_content or "").lower()))
    return bool(new_words & _NEGATIONS) != bool(old_words & _NEGATIONS)


def _near_duplicate_warning(near_dup: dict, entity: str | None = None,
                            conflict: bool = False,
                            update: bool = False) -> str:
    """Warning for a write that dedupe did not match but that still reads
    like a reworded copy of an active memory (remember + remember_batch).

    `entity` is the entity of the new write. When it differs from the matched
    memory's entity, the warning must NOT suggest `supersede`: following that
    advice would replace a fact about a different subject merely because the
    sentences look alike.

    `update=True` marks the update paths (update_memory/update_batch): the
    edit already applied to one row and the warning names ANOTHER active
    row. An update call has no `supersede` parameter, so the advice must
    point at a separate supersede (remember call) or an edit of that row -
    "pass supersede=<id>" on the tool that is running would be advice the
    caller cannot execute.
    """
    if not _same_subject_entity(near_dup, entity):
        old_entity = near_dup.get("entity")
        new_entity = entity if entity is not None else "untagged"
        preview = (near_dup.get("content") or "").strip()
        if len(preview) > 120:
            preview = preview[:117] + "..."
        if update:
            return (
                f"near-duplicate across entities: active memory "
                f"{near_dup['id']} (entity {old_entity!r}) matches the "
                f"updated content with similarity {near_dup['similarity']}, "
                f"but this memory belongs to entity {new_entity!r} - the "
                f"update did not touch that fact. If both describe the same "
                f"subject, tag them with the same entity; replace the other "
                f"fact in a separate remember call (supersede="
                f"{near_dup['id']}) only if you are sure it must be "
                f"replaced. Matched content: {preview!r}"
            )
        return (
            f"near-duplicate across entities: active memory {near_dup['id']} "
            f"(entity {old_entity!r}) matches this content with similarity "
            f"{near_dup['similarity']}, but this write belongs to entity "
            f"{new_entity!r} - stored as a separate fact. If both describe the "
            f"same subject, tag them with the same entity; only pass "
            f"supersede={near_dup['id']} if you are sure the other fact must "
            f"be replaced. Matched content: {preview!r}"
        )
    if conflict:
        if update:
            return (
                f"near-duplicate: active memory {near_dup['id']} matches the "
                f"updated content with similarity {near_dup['similarity']}, "
                f"but the two differ in a number or a negation - that is an "
                f"attribute update or a contradiction, not a reworded copy. "
                f"This update applied without replacing that memory; to "
                f"retire it, supersede it in a separate remember call "
                f"(supersede={near_dup['id']}), or rerun this update with "
                f"dedupe='off' to skip this warning"
            )
        return (
            f"near-duplicate: active memory {near_dup['id']} matches this "
            f"content with similarity {near_dup['similarity']}, but the two "
            f"differ in a number or a negation - that is an attribute update "
            f"or a contradiction, not a reworded copy. Pass "
            f"check_contradictions=true to let NLI decide, "
            f"supersede={near_dup['id']} to replace the old fact, or "
            f"dedupe='off' to store without this warning"
        )
    if update:
        return (
            f"near-duplicate: active memory {near_dup['id']} matches the "
            f"updated content with similarity {near_dup['similarity']} - the "
            f"edit looks like a reworded copy and that memory is still "
            f"active. To retire it, supersede it in a separate remember "
            f"call (supersede={near_dup['id']}); rerun this update with "
            f"dedupe='off' to skip this warning"
        )
    return (
        f"near-duplicate: active memory {near_dup['id']} matches this content "
        f"with similarity {near_dup['similarity']} - the write looks like a "
        f"reworded copy; pass supersede={near_dup['id']} to replace it, or "
        f"dedupe='off' to store without this warning"
    )


def _ttl_past_warning(value: Any) -> str | None:
    """Warn when an accepted TTL is already in the past.

    A past timestamp is valid input (imports restore expired rows), but a
    write that stores an immediately invisible fact must say so instead of
    answering "stored" and letting the caller search for it.
    """
    if value is None:
        return None
    try:
        stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    if stamp <= datetime.now(timezone.utc):
        return (
            "expires_at is in the past: the memory was stored but is "
            "immediately inactive - it is hidden from every active read "
            "(use include_expired=true to see it)")
    return None


def _read_metadata(cur, store_project: str | None, memory_id: str) -> dict:
    """Metadata of a row that is about to be replaced (project, then global)."""
    table = _mem_table(store_project)
    cur.execute(
        f"SELECT metadata FROM {table} WHERE id = %s AND deleted_at IS NULL",
        (memory_id,))
    row = cur.fetchone()
    if not row and store_project:
        # A cross-scope supersede replaced a shared fact: its metadata lives
        # in the global table, not in the project one.
        cur.execute(
            "SELECT metadata FROM memories WHERE id = %s AND deleted_at IS NULL",
            (memory_id,))
        row = cur.fetchone()
    if not row:
        return {}
    try:
        meta = row[0] if isinstance(row[0], dict) else json.loads(row[0] or "{}")
    except (TypeError, ValueError):
        return {}
    return meta if isinstance(meta, dict) else {}


def _is_past(value: Any) -> bool:
    if not isinstance(value, datetime):
        return False
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value <= datetime.now(timezone.utc)


def _hidden_state_reason(cur, memory_id: str,
                         tables: list[str]) -> str | None:
    """Why an existing row cannot be updated or touched.

    A plain "NOT FOUND" hides that the id *does* exist but is superseded,
    forgotten or expired - the caller then keeps looking for the wrong
    problem. This returns the actionable reason, or None for an unknown id.
    """
    for table in tables:
        cur.execute(
            f"SELECT superseded_by, deleted_at, expires_at FROM {table} "
            "WHERE id = %s", (memory_id,))
        row = cur.fetchone()
        if not row:
            continue
        superseded_by, deleted_at, expires_at = row
        reasons: list[str] = []
        if deleted_at is not None:
            reasons.append(
                "soft-forgotten (pass restore=true to update_memory to bring "
                "it back)")
        if (superseded_by is not None
                and superseded_by != SENTINEL_FORGOTTEN):
            reasons.append(
                f"superseded by {superseded_by} (history only; read it with "
                f"include_superseded=true)")
        if _is_past(expires_at):
            reasons.append(
                "expired (TTL passed; set clear_expires_at=true or a future "
                "expires_at)")
        return "; ".join(reasons) if reasons else None
    return None


def _coerce_metadata(value: Any) -> tuple[dict | None, str | None]:
    """Accept a metadata argument as an object or as a JSON-encoded string.

    LLM callers sometimes send `metadata` as a string instead of a dict.
    `remember_batch` already parsed that (and `metadata_filter` accepts both
    shapes too), so the write tools do the same: an object passes through, a
    JSON string of an object is parsed, anything else fails with a message
    that names the expected shape.
    """
    if isinstance(value, dict):
        return value, None
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None, ("metadata is neither an object nor valid JSON - "
                          "pass an object like {\"env\": \"test\"}")
        if isinstance(parsed, dict):
            return parsed, None
        return None, (f"metadata must be a JSON object, e.g. "
                      f"{{\"env\": \"test\"}} (got {type(parsed).__name__})")
    if isinstance(value, str):
        return None, ("metadata must not be a blank string: pass a JSON object "
                      "like {\"env\": \"test\"}, or a JSON-encoded object "
                      "string")
    return None, (f"metadata must be a JSON object, e.g. "
                  f"{{\"env\": \"test\"}} (got {type(value).__name__})")


def _cross_scope_warning(ids: list[str]) -> str:
    """Warning for a project write that invalidated shared (global) facts:
    they are hidden in EVERY scope now, not only in the project."""
    return (
        f"cross-scope supersede: {len(ids)} shared (global) "
        f"memory/memories invalidated by this project write: "
        + ", ".join(ids)
        + " - a global fact is hidden in EVERY scope now, not only in this "
        "project"
    )


def _cross_scope_duplicates(cur, content: str,
                            store_project: str | None) -> list[dict[str, Any]]:
    """Active copies of exactly this content in scopes other than the target.

    Dedupe only looks at the table the fact is written to, so the same content
    stored in a project and in the shared scope was accepted silently. Reads
    then hide one copy as a cross-scope duplicate, which looks like a lost
    write. The caller gets this list as a warning (the fact is still stored:
    a deliberate copy is allowed, it just must not be invisible).
    """
    c_hash = _content_hash(content)
    target = _mem_table(store_project)
    found: list[dict[str, Any]] = []
    for table in _all_mem_tables(cur):
        if table == target:
            continue
        cur.execute(
            f"SELECT id FROM {table} WHERE content_hash = %s "
            f"AND {ACTIVE_ROW_SQL} LIMIT 1",
            (c_hash,))
        row = cur.fetchone()
        if row:
            found.append({"id": row[0],
                          "project": _scope_of_table(cur, table)})
    return found


def _cross_scope_duplicate_warning(found: list[dict[str, Any]]) -> str:
    detail = ", ".join(f"{d['project']} ({d['id']})" for d in found)
    return (
        f"identical content already exists in another scope: {detail}; "
        f"recall and list_memories hide cross-scope copies by default - pass "
        f"include_duplicates=true (list_memories) to see every copy"
    )


class BatchUpdate(BaseModel):
    """One entry of an update batch. Only `memory_id` is required."""

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="before")
    @classmethod
    def _allow_missing_id(cls, data: Any) -> Any:
        """A malformed entry must be reported individually, not abort the
        batch: the runtime validation answers "memory_id is required"."""
        if isinstance(data, dict) and "memory_id" not in data:
            return {**data, "memory_id": None}
        return data

    memory_id: Any = Field(json_schema_extra={"type": "string"},
                           description="Id of the memory to update (required).")
    content: Any = Field(default=None, json_schema_extra={"type": ["string", "null"]},
                         description="New content; re-embeds the memory.")
    importance: Any = Field(default=None, json_schema_extra={"type": ["number", "null"]},
                            description="New importance in 0.0-1.0.")
    metadata: Any = Field(default=None, json_schema_extra={"type": ["object", "null"]},
                          description="Metadata object, e.g. "
                                      "{\"env\": \"test\"}. A JSON-encoded "
                                      "string of an object is accepted and "
                                      "parsed. Replaces the stored object "
                                      "unless metadata_mode='merge'.")
    metadata_mode: Any = Field(default=None, json_schema_extra={
        "type": ["string", "null"], "enum": ["replace", "merge", None],
    }, description="'replace' (default) or 'merge' (keep stored keys the new "
                   "object does not set).")
    clear_metadata_keys: Any = Field(default=None, json_schema_extra={
        "type": ["array", "null"], "items": {"type": "string"},
    }, description="Metadata keys to remove.")
    entity: Any = Field(default=None, json_schema_extra={"type": ["string", "null"]},
                        description="New entity the memory belongs to.")
    mem_type: Any = Field(default=None, json_schema_extra={"type": "string"},
                          description="New memory type. " + MEMORY_TYPE_HELP)
    project: Any = Field(default=None, json_schema_extra={"type": ["string", "null"]},
                         description="Overrides the batch project for this entry.")
    dedupe: Any = Field(default=None, json_schema_extra={
        "type": ["string", "null"], "enum": ["exact", "semantic", "off", None],
    }, description="Duplicate check for a content change: 'exact', 'semantic' or 'off'.")
    fuzzy_threshold: Any = Field(default=None, json_schema_extra={"type": ["number", "null"]},
                                 description="Similarity threshold 0.0-1.0 for semantic dedupe.")
    expires_at: Any = Field(default=None, json_schema_extra={
        "type": ["string", "null"]},
        description="New ISO 8601 TTL timestamp.")
    clear_expires_at: Any = Field(default=None, json_schema_extra={
        "type": ["boolean", "null"]},
        description="Remove the TTL (mutually exclusive with expires_at).")
    touch: Any = Field(default=None, json_schema_extra={
        "type": ["boolean", "null"]},
        description="Refresh the memory without changing its content: sets "
                    "updated_at to now so the decay age starts over. Can be "
                    "combined with the other update fields.")
    restore: Any = Field(default=None, json_schema_extra={
        "type": ["boolean", "null"]},
        description="Undo a soft delete (mutually exclusive with every other "
                    "update field).")


def _sanitize_batch_update(value: Any) -> Any:
    """Accept a malformed entry without losing the whole batch."""
    if isinstance(value, (dict, BatchUpdate)):
        return value
    return {"memory_id": None, "_not_an_object": True}


BatchUpdateInput = Annotated[BatchUpdate, BeforeValidator(_sanitize_batch_update)]


def remember(
    content: Annotated[str, Field(description="The fact to remember (non-empty).")],
    mem_type: Annotated[
        Literal["fact", "decision", "learning", "preference", "pattern", "incident",
                "runbook", "context", "observation", "convention", "command", "note"],
        Field(description=MEMORY_TYPE_HELP)] = "fact",
    entity: Annotated[
        str | None,
        Field(description="Entity this memory belongs to; connects it in the graph "
                          "and enables graph_boost / find_connections.")] = None,
    importance: Annotated[
        float,
        Field(description="Importance in 0.0-1.0 (default 0.5). Scales the "
                          "decay weight, so low-importance memories fade "
                          "faster. A default of 1.0 made decay nearly "
                          "ineffective; raise it for facts that must stay "
                          "prominent.")] = 0.5,
    metadata: Annotated[
        Any,
        Field(json_schema_extra={"type": ["object", "null"]},
              description="Free-form JSON object stored with the memory, "
                          "e.g. {\"env\": \"test\"}. Must be an object; a "
                          "JSON-encoded string of an object is accepted and "
                          "parsed.")] = None,
    scope: Annotated[
        Literal["global", "project"] | None,
        Field(description="Where to store the memory: 'project' (requires a "
                          "project) or 'global' (shared table). Defaults to "
                          "'project' when a project is given, else 'global'.")] = None,
    project: Annotated[
        str | None,
        Field(description="Project scope. 'global' is reserved and means the "
                          "shared scope.")] = None,
    supersede: Annotated[
        str | None,
        Field(description="Memory id that this fact replaces; the old memory is "
                          "marked as superseded.")] = None,
    check_contradictions: Annotated[
        bool,
        Field(description="Check the new fact (NLI) against the nearest active "
                          "memories of the scope and auto-supersede "
                          "contradictions. With an entity, that entity's "
                          "memories are candidates first; without one, all "
                          "entities compete. Needs a configured LLM provider.")] = False,
    fuzzy_threshold: Annotated[
        float | None,
        Field(description="Similarity threshold 0.0-1.0 for semantic dedup "
                          "(default 0.85). Only used by dedupe='semantic'.")] = None,
    dedupe: Annotated[
        Literal["exact", "semantic", "off"],
        Field(description="Duplicate detection: 'exact' (content hash, default), "
                          "'semantic' (also vector similarity) or 'off'.")] = "exact",
    supersede_content: Annotated[
        str | None,
        Field(description="Exact content of the memory this fact replaces. "
                          "Matches the target scope first, then the global "
                          "scope (a global match hides that fact in every "
                          "scope and is reported as a cross-scope supersede).")] = None,
    supersede_last: Annotated[
        bool,
        Field(description="Replace the memory that best matches the new fact "
                          "(most similar, not the newest) - of the entity, "
                          "or across the whole scope when no entity is set. "
                          "Only replaces it when the new fact is about "
                          "the same topic (similarity >= 0.5); otherwise the "
                          "fact is stored additionally and a warning is "
                          "returned.")] = False,
    auto_supersede_near_duplicate: Annotated[
        bool,
        Field(description="When the new fact is a reworded copy of an "
                          "active memory (near-duplicate similarity >= 0.75), "
                          "replace that memory instead of storing a "
                          "duplicate; the response lists it in "
                          "`superseded`. Backwards-compatible alias for "
                          "`on_near_duplicate='supersede'`.")] = False,
    on_near_duplicate: Annotated[
        Literal["warn", "supersede", "reject"],
        Field(description="What to do with a near-duplicate that dedupe did "
                          "not reject: 'warn' stores it and warns (default), "
                          "'supersede' replaces the old memory, 'reject' "
                          "stores nothing and answers `duplicate` with "
                          "`duplicate_mode='near'`.")] = "warn",
    metadata_mode: Annotated[
        Literal["merge", "replace"],
        Field(description="Metadata handling when this fact supersedes "
                          "another memory: 'merge' (default) carries keys the "
                          "new fact does not set from the old memory, "
                          "'replace' ignores the old metadata entirely.")] = "merge",
    expires_at: Annotated[
        str | None,
        Field(description="Optional ISO 8601 timestamp (e.g. '2027-01-15' or "
                          "'2026-10-25T12:00:00Z') when this memory stops being "
                          "active (TTL). Expired memories disappear from all "
                          "active reads and dedupe; the row itself stays until "
                          "it is pruned or soft-forgotten. Omit for an "
                          "open-ended memory.")] = None,
    links: Annotated[
        list[dict[str, Any]] | None,
        Field(description="Optional graph edges to create in the same call "
                          "(instead of a separate `relate` round trip). Each "
                          "item needs 'to_entity' and 'relation_type'; "
                          "optional 'from_entity' (default: the memory's "
                          "entity), 'weight' (0.0-1.0) and 'context'.")] = None,
) -> dict[str, Any]:
    """Store a memory (fact, decision, learning, preference, ...).

    Automatically embedded for vector search. Attach to an entity for graph connectivity.
    If `check_contradictions` is true, returns warnings for conflicting existing facts.
    Use `supersede` to replace an existing memory by ID, `supersede_content` to
    replace the fact with the same wording, or `supersede_last` to replace the
    best-matching fact of the same entity (most similar, not the newest);
    without an entity the best-matching memory of the whole scope is replaced
    (same topic guard).
    `supersede_last` refuses to replace a fact
    that is about a different topic (it would destroy it); the new fact is
    stored additionally and the response carries a warning.
    `on_near_duplicate` decides what happens when dedupe did not reject a
    reworded copy: "warn" (default) stores it and warns, "supersede" replaces
    the old memory, "reject" stores nothing. `auto_supersede_near_duplicate`
    is the backwards-compatible alias for "supersede".
    `metadata_mode="merge"` (default) carries metadata keys of a replaced
    memory that the new fact does not set; "replace" drops them.
    `dedupe` selects how duplicates are detected: "exact" (default, content
    hash), "semantic" (also vector similarity, see `fuzzy_threshold`) or "off".
    A duplicate answer names the mode that matched ("exact" or "semantic") and
    the measured similarity. A stored memory that is still very similar to an
    active one (but not similar enough to dedupe) returns a `near-duplicate`
    warning naming the other id.
    A project write that invalidates a shared (global) fact returns a
    `cross-scope supersede` warning plus `cross_scope_superseded`: the global
    fact is hidden in every scope now, not only in this project.
    `scope` defaults to "project" when a project is given, else "global".
    scope="global" stores in the shared table even when a project is given.
    `links` creates graph edges in the same call: each item is a `relate`
    edge whose `from_entity` defaults to the memory's `entity`. Edge failures
    are warnings, never a lost fact.

    Returns the standard envelope: {"ok", "status", "id", "error", "warnings"}.
    `metadata` must be a JSON object, e.g. {"env": "test"} - a JSON-encoded
    string of an object is also accepted; any other type fails.
    """
    logger.info("remember entity=%s mem_type=%s importance=%s check_contradictions=%s fuzzy=%s scope=%s dedupe=%s",
                entity, mem_type, importance, check_contradictions, fuzzy_threshold, scope, dedupe)
    raw_project = project
    try:
        project = _norm_project(project)
        entity = _norm_entity(entity)
    except ValueError as e:
        return fail(STATUS_ERROR, str(e))
    if mem_type not in MEMORY_TYPES:
        return fail(STATUS_ERROR,
                    f"Invalid mem_type. Choose from: {', '.join(sorted(MEMORY_TYPES))}")
    if scope is not None and scope not in VALID_SCOPES:
        return fail(STATUS_ERROR,
                    f"Invalid scope. Choose from: {', '.join(sorted(VALID_SCOPES))}")
    if dedupe not in DEDUPE_MODES:
        return fail(STATUS_ERROR,
                    f"Invalid dedupe. Choose from: {', '.join(sorted(DEDUPE_MODES))}")
    if on_near_duplicate not in ("warn", "supersede", "reject"):
        return fail(STATUS_ERROR,
                    "on_near_duplicate must be 'warn', 'supersede' or 'reject'")
    if metadata_mode not in ("merge", "replace"):
        return fail(STATUS_ERROR,
                    "metadata_mode must be 'merge' or 'replace'")
    near_policy = on_near_duplicate
    if auto_supersede_near_duplicate and on_near_duplicate == "warn":
        near_policy = "supersede"
    content = (content or "").strip()
    if not content:
        return fail(STATUS_ERROR, "content must not be empty")
    try:
        importance = float(importance)
    except (TypeError, ValueError):
        return fail(STATUS_ERROR, "importance must be a number between 0.0 and 1.0")
    if not 0.0 <= importance <= 1.0:
        return fail(STATUS_ERROR, "importance must be between 0.0 and 1.0")
    if fuzzy_threshold is not None and not 0.0 <= fuzzy_threshold <= 1.0:
        return fail(STATUS_ERROR, "fuzzy_threshold must be between 0.0 and 1.0")
    if expires_at is not None:
        try:
            datetime.fromisoformat(str(expires_at).replace("Z", "+00:00"))
        except ValueError:
            return fail(STATUS_ERROR,
                        "expires_at must be an ISO 8601 timestamp "
                        "(e.g. '2027-01-15' or '2027-01-15T12:00:00Z')")
    if metadata is not None:
        metadata, meta_error = _coerce_metadata(metadata)
        if meta_error:
            return fail(STATUS_ERROR, meta_error)
    per_link_errors = _validate_links(links, entity)
    if per_link_errors:
        return fail(STATUS_ERROR, "; ".join(per_link_errors))
    effective_scope = scope or ("project" if project else "global")
    if effective_scope == "project" and not project:
        if str(raw_project or "").strip().lower() == "global":
            return fail(
                STATUS_ERROR,
                "project='global' is the shared scope and cannot be combined "
                "with scope='project'; pass a real project name, or use "
                "scope='global' (the default when no project is set)")
        return fail(STATUS_ERROR, "project required when scope='project'")
    store_project = project if effective_scope == "project" else None

    meta = metadata or {}
    ttl_warning = _ttl_past_warning(expires_at)
    emb = embed([content])[0]
    conn = get_conn()
    try:
        cur = conn.cursor()
        created_scope = bool(store_project) and not _project_exists(
            cur, store_project)
        if store_project:
            _ensure_project_table(cur, store_project)
        scope_warning = (
            f"created a new project scope {store_project!r}; it did not exist "
            f"before this write (check the name if this was a typo, otherwise "
            f"the scope now appears in init_context(project='*'))"
            if created_scope else None)

        existing = _deduplicate(cur, content, store_project, emb, fuzzy_threshold,
                                dedupe=dedupe)
        if existing:
            existing_id, mode, similarity = existing
            detail = ("exact content hash" if mode == "exact"
                      else f"semantic match (similarity {similarity})")
            if mode == "exact" and dedupe == "semantic":
                detail += "; the content hash short-circuits the semantic check"
            warnings = [f"memory already exists; nothing stored ({detail})"]
            if scope_warning:
                warnings.append(scope_warning)
            superseded_now: list[str] = []
            cross_now: list[str] = []
            # A duplicate must not swallow a supersede instruction: the fact is
            # already stored, but the caller still asked to invalidate a memory.
            # Apply the replacement through the existing id and say so.
            if supersede or supersede_content or supersede_last:
                try:
                    (superseded_now, _self_targets, extra_warnings,
                     cross_now) = _apply_duplicate_supersede(
                        cur, store_project, existing_id, entity, supersede,
                        supersede_content, supersede_last, emb, content)
                except ValueError as e:
                    conn.rollback()
                    return fail(STATUS_ERROR, str(e))
                warnings.extend(extra_warnings)
                if superseded_now:
                    conn.commit()
            payload: dict[str, Any] = {
                "duplicate_mode": mode, "similarity": similarity,
                "scope": effective_scope,
                "project": (store_project or "global"),
            }
            if superseded_now:
                payload["superseded"] = superseded_now
                if cross_now:
                    payload["cross_scope_superseded"] = cross_now
                    warnings.append(_cross_scope_warning(cross_now))
            link_results, link_warnings = _apply_remember_links(
                cur, links, entity, store_project)
            warnings.extend(link_warnings)
            if link_results:
                payload["links"] = link_results
            conn.commit()
            return ok(STATUS_DUPLICATE, id=existing_id,
                      warnings=warnings, **payload)
        # Dedupe only covers the target table. The same content may already be
        # active in another scope; reads then hide the copy, so the write has
        # to say that it forked the fact across scopes. (Filled after the
        # supersede targets are known, so a copy this write replaces is not
        # reported as a fork.)
        cross_duplicates: list[dict[str, Any]] = []

        # No active duplicate, but the content may match a memory that was
        # superseded earlier. Storing it would silently re-activate an
        # outdated fact, so report the match (without blocking the write).
        superseded_match = _find_superseded_match(
            cur, content, store_project, emb, fuzzy_threshold, dedupe=dedupe)

        # Also no dedupe match - but possibly still a reworded copy. The
        # thresholds above call it a different fact; this check only reports
        # the nearest active memory above the near-duplicate floor so the
        # response can warn. dedupe='off' means "store everything", so it
        # also skips this warning.
        near_dup = (_find_near_duplicate(cur, content, store_project, emb)
                    if (dedupe != "off" or near_policy != "warn")
                    else None)

        contradictions = []
        contradiction_skip: str | None = None
        nli_unavailable = check_contradictions and nli_configured() is False
        if check_contradictions and not nli_unavailable:
            try:
                contradictions = _check_contradictions(
                    cur, content, entity, store_project, emb
                )
            except Exception as e:
                # A failed NLI must not kill the write: the fact is stored
                # unencumbered, but the skipped check is surfaced loudly so
                # the caller does not mistake it for a clean scan.
                logger.warning("contradiction check failed: %s", e)
                contradictions = []
                contradiction_skip = f"NLI unavailable: {e}"

        # Resolve the replaced memories before storing: "supersede_last" must
        # see the state before this fact exists.
        superseded: list[str] = []
        guard_warnings: list[str] = []
        try:
            superseded, guard_warnings = _supersede_targets(
                cur, store_project, entity, supersede, supersede_content,
                supersede_last, emb, content
            )
        except ValueError as e:
            conn.rollback()
            return fail(STATUS_ERROR, str(e))

        cross_duplicates = [
            dup for dup in _cross_scope_duplicates(cur, content, store_project)
            if dup["id"] not in superseded
        ]

        # The near-duplicate policy turns the warning into an action. `reject`
        # writes nothing at all - the caller explicitly asked not to fork a
        # reworded copy and expects a duplicate answer instead.
        auto_near: dict | None = None
        same_subject = (near_dup is not None
                        and _same_subject_entity(near_dup, entity))
        near_conflict = bool(
            same_subject
            and _content_conflict(content, near_dup.get("content") or ""))
        if near_dup and same_subject and near_policy == "reject" \
                and near_dup["id"] not in superseded:
            conn.rollback()
            return ok(
                STATUS_DUPLICATE, id=near_dup["id"],
                duplicate_of=near_dup["id"], duplicate_mode="near",
                similarity=near_dup["similarity"],
                scope=effective_scope,
                project=(store_project or "global"),
                warnings=[
                    f"near-duplicate rejected: active memory {near_dup['id']} "
                    f"matches this content with similarity "
                    f"{near_dup['similarity']}"
                    + (", and a number or negation differs - this may be a "
                       "contradiction rather than a reworded copy"
                       if near_conflict else "")
                    + "; nothing was stored (pass on_near_duplicate="
                      "'supersede' to replace it or 'warn' to store it "
                      "anyway)"])
        if near_dup and same_subject and near_policy == "supersede" \
                and near_dup["id"] not in superseded:
            auto_near = near_dup
            superseded.append(near_dup["id"])

        mem_id = _store_memory(cur, content, emb, mem_type, entity,
                               importance, meta, store_project, None,
                               scope=effective_scope, expires_at=expires_at)

        for old_id in superseded:
            _supersede(cur, store_project, mem_id, old_id)

        for c in contradictions:
            eid = c["existing_id"]
            if eid not in superseded:
                _supersede(cur, store_project, mem_id, eid)
                superseded.append(eid)

        # A project write that invalidates a SHARED (global) fact hides that
        # fact in every scope, not only in this project. Without this check
        # the response reads like a project-local change, and the agent keeps
        # trusting a global fact this write just invalidated.
        cross_scope_superseded = (
            [old for old in superseded
             if _scope_of_id(cur, old, store_project) == "global"]
            if store_project else [])

        # Keep context when a replaced fact is superseded: keys of the old
        # metadata that the new memory does not set are merged into it, so a
        # superseding fact does not silently drop provenance (tickets, ...)
        # on the previous fact. Keys set explicitly by the caller win.
        carried_keys: list[str] = []
        if superseded and metadata_mode == "merge":
            table = _mem_table(store_project)
            for old_id in superseded:
                cur.execute(
                    f"SELECT metadata FROM {table} "
                    f"WHERE id = %s AND deleted_at IS NULL",
                    (old_id,))
                row = cur.fetchone()
                if not row and store_project:
                    # A cross-scope supersede replaced a shared fact: its
                    # metadata lives in the global table, not in the project
                    # one. The promised merge must not stop at the scope
                    # boundary either.
                    cur.execute(
                        "SELECT metadata FROM memories "
                        "WHERE id = %s AND deleted_at IS NULL",
                        (old_id,))
                    row = cur.fetchone()
                if not row:
                    continue
                try:
                    old_meta = (row[0] if isinstance(row[0], dict)
                                else json.loads(row[0] or "{}"))
                except (TypeError, ValueError):
                    continue
                if not isinstance(old_meta, dict):
                    continue
                for key, value in old_meta.items():
                    if key not in meta and key not in carried_keys:
                        meta[key] = value
                        carried_keys.append(key)
            if carried_keys:
                cur.execute(
                    f"UPDATE {table} SET metadata = %s::jsonb WHERE id = %s",
                    (json.dumps(meta), mem_id))

        # The graph edges belong to the SAME transaction as the memory: one
        # commit stores the fact and its links together, and a savepoint
        # keeps an edge failure from losing the fact.
        link_results, link_warnings = _apply_remember_links(
            cur, links, entity, store_project)
        conn.commit()

        warnings: list[str] = []
        warnings.extend(guard_warnings)
        warnings.extend(link_warnings)
        if scope_warning:
            warnings.append(scope_warning)
        if ttl_warning:
            warnings.append(ttl_warning)
        if superseded_match:
            mode = superseded_match["mode"]
            detail = ("the same content" if mode == "exact"
                      else f"a similar content (similarity "
                           f"{superseded_match['similarity']})")
            warnings.append(
                f"this fact matches a superseded memory "
                f"({superseded_match['id']}) that was replaced by "
                f"{superseded_match['superseded_by']} ({detail} was already "
                f"invalidated). The new fact was stored as active, so both "
                f"versions are reachable now; verify this revert is intended"
            )
        if auto_near:
            warnings.append(
                f"auto-supersede: replaced near-duplicate "
                f"{near_dup['id']} (similarity {near_dup['similarity']}) "
                f"instead of storing a reworded copy"
                + ("; a number or negation differs, so verify this was an "
                   "intended update and not a contradiction"
                   if near_conflict else ""))
        elif near_dup and near_dup["id"] not in superseded:
            # A near-duplicate that some other part of this write already
            # replaced (explicit supersede or the contradiction check) needs
            # no extra advice: the `superseded` list already names it.
            warnings.append(
                _near_duplicate_warning(near_dup, entity, near_conflict))
        if cross_scope_superseded:
            warnings.append(_cross_scope_warning(cross_scope_superseded))
        if cross_duplicates:
            warnings.append(_cross_scope_duplicate_warning(cross_duplicates))
        if contradictions:
            warnings.append(
                f"superseded {len(contradictions)} contradicting memory/facts")
        elif nli_unavailable:
            warnings.append("contradiction check skipped: NLI provider is not configured")
        elif contradiction_skip:
            warnings.append(f"contradiction check skipped: {contradiction_skip}")
        if check_contradictions and not entity and not nli_unavailable \
                and not contradiction_skip:
            # Independent of the outcome: even a scan that found something
            # compared the fact against nearest memories across all entities
            # when no entity was set. Hiding it behind `elif` made the
            # broader comparison silent exactly when it had consequences.
            warnings.append(
                "no entity set: the contradiction check compared the fact "
                "against the nearest memories across all entities")
        auto_count = 1 if auto_near else 0
        if superseded and not check_contradictions:
            warnings.append(f"superseded {len(superseded)} memory/facts")
        if carried_keys:
            carried_detail = ", ".join(
                f"{key}={meta.get(key)!r}" for key in sorted(carried_keys))
            warnings.append(
                "metadata carried over from the replaced memories (these "
                "values come from the old fact; verify they still apply): "
                + carried_detail)
        payload: dict[str, Any] = {"superseded": superseded,
                                   # The scope the fact really landed in. An
                                   # agent that asked for a project but wrote
                                   # to the shared scope has to know that:
                                   # later reads and deletes need that name.
                                    "scope": effective_scope,
                                    "project": (store_project or "global")}
        if cross_scope_superseded:
            # Machine-readable: which shared facts this project write hid.
            payload["cross_scope_superseded"] = cross_scope_superseded
        if cross_duplicates:
            # Machine-readable: other scopes already hold this exact content.
            payload["cross_scope_duplicates"] = cross_duplicates
        if check_contradictions:
            contradictions_payload: dict[str, Any] = {
                "found": bool(contradictions),
                "superseded": contradictions,
            }
            skip = (contradiction_skip
                    or ("NLI provider is not configured"
                        if nli_unavailable
                        else None))
            if skip:
                contradictions_payload["skipped"] = skip
            if not entity and not skip:
                # Transparency: the check ran, but beyond the entity tag.
                contradictions_payload["cross_entity"] = True
            payload["contradictions"] = contradictions_payload
        if link_results:
            payload["links"] = link_results
        return ok(STATUS_STORED, id=mem_id, warnings=warnings, **payload)
    except Exception as e:
        conn.rollback()
        logger.exception("remember failed")
        return fail(STATUS_ERROR, str(e))
    finally:
        conn.close()


def _norm_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _get_field(link: dict, key: str) -> Any:
    return link.get(key)


def _validate_links(links: Any, default_entity: str | None) -> list[str]:
    """Validate `remember(links=[...])` before anything is stored.

    A malformed link is an argument error, not a warning: the caller built
    the call around memory + edges, and a silent half-result would need a
    second discovery round trip. The per-item errors name the index, and the
    item keys and ranges match the `relate` tool exactly.
    """
    if links is None:
        return []
    if not isinstance(links, list):
        return ["links must be a list of edge objects ({to_entity, "
                "relation_type, weight?, context?, from_entity?})"]
    errors: list[str] = []
    for i, link in enumerate(links):
        if not isinstance(link, dict):
            errors.append(f"links[{i}] must be an object")
            continue
        to_entity = _norm_str(_get_field(link, "to_entity"))
        from_entity = _norm_str(_get_field(link, "from_entity"))
        if not to_entity:
            errors.append(f"links[{i}]: 'to_entity' is required")
        if not from_entity and not default_entity:
            errors.append(
                f"links[{i}]: 'from_entity' is required when the memory "
                f"has no entity")
        relation_type = _norm_str(_get_field(link, "relation_type"))
        if not relation_type:
            errors.append(f"links[{i}]: 'relation_type' is required")
        else:
            try:
                _safe_ident(relation_type, "relation_type")
            except ValueError as e:
                errors.append(f"links[{i}]: {e}")
        weight = link.get("weight", 1.0)
        try:
            _validate_weight(weight)
        except (TypeError, ValueError) as e:
            errors.append(f"links[{i}]: weight: {e}")
        context = link.get("context")
        if context is not None and not isinstance(context, str):
            errors.append(f"links[{i}]: 'context' must be a string")
    return errors


def _prepare_links(links: list[dict], default_entity: str | None) -> list[dict]:
    """Normalize `links` items into `_relate_edges` edge dicts."""
    edges: list[dict] = []
    for link in links:
        edges.append({
            "from_entity": (_norm_str(link.get("from_entity"))
                            or default_entity),
            "to_entity": link.get("to_entity").strip()
            if isinstance(link.get("to_entity"), str) else "",
            "relation_type": link.get("relation_type"),
            "weight": _validate_weight(link.get("weight", 1.0)),
            "context": link.get("context"),
        })
    return edges


def _apply_remember_links(
    cur,
    links: list[dict] | None,
    default_entity: str | None,
    store_project: str | None,
) -> tuple[list[dict], list[str]]:
    """Apply `remember(links=[...])` inside the caller's transaction.

    The edges live in the SAME transaction as the memory: a second
    connection would block on the locks this connection already holds
    (registry/graph), turning a combined write into a deadlock. An AGE
    failure rolls back to the savepoint only - the stored fact survives and
    the failure is a warning, never a lost fact.
    """
    results: list[dict] = []
    warnings: list[str] = []
    if not links:
        return results, warnings
    edges_added = False
    try:
        cur.execute("SAVEPOINT remember_links")
        cur.execute("SET search_path TO ag_catalog, public")
        edges_added = True
        for edge in _prepare_links(links, default_entity):
            for lr in _relate_edges(cur, [edge], store_project):
                entry = dict(lr)
                if lr.get("self_loop"):
                    warnings.append(
                        f"links: self-loop {entry['from_entity']!r} ->["
                        f"{entry['relation_type']!r}]-> itself; graph "
                        "traversal cannot leave that node through this edge")
                    entry.pop("self_loop", None)
                results.append(entry)
        cur.execute("RELEASE SAVEPOINT remember_links")
    except Exception as e:
        if edges_added:
            cur.execute("ROLLBACK TO SAVEPOINT remember_links")
            cur.execute("RELEASE SAVEPOINT remember_links")
        warnings.append(f"links failed: {e}")
    return results, warnings


def _supersede_targets(cur, store_project: str | None, entity: str | None,
                       supersede: str | None, supersede_content: str | None,
                       supersede_last: bool,
                       embedding: list[float] | None = None,
                       content: str | None = None
                       ) -> tuple[list[str], list[str]]:
    """Resolve which existing memories a new fact replaces.

    Returns the list of target ids and a list of warnings. `supersede_last`
    only replaces the most similar memory of the entity when the new fact is
    about the same topic: replacing an unrelated fact would destroy it
    silently.
    """
    targets: list[str] = []
    warnings: list[str] = []
    if supersede:
        # The id is used as given, but it must exist: reporting a
        # replacement of a memory that is not there is a lie the caller
        # would act on.
        found = _find_by_id(cur, supersede, store_project)
        if found:
            targets.append(found)
        else:
            warnings.append(
                f"supersede skipped: no memory with id {supersede!r} in the "
                f"target scope; nothing was replaced"
            )
    if supersede_content:
        found = _find_by_content(cur, supersede_content, store_project, entity)
        if not found and store_project and not entity:
            # A global fact is replaced by its exact content from inside a
            # project too (like `supersede=<id>` already does with ids); the
            # replacement hides the global fact in every scope, which the
            # response names. The entity filter stays out of the fallback:
            # the caller said "replace this text", and the exact content is
            # the identity of the target.
            c_hash = _content_hash((supersede_content or "").strip())
            found = _by_content_hash(cur, "memories", c_hash, None)
        if found:
            targets.append(found)
        else:
            warnings.append(
                f"supersede_content skipped: no active memory with exactly "
                f"this content ({supersede_content[:80]!r}) in the target "
                f"scope; nothing was replaced"
            )
    if supersede_last:
        found_entity: str | None = None
        if entity:
            found = _find_last_for_entity(cur, entity, store_project, embedding)
        else:
            # No entity: the best match across the whole scope. The
            # similarity guard still decides - a scope-wide fallback must
            # not replace an unrelated memory either.
            found = _find_last_in_scope(cur, store_project, embedding)
            if found:
                found_entity = found[3] if len(found) > 3 else None
                warnings.append(
                    "supersede_last without an entity: resolved the "
                    "best-matching memory across the scope instead of one "
                    "entity")
        if not found and not entity:
            warnings.append(
                "supersede_last skipped: no embedding available, so the "
                "same-topic match without an entity cannot be resolved; "
                "nothing was replaced")
        elif not found:
            warnings.append(
                f"supersede_last skipped: no memory of entity {entity!r} "
                f"is active; nothing was replaced")
        if found:
            # The scope-wide lookup returns an entity; the entity-scoped one
            # necessarily matched the requested entity.
            target_id, old_content, similarity = found[:3]
            topic_floor = (SUPERSEDE_SIMILARITY_MIN if entity
                           else max(SUPERSEDE_SIMILARITY_MIN,
                                    NEAR_DUPLICATE_THRESHOLD))
            if not entity and found_entity is not None:
                # An untagged write must not replace a fact that belongs to a
                # tagged entity just because the vectors are close. The caller
                # has to name the entity (or the id) to make that choice.
                warnings.append(
                    f"supersede_last skipped: the best scope-wide match "
                    f"{target_id} belongs to entity {found_entity!r}, but the "
                    f"new fact has no entity; pass entity={found_entity!r} or "
                    f"supersede={target_id} to replace it explicitly. The new "
                    f"fact was stored as an additional memory"
                )
            elif similarity is not None and similarity < topic_floor:
                pool = repr(entity) if entity else "the scope"
                warnings.append(
                    f"supersede_last skipped: no memory of "
                    f"{pool}"
                    f" is about the same topic (best match {old_content!r} "
                    f"with similarity {similarity} < {topic_floor}); the new "
                    f"fact was stored as an additional memory"
                )
            elif content is not None and not _shares_subject(
                    content, old_content,
                    entity or found_entity, similarity):
                # Similarity is dominated by the shared sentence template
                # ("the QA service ..."), so a fact about another attribute
                # still scores high. Requiring a shared content word after
                # removing the entity makes the topic check real.
                pool = repr(entity) if entity else "the scope"
                warnings.append(
                    f"supersede_last skipped: the best match in {pool} "
                    f"({old_content!r}, similarity {similarity}) shares no "
                    f"subject term with the new fact - it is a different "
                    f"topic; the new fact was stored as an additional memory"
                )
            else:
                targets.append(target_id)
                if not entity:
                    warnings.append(
                        f"supersede_last: replaced the best-matching memory "
                        f"of the scope (similarity {similarity}); an entity "
                        f"tag would make this targeting explicit")
    # Keep order, drop duplicates.
    return list(dict.fromkeys(targets)), warnings


def _apply_duplicate_supersede(cur, store_project: str | None,
                               existing_id: str, entity: str | None,
                               supersede: str | None,
                               supersede_content: str | None,
                               supersede_last: bool,
                               embedding: list[float] | None,
                               content: str | None = None
                               ) -> tuple[list[str], list[str], list[str],
                                          list[str]]:
    """Apply `supersede` even when the write itself is a duplicate.

    The fact is already stored under `existing_id`; the caller still asked to
    invalidate another memory. Resolving the targets and superseding them with
    the existing id keeps that instruction from being silently dropped. A
    target that *is* the duplicate is not superseded by itself (that would
    create a bogus self-link) and is reported instead.

    Returns (applied, self_targets, warnings, cross_scope_ids).
    """
    targets, warnings = _supersede_targets(
        cur, store_project, entity, supersede, supersede_content,
        supersede_last, embedding, content)
    applied = [t for t in targets if t != existing_id]
    self_targets = [t for t in targets if t == existing_id]
    for old_id in applied:
        _supersede(cur, store_project, existing_id, old_id)
    if self_targets:
        warnings.append(
            f"supersede target is the existing duplicate {existing_id} "
            f"itself; nothing was replaced")
    if applied:
        warnings.append(
            f"supersede applied via the existing duplicate {existing_id}: "
            f"replaced {len(applied)} memory/facts")
    cross_scope = ([old for old in applied
                    if _scope_of_id(cur, old, store_project) == "global"]
                   if store_project else [])
    return applied, self_targets, warnings, cross_scope


def remember_batch(
    facts: Annotated[
        list[BatchFactInput],
        Field(description="Facts to store. Each dict needs 'content'; invalid "
                          "entries are reported individually and the rest is "
                          "still stored.")],
    project: Annotated[
        str | None,
        Field(description="Default project scope for all facts (per-fact "
                          "project wins).")]= None,
    skip_duplicates: Annotated[
        bool,
        Field(description="Skip facts that already exist instead of storing "
                          "them again.")] = True,
    fuzzy_threshold: Annotated[
        float | None,
        Field(description="Similarity threshold 0.0-1.0 for semantic dedup "
                          "across all facts.")] = None,
    check_contradictions: Annotated[
        bool,
        Field(description="NLI-check every fact against the nearest active "
                          "memories of its scope and auto-supersede "
                          "contradictions (facts with an entity get that "
                          "entity's memories first).")] = False,
    scope: Annotated[
        Literal["global", "project"] | None,
        Field(description="Default scope for all facts (per-fact scope wins). "
                          "scope='project' requires a top-level project.")] = None,
    dedupe: Annotated[
        Literal["exact", "semantic", "off"],
        Field(description="Duplicate detection: 'exact' (default), 'semantic' or "
                          "'off'.")] = "exact",
    on_near_duplicate: Annotated[
        Literal["warn", "supersede", "reject"],
        Field(description="Default policy for near-duplicates (per-fact "
                          "'on_near_duplicate' wins): 'warn', 'supersede' or "
                          "'reject'.")] = "warn",
) -> dict[str, Any]:
    """Store multiple facts at once. Each dict needs 'content'.

    Optional keys: mem_type (one of the memory types), entity, importance,
    metadata (a JSON object - a JSON-encoded string of an object is accepted),
    scope ("project" or "global"; 'project' requires a project), project,
    supersede (memory id
    or content), supersede_content, supersede_last, on_near_duplicate
    ("warn"/"supersede"/"reject"),
    auto_supersede_near_duplicate, expires_at (ISO 8601 TTL timestamp),
    fuzzy_threshold, and dedupe ("exact", "semantic", or "off").
    Set skip_duplicates=False to allow storing identical content multiple times.
    Set fuzzy_threshold (0.0-1.0) for semantic dedup across all facts.
    Per-fact 'fuzzy_threshold' overrides the top-level value.
    `dedupe` selects how duplicates are detected: "exact" (default, content
    hash), "semantic" (also vector similarity) or "off".
    A stored fact that is still very similar to an active one returns a
    `near-duplicate` warning in its result entry; a project fact that
    invalidates a shared (global) fact adds `cross_scope_superseded` and a
    `cross-scope supersede` warning to that entry.
    `scope` sets the default scope for all facts; per-fact 'scope' wins.
    Set check_contradictions=True to NLI-check each fact against the nearest
    active memories of its scope and auto-supersede contradictions (an
    entity-scoped fact gets that entity's memories first; without an entity
    all entities compete).
    Invalid facts are reported individually; the remaining facts are stored.
    """
    logger.info("remember_batch count=%s skip_duplicates=%s fuzzy=%s check_contradictions=%s scope=%s dedupe=%s",
                len(facts) if isinstance(facts, list) else -1, skip_duplicates,
                fuzzy_threshold, check_contradictions, scope, dedupe)
    try:
        project = _norm_project(project)
    except ValueError as e:
        return fail(STATUS_ERROR, str(e))
    if scope is not None and scope not in VALID_SCOPES:
        return fail(STATUS_ERROR,
                    f"invalid scope {scope!r}; choose from "
                    f"{', '.join(sorted(VALID_SCOPES))}")
    if dedupe not in DEDUPE_MODES:
        return fail(STATUS_ERROR,
                    f"invalid dedupe {dedupe!r}; choose from "
                    f"{', '.join(sorted(DEDUPE_MODES))}")
    if on_near_duplicate not in ("warn", "supersede", "reject"):
        return fail(STATUS_ERROR,
                    "on_near_duplicate must be 'warn', 'supersede' or "
                    "'reject'")
    if fuzzy_threshold is not None and not 0.0 <= fuzzy_threshold <= 1.0:
        return fail(STATUS_ERROR, "fuzzy_threshold must be between 0.0 and 1.0")
    if not isinstance(facts, list) or not facts:
        return fail(STATUS_ERROR, "facts must be a non-empty list")

    facts = [fact.model_dump(exclude_unset=True) if isinstance(fact, BatchFact) else fact
             for fact in facts]


    # Validate everything before embedding so one malformed fact cannot abort
    # the batch with a KeyError.
    prepared: list[tuple[dict | None, str | None]] = []
    for fact in facts:
        if not isinstance(fact, dict):
            prepared.append((None, "fact must be an object"))
            continue
        if fact.get("_not_an_object"):
            prepared.append((None, "fact must be an object"))
            continue
        content = fact.get("content")
        content = content.strip() if isinstance(content, str) else ""
        if not content:
            prepared.append((None, "content is required"))
            continue
        mem_type = fact.get("mem_type", "fact")
        if mem_type not in MEMORY_TYPES:
            prepared.append((None, f"invalid mem_type {mem_type!r}"))
            continue
        try:
            importance = float(fact.get("importance", 0.5))
        except (TypeError, ValueError):
            prepared.append((None, "importance must be a number between 0.0 and 1.0"))
            continue
        if not 0.0 <= importance <= 1.0:
            prepared.append((None, "importance must be between 0.0 and 1.0"))
            continue
        try:
            fp = _norm_project(fact.get("project", project))
        except ValueError as e:
            # The whole batch must not fail for one malformed fact.
            prepared.append((None, str(e)))
            continue
        fscope = fact.get("scope") or scope or ("project" if fp else "global")
        if fscope not in VALID_SCOPES:
            prepared.append((None, f"invalid scope {fscope!r}"))
            continue
        if fscope == "project" and not fp:
            prepared.append((None, "project required when scope='project'"))
            continue
        try:
            fentity = _norm_entity(fact.get("entity"))
        except ValueError as e:
            prepared.append((None, str(e)))
            continue
        ft = fact.get("fuzzy_threshold", fuzzy_threshold)
        if ft is not None and not 0.0 <= ft <= 1.0:
            prepared.append((None, "fuzzy_threshold must be between 0.0 and 1.0"))
            continue
        fact_near = fact.get("on_near_duplicate") or on_near_duplicate
        if fact_near not in ("warn", "supersede", "reject"):
            prepared.append((None,
                             "on_near_duplicate must be 'warn', 'supersede' "
                             "or 'reject'"))
            continue
        if fact.get("auto_supersede_near_duplicate") and fact_near == "warn":
            fact_near = "supersede"
        meta = fact.get("metadata") or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except json.JSONDecodeError:
                prepared.append((None,
                                 ("metadata is neither an object nor valid "
                                  "JSON - pass an object like "
                                  "{\"env\": \"test\"}")))
                continue
        if not isinstance(meta, dict):
            prepared.append((None,
                             ("metadata must be a JSON object, e.g. "
                              "{\"env\": \"test\"} "
                              f"(got {type(meta).__name__})")))
            continue
        fexpires = fact.get("expires_at")
        if fexpires is not None:
            try:
                datetime.fromisoformat(str(fexpires).replace("Z", "+00:00"))
            except ValueError:
                prepared.append((None,
                                 "expires_at must be an ISO 8601 timestamp"))
                continue
        prepared.append(({
            "content": content,
            "mem_type": mem_type,
            "importance": importance,
            "project": fp if fscope == "project" else None,
            "scope": fscope,
            "entity": fentity,
            "fuzzy_threshold": ft,
            "dedupe": fact.get("dedupe"),
            "metadata": meta,
            "supersede": fact.get("supersede"),
            "supersede_content": fact.get("supersede_content"),
            "supersede_last": bool(fact.get("supersede_last")),
            "on_near_duplicate": fact_near,
            "expires_at": fexpires,
            "ttl_warning": _ttl_past_warning(fexpires),
        }, None))

    valid = [(i, spec) for i, (spec, err) in enumerate(prepared)
             if spec is not None]
    embeddings: dict[int, list[float]] = {}
    if valid:
        try:
            vectors = embed([spec["content"] for _, spec in valid])
        except Exception as e:
            logger.exception("remember_batch embedding failed")
            return fail(STATUS_ERROR, f"embedding failed: {e}")
        embeddings = {i: vec for (i, _), vec in zip(valid, vectors)}

    conn = get_conn()
    results: list[dict] = []
    ensured: set[str] = set()
    try:
        cur = conn.cursor()
        for i, (spec, err) in enumerate(prepared):
            if spec is None:
                raw = facts[i].get("content", "") if isinstance(facts[i], dict) else ""
                results.append({"id": None, "content": raw, "status": "error",
                                "error": err})
                continue
            store_project = spec["project"]
            savepoint = f"fact_{i}"
            cur.execute(f"SAVEPOINT {savepoint}")
            try:
                scope_warning = None
                if store_project and store_project not in ensured:
                    if not _project_exists(cur, store_project):
                        scope_warning = (
                            f"created a new project scope {store_project!r}; "
                            f"it did not exist before this write (check the "
                            f"name if this was a typo)")
                    _ensure_project_table(cur, store_project)
                    ensured.add(store_project)

                near_dup = None
                exact_copy: str | None = None
                fact_dedupe = spec.get("dedupe") or dedupe
                if not skip_duplicates and fact_dedupe != "off":
                    # skip_duplicates=false means "store it anyway", but the
                    # caller still has to learn that an identical active row
                    # already exists. _find_near_duplicate excludes exact
                    # hashes, so only this explicit check can report it.
                    existing_exact = _deduplicate(
                        cur, spec["content"], store_project, None,
                        dedupe="exact")
                    if existing_exact:
                        exact_copy = existing_exact[0]
                if skip_duplicates:
                    existing = _deduplicate(cur, spec["content"], store_project,
                                            embeddings[i], spec["fuzzy_threshold"],
                                            dedupe=fact_dedupe)
                    if existing:
                        existing_id, mode, similarity = existing
                        entry: dict[str, Any] = {
                            "id": existing_id, "content": spec["content"],
                            "status": "duplicate",
                            "duplicate_mode": mode,
                            "similarity": similarity,
                            "scope": spec["scope"],
                            "project": (store_project or "global"),
                        }
                        if scope_warning:
                            entry["warnings"] = [scope_warning]
                        if (spec.get("supersede") or spec.get("supersede_content")
                                or spec.get("supersede_last")):
                            try:
                                (superseded_now, _self_targets,
                                 extra_warnings, cross_now) = (
                                    _apply_duplicate_supersede(
                                        cur, store_project, existing_id,
                                        spec["entity"], spec.get("supersede"),
                                        spec.get("supersede_content"),
                                        spec.get("supersede_last"),
                                        embeddings[i], spec["content"]))
                            except ValueError as exc:
                                cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                                results.append({
                                    "id": existing_id,
                                    "content": spec["content"],
                                    "status": "error", "error": str(exc),
                                    "scope": spec["scope"],
                                    "project": (store_project or "global"),
                                })
                                continue
                            entry.setdefault("warnings", []).extend(
                                extra_warnings)
                            if superseded_now:
                                entry["superseded"] = superseded_now
                                if cross_now:
                                    entry["cross_scope_superseded"] = cross_now
                                    entry["warnings"].append(
                                        _cross_scope_warning(cross_now))
                        cur.execute(f"RELEASE SAVEPOINT {savepoint}")
                        results.append(entry)
                        continue
                    # No duplicate, but maybe a reworded copy: report it in
                    # the entry so a batch write cannot fork silently either.
                    # skip_duplicates=False means "store duplicates anyway",
                    # so the whole check (and its warning) is skipped then.
                    if fact_dedupe != "off" \
                            or spec["on_near_duplicate"] != "warn":
                        near_dup = _find_near_duplicate(
                            cur, spec["content"], store_project, embeddings[i])
                elif spec["on_near_duplicate"] != "warn":
                    near_dup = _find_near_duplicate(
                        cur, spec["content"], store_project, embeddings[i])

                contradictions = []
                contradiction_skip: str | None = None
                if check_contradictions:
                    try:
                        contradictions = _check_contradictions(
                            cur, spec["content"], spec["entity"], store_project,
                            embeddings[i],
                        )
                    except Exception as exc:
                        # A failed NLI must not roll the fact back: report the
                        # skipped check instead of a wrong "clean" result.
                        logger.warning("contradiction check failed: %s", exc)
                        contradiction_skip = f"NLI unavailable: {exc}"

                # Resolve replaced memories before this fact is stored.
                superseded, guard_warnings = _supersede_targets(
                    cur, store_project, spec["entity"], spec["supersede"],
                    spec["supersede_content"], spec["supersede_last"],
                    embeddings[i], spec["content"],
                )
                # Per-fact near-duplicate policy (same semantics as in
                # `remember`): reject stores nothing, supersede replaces.
                # A high similarity across different entities is not a
                # duplicate; the policy must not delete the other subject.
                fact_near_policy = spec["on_near_duplicate"]
                same_fact_subject = (near_dup is not None
                                     and _same_subject_entity(
                                         near_dup, spec["entity"]))
                fact_near_conflict = bool(
                    same_fact_subject
                    and _content_conflict(spec["content"],
                                          near_dup.get("content") or ""))
                if near_dup and same_fact_subject \
                        and fact_near_policy == "reject" \
                        and near_dup["id"] not in superseded:
                    cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                    results.append({
                        "id": near_dup["id"],
                        "content": spec["content"],
                        "status": "duplicate",
                        "duplicate_of": near_dup["id"],
                        "duplicate_mode": "near",
                        "similarity": near_dup["similarity"],
                        "scope": spec["scope"],
                        "project": (store_project or "global"),
                        "warnings": [
                            f"near-duplicate rejected: active memory "
                            f"{near_dup['id']} matches this content with "
                            f"similarity {near_dup['similarity']}"
                            + (", and a number or negation differs - this "
                               "may be a contradiction rather than a "
                               "reworded copy" if fact_near_conflict else "")
                            + "; nothing was stored"],
                    })
                    continue
                if near_dup and same_fact_subject \
                        and fact_near_policy == "supersede" \
                        and near_dup["id"] not in superseded:
                    superseded.append(near_dup["id"])

                cross_duplicates = [
                    dup for dup in _cross_scope_duplicates(
                        cur, spec["content"], store_project)
                    if dup["id"] not in superseded
                ]

                mid = _store_memory(
                    cur, spec["content"], embeddings[i], spec["mem_type"],
                    spec["entity"], spec["importance"], spec["metadata"],
                    store_project, scope=spec["scope"],
                    expires_at=spec.get("expires_at"),
                )

                for old_id in superseded:
                    _supersede(cur, store_project, mid, old_id)

                auto_superseded = []
                for c in contradictions:
                    if c["existing_id"] in superseded:
                        continue
                    _supersede(cur, store_project, mid, c["existing_id"])
                    auto_superseded.append(c)

                # Carry metadata over from every replaced fact (same contract
                # as `remember`): a batch write must not drop provenance that
                # a single write preserves.
                replaced_ids = list(dict.fromkeys(
                    superseded + [c["existing_id"] for c in auto_superseded]))
                carried: list[str] = []
                if replaced_ids:
                    merged_meta = dict(spec["metadata"])
                    for old_id in replaced_ids:
                        for key, value in _read_metadata(
                                cur, store_project, old_id).items():
                            if key not in merged_meta:
                                merged_meta[key] = value
                                carried.append(key)
                    if carried:
                        cur.execute(
                            f"UPDATE {_mem_table(store_project)} "
                            "SET metadata = %s::jsonb WHERE id = %s",
                            (json.dumps(merged_meta), mid))
                # Shared facts this entry invalidated: a project write hides
                # them in every scope, so the entry has to name them.
                cross_scope_superseded = (
                    [old for old in dict.fromkeys(
                        superseded + [c["existing_id"]
                                      for c in auto_superseded])
                     if _scope_of_id(cur, old, store_project) == "global"]
                    if store_project else [])
                cur.execute(f"RELEASE SAVEPOINT {savepoint}")
            except Exception as exc:
                cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                results.append({"id": None, "content": spec["content"],
                                "status": "error", "error": str(exc),
                                "scope": spec["scope"],
                                "project": (store_project or "global")})
                continue

            entry = {"id": mid, "content": spec["content"], "status": "stored",
                     "superseded": superseded,
                     # Where the fact really landed: a batch can mix scopes,
                     # so the caller has to be able to see it per fact.
                     "scope": spec["scope"],
                     "project": (store_project or "global"),
                     # Stable shape: a caller can read entry["warnings"]
                     # without a key check, also when there is nothing to say.
                     "warnings": []}
            if guard_warnings:
                entry["warnings"] = guard_warnings
            if exact_copy:
                entry.setdefault("warnings", []).append(
                    f"identical content is already active as {exact_copy}; "
                    f"another copy was stored because skip_duplicates=false")
            if scope_warning:
                entry.setdefault("warnings", []).append(scope_warning)
            if spec.get("ttl_warning"):
                entry.setdefault("warnings", []).append(spec["ttl_warning"])
            if carried:
                entry.setdefault("warnings", []).append(
                    "metadata carried over from the replaced memories (these "
                    "values come from the old fact; verify they still apply): "
                    + ", ".join(sorted(set(carried))))
            if near_dup and same_fact_subject \
                    and spec["on_near_duplicate"] == "supersede" \
                    and near_dup["id"] in superseded:
                entry.setdefault("warnings", []).append(
                    f"auto-supersede: replaced near-duplicate "
                    f"{near_dup['id']} (similarity {near_dup['similarity']}) "
                    f"instead of storing a reworded copy"
                    + ("; a number or negation differs, so verify this was "
                       "an intended update and not a contradiction"
                       if fact_near_conflict else ""))
            elif near_dup and near_dup["id"] not in superseded:
                # Already replaced through the explicit supersede path: the
                # entry's `superseded` list already names it, so no extra
                # advice is needed.
                entry.setdefault("warnings", []).append(
                    _near_duplicate_warning(near_dup, spec["entity"],
                                            fact_near_conflict))
            if cross_scope_superseded:
                entry["cross_scope_superseded"] = cross_scope_superseded
                entry.setdefault("warnings", []).append(
                    _cross_scope_warning(cross_scope_superseded))
            if cross_duplicates:
                entry["cross_scope_duplicates"] = cross_duplicates
                entry.setdefault("warnings", []).append(
                    _cross_scope_duplicate_warning(cross_duplicates))
            if check_contradictions:
                entry["contradictions"] = {
                    "found": bool(auto_superseded),
                    "superseded": auto_superseded,
                }
                if contradiction_skip:
                    entry["contradictions"]["skipped"] = contradiction_skip
                    entry.setdefault("warnings", []).append(
                        f"contradiction check skipped: {contradiction_skip}")
                elif not spec["entity"]:
                    # Transparency: the check ran, but across all entities.
                    entry["contradictions"]["cross_entity"] = True
                    entry.setdefault("warnings", []).append(
                        "no entity set: the contradiction check compared the "
                        "fact against the nearest memories across all entities")
            results.append(entry)
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.exception("remember_batch failed")
        return fail(STATUS_ERROR, str(e))
    finally:
        conn.close()
    failed = [r for r in results if r.get("status") == "error"]
    succeeded = len(results) - len(failed)
    if failed and len(failed) == len(results):
        return fail(STATUS_ERROR, "no fact could be stored",
                    count=len(results), succeeded=0, failed=len(failed),
                    results=results)
    status = STATUS_PARTIAL if failed else STATUS_OK
    warnings = [f"{len(failed)} fact(s) failed"] if failed else []
    return ok(status, count=len(results), succeeded=succeeded,
              failed=len(failed), results=results, warnings=warnings)


def _restore_blocked_reason(cur, memory_id: str,
                            tables: list[str]) -> tuple[str, str] | None:
    """Status and reason for an existing row that cannot be restored.

    "not found" was a lie for an active memory: the id exists, it just has
    nothing to restore. Such a request answers `conflict`; an unknown id stays
    `not_found`. A superseded history row is a conflict too - a restore never
    undoes a supersede.
    """
    for table in tables:
        cur.execute(
            f"SELECT deleted_at, superseded_by FROM {table} WHERE id = %s",
            (memory_id,))
        row = cur.fetchone()
        if not row:
            continue
        deleted_at, superseded_by = row
        if deleted_at is None:
            if superseded_by is not None \
                    and superseded_by != SENTINEL_FORGOTTEN:
                return (
                    STATUS_CONFLICT,
                    f"memory {memory_id} exists but is not soft-forgotten: "
                    f"it is superseded by {superseded_by}, and a restore "
                    f"does not undo a supersede (read it with "
                    f"include_superseded=true)")
            return (
                STATUS_CONFLICT,
                f"memory {memory_id} exists but is not soft-forgotten (it "
                f"is active); there is nothing to restore")
        return (
            STATUS_CONFLICT,
            f"memory {memory_id} exists but its deleted_at is not a "
            f"restorable soft delete")
    return None


def _restore_memory(memory_id: str, project: str | None) -> dict[str, Any]:
    """Clear the soft delete of one memory (inverse of `forget(soft=True)`).

    A real `superseded_by` id is kept: a restored memory cannot resurrect
    next to the fact that replaced it. With a project only that project's
    table is searched (the explicit scope); without one every scope is
    searched and an ambiguous id is refused.
    """
    conn = get_conn()
    try:
        cur = conn.cursor()
        if project:
            tables = [_mem_table(project)]
            if not _table_exists(cur, tables[0]):
                return fail(STATUS_NOT_FOUND, f"NOT FOUND: {memory_id}",
                            id=memory_id)
        else:
            scopes = _memory_scopes(cur, memory_id)
            if len(scopes) > 1:
                conn.rollback()
                return fail(STATUS_ERROR,
                            f"ambiguous memory id {memory_id}: it exists in "
                            f"the scopes {scopes}; pass project=<name> to say "
                            f"which one to restore",
                            id=memory_id, scopes=scopes)
            tables = _all_mem_tables(cur)
        restored = 0
        restored_table = None
        for table in tables:
            # NULLIF clears only the forgotten sentinel; a `superseded_by` id
            # from a real supersede stays set, so a restored memory cannot
            # resurrect next to the fact that replaced it.
            cur.execute(
                f"UPDATE {table} SET deleted_at = NULL, "
                f"superseded_by = NULLIF(superseded_by, %s), "
                f"updated_at = NOW() WHERE id = %s AND deleted_at IS NOT NULL",
                (SENTINEL_FORGOTTEN, memory_id),
            )
            if cur.rowcount:
                restored += cur.rowcount
                restored_table = table
                break
        conn.commit()
        if not restored:
            blocked = _restore_blocked_reason(cur, memory_id, tables)
            if blocked:
                status, reason = blocked
                return fail(status, reason, id=memory_id)
            return fail(STATUS_NOT_FOUND, f"NOT FOUND: {memory_id}",
                        id=memory_id)
        scope = _scope_of_table(cur, restored_table) if restored_table else None
    except Exception as e:
        conn.rollback()
        return fail(STATUS_ERROR, str(e), id=memory_id)
    finally:
        conn.close()
    return ok(STATUS_RESTORED, id=memory_id, project=scope)


def update_memory(
    memory_id: Annotated[
        str,
        Field(description="Id of the memory to update.")],
    content: Annotated[
        str | None,
        Field(description="New content; re-embeds the memory.")] = None,
    importance: Annotated[
        float | None,
        Field(description="New importance in 0.0-1.0.")] = None,
    metadata: Annotated[
        Any,
        Field(json_schema_extra={"type": ["object", "null"]},
              description="Metadata object, e.g. {\"env\": \"test\"}. A "
                          "JSON-encoded string of an object is accepted and "
                          "parsed. Replaces the stored object unless "
                          "metadata_mode='merge'.")] = None,
    metadata_mode: Annotated[
        Literal["replace", "merge"],
        Field(description="'replace' (default) writes the metadata object as "
                          "given; 'merge' keeps stored keys that the new "
                          "object does not set.")] = "replace",
    clear_metadata_keys: Annotated[
        list[str] | None,
        Field(description="Metadata keys to remove. Can be combined with "
                          "metadata and metadata_mode.")] = None,
    project: Annotated[
        str | None,
        Field(description="Project scope. Without it the id is looked up in "
                          "every scope.")] = None,
    entity: Annotated[
        str | None,
        Field(description="New entity the memory belongs to.")] = None,
    mem_type: Annotated[
        str | None,
        Field(description="New memory type. " + MEMORY_TYPE_HELP)] = None,
    dedupe: Annotated[
        Literal["exact", "semantic", "off"],
        Field(description="Duplicate check for a content change: 'exact' "
                          "(content hash, default) refuses an edit that would "
                          "duplicate another active memory, 'semantic' also "
                          "checks vector similarity, 'off' skips the check. A "
                          "near-duplicate below the dedupe threshold is only "
                          "reported as a warning.")] = "exact",
    fuzzy_threshold: Annotated[
        float | None,
        Field(description="Similarity threshold 0.0-1.0 for semantic dedupe "
                          "(default 0.85). Only used by dedupe='semantic'.")] = None,
    expires_at: Annotated[
        str | None,
        Field(description="New ISO 8601 TTL timestamp for the memory.")] = None,
    clear_expires_at: Annotated[
        bool,
        Field(description="Remove the TTL entirely (makes the memory "
                          "open-ended again). Mutually exclusive with "
                          "expires_at.")] = False,
    touch: Annotated[
        bool,
        Field(description="Refresh the memory without changing its content: "
                          "sets updated_at to now so the decay age starts "
                          "over. Can be combined with importance.")] = False,
    restore: Annotated[
        bool,
        Field(description="Undo a soft delete: clears deleted_at (and the "
                          "forgotten sentinel) of a soft-forgotten memory. "
                          "Mutually exclusive with every other update "
                          "field.")] = False,
) -> dict[str, Any]:
    """Update an existing memory's content, importance, metadata, entity, type or TTL.

    Without `project`, the memory is looked up in every scope. A memory id is
    unique per scope, so an id that exists in several scopes (an import copies
    the id) is refused: the response names the scopes, and `project` says
    which one to update.
    `expires_at` sets a new TTL (`clear_expires_at=True` removes it).
    `touch=True` only refreshes the decay age (no content change);
    `restore=True` undoes a soft delete (the inverse of `forget(soft=True)`).
    A real `superseded_by` is never cleared by a restore.
    """
    logger.info("update_memory id=%s has_content=%s has_importance=%s has_metadata=%s",
                memory_id, content is not None, importance is not None, metadata is not None)
    try:
        project = _norm_project(project)
        entity = _norm_entity(entity)
    except ValueError as e:
        return fail(STATUS_ERROR, str(e))
    if dedupe not in DEDUPE_MODES:
        return fail(STATUS_ERROR,
                    f"Invalid dedupe. Choose from: {', '.join(sorted(DEDUPE_MODES))}")
    if fuzzy_threshold is not None and not 0.0 <= fuzzy_threshold <= 1.0:
        return fail(STATUS_ERROR, "fuzzy_threshold must be between 0.0 and 1.0")
    if expires_at is not None and clear_expires_at:
        return fail(STATUS_ERROR,
                    "pass either expires_at or clear_expires_at, not both")
    if metadata_mode not in ("replace", "merge"):
        return fail(STATUS_ERROR,
                    "metadata_mode must be 'replace' or 'merge'")
    clear_keys: list[str] = []
    if clear_metadata_keys is not None:
        if not isinstance(clear_metadata_keys, list) or any(
                not isinstance(key, str) or not key.strip()
                for key in clear_metadata_keys):
            return fail(STATUS_ERROR,
                        "clear_metadata_keys must be a list of non-empty "
                        "strings")
        clear_keys = list(dict.fromkeys(
            key.strip() for key in clear_metadata_keys))
    expires_value: datetime | None = None
    if expires_at is not None:
        try:
            expires_value = datetime.fromisoformat(
                str(expires_at).replace("Z", "+00:00"))
        except ValueError:
            return fail(STATUS_ERROR,
                        "expires_at must be an ISO 8601 timestamp "
                        "(e.g. '2027-01-15' or '2027-01-15T12:00:00Z')")
    if restore:
        if (content is not None or importance is not None
                or metadata is not None or entity is not None
                or mem_type is not None or expires_at is not None
                or clear_expires_at or touch or clear_keys):
            return fail(STATUS_ERROR,
                        "restore=true cannot be combined with other update "
                        "fields; it only clears the soft delete")
        return _restore_memory(memory_id, project)

    sets: list[str] = []
    params: list[Any] = []
    emb: list[float] | None = None

    if content is not None:
        content = (content or "").strip()
        if not content:
            return fail(STATUS_ERROR, "content must not be empty")
        emb = embed([content])[0]
        sets.append("content = %s")
        params.append(content)
        sets.append("content_hash = %s")
        params.append(_content_hash(content))
        sets.append("embedding = %s::vector")
        params.append(emb)
    if importance is not None:
        try:
            importance = float(importance)
        except (TypeError, ValueError):
            return fail(STATUS_ERROR, "importance must be a number between 0.0 and 1.0")
        if not 0.0 <= importance <= 1.0:
            return fail(STATUS_ERROR, "importance must be between 0.0 and 1.0")
        sets.append("importance = %s")
        params.append(importance)
    metadata_value: dict[str, Any] | None = None
    if metadata is not None:
        metadata_value, meta_error = _coerce_metadata(metadata)
        if meta_error:
            return fail(STATUS_ERROR, meta_error)
    if entity is not None:
        sets.append("entity = %s")
        params.append(entity)
    if mem_type is not None:
        if mem_type not in MEMORY_TYPES:
            return fail(STATUS_ERROR,
                        f"Invalid mem_type. Choose from: "
                        f"{', '.join(sorted(MEMORY_TYPES))}")
        sets.append("mem_type = %s")
        params.append(mem_type)
    if expires_at is not None:
        sets.append("expires_at = %s")
        params.append(expires_value)
    elif clear_expires_at:
        sets.append("expires_at = NULL")

    if sets:
        # Updated memories must also age like new ones for decay scoring.
        sets.append("updated_at = NOW()")
    elif touch:
        # A pure refresh: no content change, only the decay age restarts.
        sets.append("updated_at = NOW()")

    conn = get_conn()
    try:
        cur = conn.cursor()
        if project:
            # Prefer the project copy over the shared one; only the first
            # active match is updated.
            tables = [_mem_table(project), "memories"]
        else:
            # A memory id is unique per scope, not globally: an import can
            # copy a memory into another project. Without a project an id
            # that exists in several scopes is ambiguous, and updating an
            # arbitrary one of them would be a coin flip.
            scopes = _memory_scopes(cur, memory_id)
            if len(scopes) > 1:
                conn.rollback()
                return fail(STATUS_ERROR,
                            f"ambiguous memory id {memory_id}: it exists in "
                            f"the scopes {scopes}; pass project=<name> to say "
                            f"which one to update",
                            id=memory_id, scopes=scopes)
            tables = _all_mem_tables(cur)
        # Locate the memory BEFORE the "nothing to update" check: an unknown
        # id must answer NOT FOUND even when no field was passed, otherwise
        # the validation message suggests the id was seen and is valid.
        active_table = None
        for table in tables:
            cur.execute(
                f"SELECT 1 FROM {table} WHERE id = %s "
                "AND superseded_by IS NULL AND deleted_at IS NULL",
                (memory_id,),
            )
            if cur.fetchone():
                active_table = table
                break
        if active_table is None:
            reason = _hidden_state_reason(cur, memory_id, tables)
            conn.rollback()
            if reason:
                return fail(
                    STATUS_NOT_FOUND,
                    f"memory {memory_id} exists but cannot be updated: {reason}",
                    id=memory_id)
            return fail(STATUS_NOT_FOUND, f"NOT FOUND: {memory_id}", id=memory_id)
        if metadata_value is not None or clear_keys:
            # Metadata merge and key removal need the stored object. This is
            # the only field that is computed from the existing row; the
            # caller's new keys win, missing ones survive in 'merge' mode.
            cur.execute(
                f"SELECT metadata FROM {active_table} WHERE id = %s",
                (memory_id,))
            row = cur.fetchone()
            current: dict[str, Any] = {}
            if row and row[0] is not None:
                try:
                    raw = (row[0] if isinstance(row[0], dict)
                           else json.loads(row[0]))
                except (TypeError, ValueError):
                    raw = {}
                if isinstance(raw, dict):
                    current = raw
            if metadata_mode == "merge" or metadata_value is None:
                # Key removal without a new object edits the stored object;
                # starting from {} would silently wipe every other key.
                merged = dict(current)
                merged.update(metadata_value or {})
            else:
                merged = dict(metadata_value or {})
            for key in clear_keys:
                merged.pop(key, None)
            sets.append("metadata = %s::jsonb")
            params.append(json.dumps(merged))
        if sets and "updated_at = NOW()" not in sets:
            # A metadata-only edit must age like every other update.
            sets.append("updated_at = NOW()")
        if not sets:
            conn.rollback()
            return fail(STATUS_ERROR,
                        "nothing to update: pass content, importance, metadata, "
                        "clear_metadata_keys, entity, mem_type, "
                        "expires_at/clear_expires_at, touch=true or "
                        "restore=true",
                        id=memory_id)
        scope = _scope_of_table(cur, active_table)
        # `entity=None` means "keep the current entity", not "untagged": the
        # near-duplicate warning needs the entity the updated row will have.
        near_entity = entity
        if near_entity is None:
            cur.execute(f"SELECT entity FROM {active_table} WHERE id = %s",
                        (memory_id,))
            row = cur.fetchone()
            near_entity = row[0] if row else None
        near_dup = None
        if content is not None and dedupe != "off":
            # An edit must not fork a fact either: `remember` already refuses
            # duplicates and warns on near-duplicates, and an update that
            # silently creates a second active copy is the same problem.
            dedupe_project = None if scope == "global" else scope
            duplicate = _deduplicate(
                cur, content, dedupe_project, emb, fuzzy_threshold,
                dedupe=dedupe, exclude_id=memory_id)
            if duplicate:
                dup_id, dup_mode, dup_similarity = duplicate
                conn.rollback()
                detail = ("exact content hash" if dup_mode == "exact"
                          else f"semantic match (similarity {dup_similarity})")
                return ok(
                    # Nothing was changed, so `id` names the existing memory
                    # that blocked the edit (same contract as remember). The
                    # id the caller tried to edit is `memory_id`.
                    STATUS_DUPLICATE, id=dup_id, memory_id=memory_id,
                    duplicate_of=dup_id,
                    duplicate_mode=dup_mode, similarity=dup_similarity,
                    project=scope,
                    warnings=[f"update not applied: the new content would "
                              f"duplicate active memory {dup_id} ({detail}); "
                              f"nothing was changed"])
            near_dup = _find_near_duplicate(
                cur, content, dedupe_project, emb, exclude_id=memory_id)
        near_conflict = bool(
            near_dup
            and _content_conflict(content or "", near_dup.get("content") or ""))
        cur.execute(
            f"UPDATE {active_table} SET {', '.join(sets)} "
            "WHERE id = %s AND superseded_by IS NULL AND deleted_at IS NULL",
            params + [memory_id],
        )
        if not cur.rowcount:
            conn.rollback()
            return fail(STATUS_NOT_FOUND, f"NOT FOUND: {memory_id}",
                        id=memory_id)
        conn.commit()
    except Exception as e:
        conn.rollback()
        return fail(STATUS_ERROR, str(e), id=memory_id)
    finally:
        conn.close()
    warnings = ([_near_duplicate_warning(near_dup, near_entity,
                                         near_conflict, update=True)]
                if near_dup else [])
    ttl_warning = _ttl_past_warning(expires_at) if expires_at is not None else None
    if ttl_warning:
        warnings.append(ttl_warning)
    return ok(STATUS_UPDATED, id=memory_id, project=scope, warnings=warnings)


def update_batch(
    updates: Annotated[
        list[BatchUpdateInput],
        Field(description="Updates to apply. Each entry needs 'memory_id'; "
                          "optional 'content', 'importance', 'metadata', "
                          "'entity', 'mem_type', 'expires_at', "
                          "'clear_expires_at', 'touch', 'restore' and "
                          "'project'. Invalid entries are reported "
                          "individually and the rest is still applied.")],
    project: Annotated[
        str | None,
        Field(description="Default project scope for all entries (per-entry "
                          "project wins). Without it every id is looked up in "
                          "every scope.")] = None,
    dedupe: Annotated[
        Literal["exact", "semantic", "off"],
        Field(description="Default duplicate check for content changes "
                          "(per-entry 'dedupe' wins). 'exact' refuses an edit "
                          "that would duplicate another active memory, "
                          "'semantic' also checks vector similarity, 'off' "
                          "skips the check.")] = "exact",
    fuzzy_threshold: Annotated[
        float | None,
        Field(description="Similarity threshold 0.0-1.0 for semantic dedupe "
                          "(default 0.85).")] = None,
) -> dict[str, Any]:
    """Update several memories at once (savepoints per entry).

    Each entry works like one `update_memory` call: `content` changes
    re-embed the memory, `importance`/`metadata`/`entity`/`mem_type` and the
    TTL (`expires_at`/`clear_expires_at`) are applied as given. `touch=true`
    refreshes the decay age without changing the content and can be combined
    with the other fields. `restore=true`
    undoes a soft delete and is mutually exclusive with the other fields.
    An unknown id answers `not_found` in its entry, an id
    that exists in several scopes without a `project` answers `error` with
    the `scopes` - one bad entry never aborts the rest (unlike
    `relate_batch`, which is all-or-nothing).
    Returns `count` + `results`; `status` is "partial" when some entries
    failed.
    """
    logger.info("update_batch count=%s project=%s",
                len(updates) if isinstance(updates, list) else -1, project)
    try:
        default_project = _norm_project(project)
    except ValueError as e:
        return fail(STATUS_ERROR, str(e))
    if not isinstance(updates, list) or not updates:
        return fail(STATUS_ERROR, "updates must be a non-empty list")
    if dedupe not in DEDUPE_MODES:
        return fail(STATUS_ERROR,
                    f"Invalid dedupe. Choose from: "
                    f"{', '.join(sorted(DEDUPE_MODES))}")
    if (fuzzy_threshold is not None
            and not 0.0 <= fuzzy_threshold <= 1.0):
        return fail(STATUS_ERROR,
                    "fuzzy_threshold must be between 0.0 and 1.0")

    updates = [u.model_dump(exclude_unset=True) if isinstance(u, BatchUpdate)
               else u for u in updates]

    # Validate everything before embedding, so one malformed entry cannot
    # abort the batch (the same contract as remember_batch).
    prepared: list[tuple[dict | None, str | None]] = []
    for entry in updates:
        if not isinstance(entry, dict) or entry.get("_not_an_object"):
            prepared.append((None, "update must be an object"))
            continue
        memory_id = entry.get("memory_id")
        memory_id = memory_id.strip() if isinstance(memory_id, str) else ""
        if not memory_id:
            prepared.append((None, "memory_id is required"))
            continue
        try:
            entry_project = _norm_project(entry.get("project", default_project))
        except ValueError as e:
            prepared.append((None, str(e)))
            continue
        fields: dict[str, Any] = {}
        error: str | None = None
        content = entry.get("content")
        if content is not None:
            content = content.strip() if isinstance(content, str) else ""
            if not content:
                error = "content must not be empty"
            else:
                fields["content"] = content
        importance = entry.get("importance")
        if error is None and importance is not None:
            try:
                importance = float(importance)
            except (TypeError, ValueError):
                error = "importance must be a number between 0.0 and 1.0"
            else:
                if not 0.0 <= importance <= 1.0:
                    error = "importance must be between 0.0 and 1.0"
                else:
                    fields["importance"] = importance
        metadata = entry.get("metadata")
        if error is None and metadata is not None:
            metadata, meta_error = _coerce_metadata(metadata)
            if meta_error:
                error = meta_error
            else:
                fields["metadata"] = metadata
        entry_metadata_mode = entry.get("metadata_mode") or "replace"
        if error is None and entry_metadata_mode not in ("replace", "merge"):
            error = "metadata_mode must be 'replace' or 'merge'"
        entry_clear_keys = entry.get("clear_metadata_keys")
        if error is None and entry_clear_keys is not None:
            if not isinstance(entry_clear_keys, list) or any(
                    not isinstance(key, str) or not key.strip()
                    for key in entry_clear_keys):
                error = ("clear_metadata_keys must be a list of non-empty "
                         "strings")
            else:
                entry_clear_keys = list(dict.fromkeys(
                    key.strip() for key in entry_clear_keys))
        entity = entry.get("entity")
        if error is None and entity is not None:
            try:
                fields["entity"] = _norm_entity(entity)
            except ValueError as exc:
                error = str(exc)
        mem_type = entry.get("mem_type")
        if error is None and mem_type is not None:
            if mem_type not in MEMORY_TYPES:
                error = (f"Invalid mem_type. Choose from: "
                         f"{', '.join(sorted(MEMORY_TYPES))}")
            else:
                fields["mem_type"] = mem_type
        entry_dedupe = entry.get("dedupe") or dedupe
        if error is None and entry_dedupe not in DEDUPE_MODES:
            error = (f"Invalid dedupe. Choose from: "
                     f"{', '.join(sorted(DEDUPE_MODES))}")
        entry_fuzzy = entry.get("fuzzy_threshold")
        if entry_fuzzy is None:
            entry_fuzzy = fuzzy_threshold
        if error is None and entry_fuzzy is not None:
            try:
                entry_fuzzy = float(entry_fuzzy)
            except (TypeError, ValueError):
                error = "fuzzy_threshold must be a number between 0.0 and 1.0"
            else:
                if not 0.0 <= entry_fuzzy <= 1.0:
                    error = "fuzzy_threshold must be between 0.0 and 1.0"
        entry_expires = entry.get("expires_at")
        entry_clear = bool(entry.get("clear_expires_at"))
        if error is None and entry_expires is not None and entry_clear:
            error = "pass either expires_at or clear_expires_at, not both"
        elif error is None and entry_expires is not None:
            try:
                fields["expires_at"] = datetime.fromisoformat(
                    str(entry_expires).replace("Z", "+00:00"))
            except ValueError:
                error = "expires_at must be an ISO 8601 timestamp"
        elif error is None and entry_clear:
            fields["expires_at"] = None
        entry_ttl_warning = (_ttl_past_warning(entry_expires)
                             if entry_expires is not None and error is None
                             else None)
        entry_restore = bool(entry.get("restore"))
        entry_touch = bool(entry.get("touch"))
        if error is None and entry_restore and (
                fields or entry_touch or entry_clear_keys):
            error = ("restore=true cannot be combined with other update "
                     "fields")
        if error is None and not fields and not entry_restore \
                and not entry_touch and not entry_clear_keys:
            error = ("nothing to update: pass content, importance, metadata, "
                     "clear_metadata_keys, entity, mem_type, "
                     "expires_at/clear_expires_at, touch=true or restore=true")
        if error:
            prepared.append((None, error))
        else:
            prepared.append(({"memory_id": memory_id,
                              "project": entry_project,
                              "fields": fields,
                              "restore": entry_restore,
                              "dedupe": entry_dedupe,
                              "fuzzy_threshold": entry_fuzzy,
                              "metadata_mode": entry_metadata_mode,
                              "clear_metadata_keys": entry_clear_keys or [],
                              "ttl_warning": entry_ttl_warning}, None))

    valid = [(i, spec) for i, (spec, err) in enumerate(prepared)
             if spec is not None]
    with_content = [(i, spec["fields"]["content"]) for i, spec in valid
                    if "content" in spec["fields"]]
    embeddings: dict[int, list[float]] = {}
    if with_content:
        try:
            vectors = embed([content for _, content in with_content])
        except Exception as e:
            logger.exception("update_batch embedding failed")
            return fail(STATUS_ERROR, f"embedding failed: {e}")
        embeddings = {i: vec for (i, _), vec in zip(with_content, vectors)}

    conn = get_conn()
    results: list[dict] = []
    try:
        cur = conn.cursor()
        for i, (spec, err) in enumerate(prepared):
            if spec is None:
                raw = updates[i]
                raw_id = raw.get("memory_id") if isinstance(raw, dict) else None
                results.append({"id": raw_id, "status": "error", "error": err})
                continue
            memory_id = spec["memory_id"]
            entry_project = spec["project"]
            savepoint = f"update_{i}"
            cur.execute(f"SAVEPOINT {savepoint}")
            try:
                if spec.get("restore"):
                    if entry_project:
                        restore_tables = [_mem_table(entry_project)]
                        if not _table_exists(cur, restore_tables[0]):
                            cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                            results.append({
                                "id": memory_id, "status": "not_found",
                                "error": f"NOT FOUND: {memory_id}"})
                            continue
                    else:
                        scopes = _memory_scopes(cur, memory_id)
                        if len(scopes) > 1:
                            cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                            results.append({
                                "id": memory_id, "status": "error",
                                "error": (f"ambiguous memory id {memory_id}: "
                                          f"it exists in the scopes {scopes}; "
                                          f"pass project=<name> to say which "
                                          f"one to restore"),
                                "scopes": scopes})
                            continue
                        restore_tables = _all_mem_tables(cur)
                    restored = 0
                    restored_table = None
                    for table in restore_tables:
                        cur.execute(
                            f"UPDATE {table} SET deleted_at = NULL, "
                            f"superseded_by = NULLIF(superseded_by, %s), "
                            f"updated_at = NOW() WHERE id = %s "
                            f"AND deleted_at IS NOT NULL",
                            (SENTINEL_FORGOTTEN, memory_id))
                        if cur.rowcount:
                            restored = cur.rowcount
                            restored_table = table
                            break
                    if not restored:
                        blocked = _restore_blocked_reason(
                            cur, memory_id, restore_tables)
                        cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                        results.append({
                            "id": memory_id,
                            "status": blocked[0] if blocked else "not_found",
                            "error": (blocked[1] if blocked
                                      else f"NOT FOUND: {memory_id}")})
                        continue
                    scope = _scope_of_table(cur, restored_table)
                    cur.execute(f"RELEASE SAVEPOINT {savepoint}")
                    results.append({"id": memory_id,
                                    "status": STATUS_RESTORED,
                                    "project": scope})
                    continue
                sets: list[str] = []
                params: list[Any] = []
                for name, value in spec["fields"].items():
                    if name == "content":
                        sets += ["content = %s", "content_hash = %s",
                                 "embedding = %s::vector"]
                        params += [value, _content_hash(value), embeddings[i]]
                    elif name == "metadata":
                        if spec["metadata_mode"] == "merge" \
                                or spec["clear_metadata_keys"]:
                            # Computed against the stored object below.
                            continue
                        sets.append("metadata = %s::jsonb")
                        params.append(json.dumps(value))
                    else:
                        sets.append(f"{name} = %s")
                        params.append(value)
                sets.append("updated_at = NOW()")

                if entry_project:
                    tables = [_mem_table(entry_project), "memories"]
                else:
                    scopes = _memory_scopes(cur, memory_id)
                    if len(scopes) > 1:
                        cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                        results.append({
                            "id": memory_id, "status": "error",
                            "error": (f"ambiguous memory id {memory_id}: it "
                                      f"exists in the scopes {scopes}; pass "
                                      f"project=<name> to say which one to "
                                      f"update"),
                            "scopes": scopes})
                        continue
                    tables = _all_mem_tables(cur)

                active_table = None
                for table in tables:
                    cur.execute(
                        f"SELECT 1 FROM {table} WHERE id = %s "
                        "AND superseded_by IS NULL AND deleted_at IS NULL",
                        (memory_id,))
                    if cur.fetchone():
                        active_table = table
                        break
                if active_table is None:
                    reason = _hidden_state_reason(cur, memory_id, tables)
                    cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                    error = (f"memory {memory_id} exists but cannot be "
                             f"updated: {reason}" if reason
                             else f"NOT FOUND: {memory_id}")
                    results.append({"id": memory_id, "status": "not_found",
                                    "error": error})
                    continue
                scope = _scope_of_table(cur, active_table)
                if spec["metadata_mode"] == "merge" \
                        or spec["clear_metadata_keys"]:
                    cur.execute(
                        f"SELECT metadata FROM {active_table} WHERE id = %s",
                        (memory_id,))
                    row = cur.fetchone()
                    current: dict[str, Any] = {}
                    if row and row[0] is not None:
                        try:
                            raw = (row[0] if isinstance(row[0], dict)
                                   else json.loads(row[0]))
                        except (TypeError, ValueError):
                            raw = {}
                        if isinstance(raw, dict):
                            current = raw
                    new_meta = spec["fields"].get("metadata")
                    if spec["metadata_mode"] == "merge" or new_meta is None:
                        merged = dict(current)
                        merged.update(new_meta or {})
                    else:
                        merged = dict(new_meta or {})
                    for key in spec["clear_metadata_keys"]:
                        merged.pop(key, None)
                    sets.append("metadata = %s::jsonb")
                    params.append(json.dumps(merged))
                near_entity = spec["fields"].get("entity")
                if near_entity is None:
                    cur.execute(
                        f"SELECT entity FROM {active_table} WHERE id = %s",
                        (memory_id,))
                    row = cur.fetchone()
                    near_entity = row[0] if row else None
                near_dup = None
                if ("content" in spec["fields"]
                        and spec.get("dedupe") != "off"):
                    dedupe_project = None if scope == "global" else scope
                    duplicate = _deduplicate(
                        cur, spec["fields"]["content"], dedupe_project,
                        embeddings[i], spec.get("fuzzy_threshold"),
                        dedupe=spec.get("dedupe", "exact"),
                        exclude_id=memory_id)
                    if duplicate:
                        dup_id, dup_mode, dup_similarity = duplicate
                        detail = ("exact content hash" if dup_mode == "exact"
                                  else f"semantic match (similarity "
                                       f"{dup_similarity})")
                        cur.execute(f"RELEASE SAVEPOINT {savepoint}")
                        results.append({
                            "id": dup_id, "memory_id": memory_id,
                            "status": "duplicate",
                            "duplicate_of": dup_id,
                            "duplicate_mode": dup_mode,
                            "similarity": dup_similarity,
                            "project": scope,
                            "warnings": [
                                f"update not applied: the new content would "
                                f"duplicate active memory {dup_id} ({detail}); "
                                f"nothing was changed"]})
                        continue
                    near_dup = _find_near_duplicate(
                        cur, spec["fields"]["content"], dedupe_project,
                        embeddings[i], exclude_id=memory_id)
                near_conflict = bool(
                    near_dup
                    and _content_conflict(
                        spec["fields"].get("content") or "",
                        near_dup.get("content") or ""))
                cur.execute(
                    f"UPDATE {active_table} SET {', '.join(sets)} "
                    "WHERE id = %s AND superseded_by IS NULL "
                    "AND deleted_at IS NULL",
                    params + [memory_id])
                if not cur.rowcount:
                    cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                    results.append({"id": memory_id, "status": "not_found",
                                    "error": f"NOT FOUND: {memory_id}"})
                    continue
                cur.execute(f"RELEASE SAVEPOINT {savepoint}")
            except Exception as exc:
                cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                results.append({"id": memory_id, "status": "error",
                                "error": str(exc)})
                continue
            entry: dict[str, Any] = {"id": memory_id,
                                     "status": STATUS_UPDATED,
                                     "project": scope}
            if near_dup:
                entry["warnings"] = [
                    _near_duplicate_warning(near_dup, near_entity,
                                            near_conflict, update=True)]
            if spec.get("ttl_warning"):
                entry.setdefault("warnings", []).append(spec["ttl_warning"])
            results.append(entry)
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.exception("update_batch failed")
        return fail(STATUS_ERROR, str(e))
    finally:
        conn.close()
    failed = [r for r in results
              if r.get("status") not in (STATUS_UPDATED, STATUS_RESTORED)]
    succeeded = len(results) - len(failed)
    if failed and len(failed) == len(results):
        return fail(STATUS_ERROR, "no memory could be updated",
                    count=len(results), succeeded=0, failed=len(failed),
                    results=results)
    status = STATUS_PARTIAL if failed else STATUS_OK
    warnings = [f"{len(failed)} update(s) failed"] if failed else []
    return ok(status, count=len(results), succeeded=succeeded,
              failed=len(failed), results=results, warnings=warnings)



def forget(
    memory_id: Annotated[
        str | None,
        Field(description="Id of a single memory to remove.")] = None,
    memory_ids: Annotated[
        list[str] | None,
        Field(description="List of memory ids to remove in one call.")] = None,
    entity_name: Annotated[
        str | None,
        Field(description="Entity whose memories (and graph node) are removed. "
                          "Alias: `entity`. Use '*' to remove every memory of "
                          "the given project.")] = None,
    project: Annotated[
        str | None,
        Field(description="Scope to act in. Without it the id/entity is looked "
                          "up in every scope.")] = None,
    soft: Annotated[
        bool,
        Field(description="True (default) hides the memory; False deletes it "
                          "permanently.")] = True,
    entity: Annotated[
        str | None,
        Field(description="Entity whose memories (and graph node) are removed. "
                          "Alias: `entity_name`.")] = None,
    dry_run: Annotated[
        bool,
        Field(description="True only reports what would be removed; nothing is "
                          "deleted or hidden.")] = False,
) -> dict[str, Any]:
    """Remove a fact by memory_id(s) or delete an entity and all its memories.

    The entity can be passed as `entity` or `entity_name`.
    When soft=True (default), hides the memory by setting deleted_at and a
    forgotten sentinel; it remains available only for database-level recovery.
    Without `project`, the id/entity is looked up in every table; deleting an
    entity also removes its memories from all scopes and its graph node.

    A `memory_id` is unique per scope, not globally: if the same id exists in
    several scopes (an import copies the id), the call is refused and the
    scopes are named - pass `project` to say which one to remove.

    `memory_ids` accepts a list of ids to remove in one call. Bulk operations
    are more efficient than multiple single calls. Each id is validated
    individually and the response reports which ids were found and which were
    not. `memory_id` and `memory_ids` are mutually exclusive.

    `entity="*"` removes every memory of the given project (memories and
    graph nodes); `project` is required for that, so a stray call cannot wipe
    the whole database. `dry_run=True` reports the counts without changing
    anything.
    """
    entity_name = entity or entity_name
    logger.info("forget memory_id=%s memory_ids=%s entity_name=%s soft=%s dry_run=%s",
                memory_id, memory_ids, entity_name, soft, dry_run)
    raw_project = project
    try:
        project = _norm_project(project)
    except ValueError as e:
        return fail(STATUS_ERROR, str(e))
    if memory_id and memory_ids:
        return fail(STATUS_ERROR, "memory_id and memory_ids are mutually exclusive")
    if not memory_id and not memory_ids and not entity_name:
        return fail(STATUS_ERROR, "memory_id, memory_ids or entity is required")
    scope_name = project if project else "global"
    purge_all = entity_name == "*"
    # An explicit `project` names the scope to purge — including the reserved
    # name "global", which _norm_project maps to None (the shared table).
    # Only a call without any scope is refused, so a stray call cannot wipe
    # the whole database.
    explicit_scope = (raw_project is not None
                      and str(raw_project).strip() != "")
    if purge_all and not explicit_scope:
        return fail(STATUS_ERROR,
                    "entity='*' removes a whole scope: pass project=<name> "
                    "(the shared scope is addressed with project=\"global\")")
    conn = get_conn()
    counts = {"memories": 0, "entities": 0, "relations": 0}
    scopes: list[str] = []
    affected_scopes: list[str] = []
    bulk_results: list[dict] = []
    try:
        cur = conn.cursor()
        if memory_ids:
            # Bulk forget: process multiple memory ids in one call
            tables = _all_mem_tables(cur) if not project else [_mem_table(project)]
            if project and not _table_exists(cur, tables[0]):
                return fail(STATUS_NOT_FOUND, f"NOT FOUND: project table does not exist")
            
            for mid in memory_ids:
                if not mid or not isinstance(mid, str):
                    bulk_results.append({"id": mid, "status": "error", "error": "invalid memory id"})
                    continue
                found = False
                for table in tables:
                    if soft:
                        if dry_run:
                            cur.execute(
                                f"SELECT COUNT(*) FROM {table} "
                                f"WHERE id = %s AND deleted_at IS NULL",
                                (mid,),
                            )
                            count = cur.fetchone()[0]
                            if count:
                                found = True
                                counts["memories"] += count
                        else:
                            cur.execute(
                                f"UPDATE {table} SET superseded_by = COALESCE(superseded_by, %s), "
                                f"deleted_at = NOW() WHERE id = %s AND deleted_at IS NULL",
                                (SENTINEL_FORGOTTEN, mid),
                            )
                            if cur.rowcount:
                                found = True
                                counts["memories"] += cur.rowcount
                    else:
                        if dry_run:
                            cur.execute(
                                f"SELECT COUNT(*) FROM {table} WHERE id = %s",
                                (mid,),
                            )
                            count = cur.fetchone()[0]
                            if count:
                                found = True
                                counts["memories"] += count
                        else:
                            cur.execute(f"DELETE FROM {table} WHERE id = %s", (mid,))
                            if cur.rowcount:
                                found = True
                                counts["memories"] += cur.rowcount
                if found:
                    bulk_results.append({"id": mid, "status": "deleted" if not soft else "soft_forgotten"})
                else:
                    bulk_results.append({"id": mid, "status": "not_found"})
            
            if not dry_run:
                if not soft:
                    clear_dangling_supersede(cur)
                conn.commit()
            else:
                conn.rollback()
            
            status = STATUS_OK if dry_run else (STATUS_SOFT_FORGOTTEN if soft else STATUS_DELETED)
            return ok(status, counts=counts, dry_run=dry_run, bulk_results=bulk_results,
                      message=f"{'Would remove' if dry_run else ('Soft-forgotten' if soft else 'Deleted')} "
                              f"{counts['memories']} memory/memories from {len(memory_ids)} id(s)")
        if memory_id:
            tables = _all_mem_tables(cur) if not project else [_mem_table(project)]
            if project and not _table_exists(cur, tables[0]):
                return fail(STATUS_NOT_FOUND, f"NOT FOUND: {memory_id}", id=memory_id)
            scopes = _memory_scopes(cur, memory_id)
            if not project and len(scopes) > 1:
                # The same id can live in several scopes (an import copies
                # the id). Removing it from all of them at once would
                # delete more than the caller asked for.
                conn.rollback()
                return fail(STATUS_ERROR,
                            f"ambiguous memory id {memory_id}: it exists in "
                            f"the scopes {scopes}; pass project=<name> to say "
                            f"which one to remove",
                            id=memory_id, scopes=scopes)
            for table in tables:
                if soft:
                    if dry_run:
                        cur.execute(
                            f"SELECT COUNT(*) FROM {table} "
                            f"WHERE id = %s AND deleted_at IS NULL",
                            (memory_id,),
                        )
                        counts["memories"] += cur.fetchone()[0]
                    else:
                        cur.execute(
                            f"UPDATE {table} SET superseded_by = COALESCE(superseded_by, %s), "
                            f"deleted_at = NOW() WHERE id = %s AND deleted_at IS NULL",
                            (SENTINEL_FORGOTTEN, memory_id),
                        )
                        counts["memories"] += cur.rowcount
                else:
                    if dry_run:
                        cur.execute(
                            f"SELECT COUNT(*) FROM {table} WHERE id = %s",
                            (memory_id,),
                        )
                        counts["memories"] += cur.fetchone()[0]
                    else:
                        cur.execute(f"DELETE FROM {table} WHERE id = %s", (memory_id,))
                        counts["memories"] += cur.rowcount
            if not counts["memories"]:
                conn.rollback()
                return fail(STATUS_NOT_FOUND, f"NOT FOUND: {memory_id}", id=memory_id)
            if not soft and not dry_run:
                clear_dangling_supersede(cur)
        if entity_name:
            cur.execute("SET search_path TO ag_catalog, public")
            en_json = json.dumps(entity_name)
            scope = _scope_key(project)
            if purge_all:
                # A whole scope goes: its memories, its graph nodes and the
                # entity/relation mirror rows of that scope.
                label = _scope_label(project)
                table = _mem_table(project)
                if _table_exists(cur, table):
                    if dry_run:
                        # The preview must count what the real call removes:
                        # a soft forget hides the active rows, a hard delete
                        # takes the whole table.
                        where = ("WHERE deleted_at IS NULL" if soft else "")
                        cur.execute(f"SELECT COUNT(*) FROM {table} {where}")
                        counts["memories"] += cur.fetchone()[0]
                    elif soft:
                        cur.execute(f"UPDATE {table} SET "
                                    f"superseded_by = COALESCE(superseded_by, %s), "
                                    f"deleted_at = NOW() WHERE deleted_at IS NULL",
                                    (SENTINEL_FORGOTTEN,))
                        counts["memories"] += cur.rowcount
                    else:
                        cur.execute(f"DELETE FROM {table}")
                        counts["memories"] += cur.rowcount
                cur.execute("SELECT COUNT(*) FROM entities WHERE project = %s",
                            (scope,))
                counts["entities"] = cur.fetchone()[0]
                cur.execute(
                    "SELECT COUNT(*) FROM relations WHERE source_id IN "
                    "(SELECT id FROM entities WHERE project = %s) OR target_id IN "
                    "(SELECT id FROM entities WHERE project = %s)",
                    (scope, scope),
                )
                counts["relations"] = cur.fetchone()[0]
                if not dry_run:
                    cur.execute(
                        f"SELECT * FROM cypher('memory_graph', $$ "
                        f"MATCH (n:{label}) DETACH DELETE n $$) AS (r agtype)",
                    )
                    cur.execute("DELETE FROM entities WHERE project = %s", (scope,))
                if dry_run:
                    conn.rollback()
                else:
                    if not soft:
                        clear_dangling_supersede(cur)
                    conn.commit()
                if not (counts["memories"] or counts["entities"]
                        or counts["relations"]):
                    conn.rollback()
                    return fail(STATUS_NOT_FOUND,
                                f"no memories in scope {scope_name}")
                status = STATUS_OK if dry_run else (
                    STATUS_SOFT_FORGOTTEN if soft else STATUS_DELETED)
                return ok(status, id=None, entity="*", project=scope_name,
                          counts=counts, dry_run=dry_run,
                          message=(
                              f"{'Would remove' if dry_run else 'Soft-forgotten' if soft else 'Deleted'} "
                              f"all memories of {scope_name}"
                              + (" (rows stay recoverable with "
                                 "update_memory(restore=true); use "
                                 "drop_project to remove the scope "
                                 "completely)" if soft and not dry_run else "")
                          ))
            # The scope filter must match the delete path exactly: without a
            # project the entity is removed from *every* scope, so the preview
            # has to count every scope as well. Otherwise `dry_run` reports
            # "NOT FOUND" for a graph node that the real call deletes.
            scope_clause = " AND project = %s" if project else ""
            scope_args: tuple[Any, ...] = ((entity_name, scope) if project
                                            else (entity_name,))
            cur.execute(
                f"SELECT id, project FROM entities WHERE name = %s{scope_clause}",
                scope_args,
            )
            entity_rows = cur.fetchall()
            entity_ids = [r[0] for r in entity_rows]
            counts["entities"] = len(entity_ids)
            affected_scopes: list[str] = []
            for row in entity_rows:
                scope_hit = row[1] or "global"
                if scope_hit not in affected_scopes:
                    affected_scopes.append(scope_hit)
            if entity_ids:
                cur.execute(
                    "SELECT COUNT(*) FROM relations "
                    "WHERE source_id = ANY(%s) OR target_id = ANY(%s)",
                    (entity_ids, entity_ids),
                )
                relation_count = cur.fetchone()
                counts["relations"] = relation_count[0] if relation_count else 0
            if project:
                label = _scope_label(project)
                if not dry_run:
                    cur.execute(
                        f"SELECT * FROM cypher('memory_graph', $$ "
                        f"MATCH (n:{label} {{name: {en_json}}}) DETACH DELETE n $$) AS (r agtype)",
                    )
                    cur.execute(
                        "DELETE FROM entities WHERE name = %s AND project = %s",
                        (entity_name, scope),
                    )
            else:
                if not dry_run:
                    cur.execute(
                        f"SELECT * FROM cypher('memory_graph', $$ "
                        f"MATCH (n {{name: {en_json}}}) DETACH DELETE n $$) AS (r agtype)",
                    )
                    cur.execute("DELETE FROM entities WHERE name = %s", (entity_name,))

            # Only the selected scope's memories: forgetting an entity in one
            # project must not wipe the global facts of the same entity.
            tables = (_all_mem_tables(cur) if not project
                      else [_mem_table(project)])
            for tbl in dict.fromkeys(tables):
                if not _table_exists(cur, tbl):
                    continue
                memories_before = counts["memories"]
                if dry_run:
                    # A soft forget hides the active rows, a hard delete also
                    # takes the superseded and forgotten ones: the preview
                    # counts exactly what the real call removes.
                    where = (" AND deleted_at IS NULL" if soft else "")
                    cur.execute(f"SELECT COUNT(*) FROM {tbl} "
                                f"WHERE entity = %s{where}",
                                (entity_name,))
                    counts["memories"] += cur.fetchone()[0]
                elif soft:
                    cur.execute(f"UPDATE {tbl} SET "
                                f"superseded_by = COALESCE(superseded_by, %s), "
                                f"deleted_at = NOW() "
                                f"WHERE entity = %s AND deleted_at IS NULL",
                                (SENTINEL_FORGOTTEN, entity_name))
                    counts["memories"] += cur.rowcount
                else:
                    cur.execute(f"DELETE FROM {tbl} WHERE entity = %s", (entity_name,))
                    counts["memories"] += cur.rowcount
                relation_context_filter = (
                    "metadata->>'relation_context' = 'true' "
                    "AND (metadata->>'from' = %s OR metadata->>'to' = %s)"
                )
                if dry_run:
                    where = (" AND deleted_at IS NULL" if soft else "")
                    cur.execute(f"SELECT COUNT(*) FROM {tbl} "
                                f"WHERE {relation_context_filter}{where}",
                                (entity_name, entity_name))
                    counts["memories"] += cur.fetchone()[0]

                elif soft:
                    cur.execute(
                        f"UPDATE {tbl} SET superseded_by = COALESCE(superseded_by, %s), "
                        f"deleted_at = NOW() "
                        f"WHERE deleted_at IS NULL AND {relation_context_filter}",
                        (SENTINEL_FORGOTTEN, entity_name, entity_name),
                    )
                    counts["memories"] += cur.rowcount
                else:
                    cur.execute(
                        f"DELETE FROM {tbl} WHERE {relation_context_filter}",
                        (entity_name, entity_name),
                    )
                    counts["memories"] += cur.rowcount
                if counts["memories"] != memories_before:
                    # Only scopes that actually lost (or would lose) rows are
                    # affected. Reporting every searched scope made a plain
                    # "touched multiple scopes" warning fire for a
                    # single-scope entity whenever an unrelated project
                    # table existed.
                    scope_hit = _scope_of_table(cur, tbl)
                    if scope_hit not in affected_scopes:
                        affected_scopes.append(scope_hit)
            if not (counts["memories"] or counts["entities"] or counts["relations"]):
                # An entity whose memories are all soft-forgotten already has
                # nothing active to hide, but it still exists. Answering
                # "NOT FOUND" would send the caller looking for a typo.
                hidden = 0
                if soft:
                    for tbl in dict.fromkeys(tables):
                        if not _table_exists(cur, tbl):
                            continue
                        cur.execute(
                            f"SELECT COUNT(*) FROM {tbl} WHERE entity = %s "
                            f"OR (metadata->>'relation_context' = 'true' AND "
                            f"(metadata->>'from' = %s OR metadata->>'to' = %s))",
                            (entity_name, entity_name, entity_name))
                        hidden += cur.fetchone()[0]
                if hidden:
                    conn.rollback()
                    return ok(
                        STATUS_OK, id=None, entity=entity_name,
                        project=scope_name, counts=counts, dry_run=dry_run,
                        message=(f"nothing to forget: all {hidden} "
                                 f"memory/memories of {entity_name!r} are "
                                 f"already soft-forgotten"))
                conn.rollback()
                return fail(STATUS_NOT_FOUND, f"NOT FOUND: {entity_name}")
        if dry_run:
            conn.rollback()
        else:
            if entity_name and not soft:
                clear_dangling_supersede(cur)
            conn.commit()
    except Exception as e:
        conn.rollback()
        return fail(STATUS_ERROR, str(e))
    finally:
        conn.close()
    status = STATUS_SOFT_FORGOTTEN if soft else STATUS_DELETED
    target = memory_id or entity_name or f"{len(memory_ids) if memory_ids else 0} ids"
    extra: dict[str, Any] = {}
    if memory_id and scopes:
        extra["scopes"] = scopes
    if memory_ids:
        extra["bulk_results"] = bulk_results
    warnings: list[str] = []
    if entity_name and not project:
        extra["scopes_affected"] = affected_scopes
        if len(affected_scopes) > 1:
            warnings.append(
                f"forget(entity=...) without project=true touched multiple "
                f"scopes ({', '.join(affected_scopes)}); pass project=<name> "
                f"to restrict the removal to one scope")
    if dry_run:
        return ok(STATUS_OK, id=memory_id, entity=entity_name, counts=counts,
                  dry_run=True, **extra, warnings=warnings,
                  message=f"Would remove {target} (nothing was changed)")
    return ok(status, id=memory_id, entity=entity_name, counts=counts,
              dry_run=False, **extra, warnings=warnings,
              message=f"{'Soft-forgotten' if soft else 'Deleted'} {target}")
