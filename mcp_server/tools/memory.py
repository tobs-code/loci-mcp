import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

from mcp_server.browse import escape_like
from mcp_server.config import (
    CONTRADICTION_AUTO_MAX_SUPERSEDES,
    CONTRADICTION_AUTO_MIN_CONFIDENCE,
    DEDUPE_MODES,
    HARD_DELETE_GRACE_DAYS,
    MEMORY_TYPE_HELP,
    MEMORY_TYPES,
    SENTINEL_FORGOTTEN,
    VALID_SCOPES,
)
from mcp_server.duplicates import (
    content_conflict as _content_conflict,
    same_topic as _shares_subject,  # noqa: F401 - re-exported for tests
)
from mcp_server.envelope import (
    STATUS_CONFLICT,
    STATUS_DELETED,
    STATUS_MERGED,
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
from mcp_server.filters import (
    MAX_TAGS,
    normalize_optional_text,
    normalize_tags,
    parse_iso_timestamp,
    validate_confidence,
)
from mcp_server.graph_store import GraphStore
from mcp_server.inference_gateway import embed
from mcp_server.lifecycle import (
    ACTIVE_ROW_SQL,
)
from mcp_server.lifecycle import (
    is_expired as _is_past,
)
from mcp_server.lifecycle import (
    parse_expires_at,
)
from mcp_server.scope_catalog import (
    _all_mem_tables,
    _mem_table,
    _memory_scopes,
    _norm_entity,
    _norm_project,
    _scope_of_table,
    _table_exists,
)
from mcp_server.scoring import _content_hash, _gen_id
from mcp_server.search import SearchRequest, _extra_conditions, _search_memories
from mcp_server.storage import (
    _deduplicate,
    _find_near_duplicate,
    _store_memory,
    clear_dangling_supersede,
)
from mcp_server.store import MemoryStore
from mcp_server.store_updates import (
    active_table as active_table_for,
    apply_fields,
    count_entity_rows_incl_context,
    forget_entity_rows,
    forget_relation_context_rows,
    lifecycle_row,
    restore_rows,
    stored_entity,
    stored_metadata,
    stored_tags,
)
from mcp_server.write_pipeline import (
    _GRAPH_METADATA_KEYS,
    WriteIntent,
    _coerce_metadata,
    _near_duplicate_warning,
    _ttl_past_warning,
    _validate_links,
    write_one,
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
    mem_type: Any = Field(default=None, json_schema_extra={
        "type": ["string", "null"],
        "enum": ["fact", "decision", "learning", "preference", "pattern",
                 "incident", "runbook", "context", "observation", "convention",
                 "command", "note", None],
    }, description=MEMORY_TYPE_HELP + " Omit it when the fact replaces another "
                                       "memory: the replaced memory's type is "
                                       "carried over.")
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
    contradiction_mode: Any = Field(
        default=None,
        json_schema_extra={"type": ["string", "null"],
                           "enum": ["supersede", "warn", None]},
        description="Overrides the batch contradiction policy for this fact: "
                    "'supersede' replaces the old memory, 'warn' only reports "
                    "it.")
    contradiction_min_confidence: Any = Field(
        default=None,
        json_schema_extra={"type": ["number", "null"]},
        description="Minimum NLI confidence (0.0-1.0) for an automatic "
                    "supersede of this fact; overrides the batch default.")
    max_contradiction_supersedes: Any = Field(
        default=None,
        json_schema_extra={"type": ["integer", "null"]},
        description="Maximum number of memories this fact may invalidate; "
                    "overrides the batch default.")
    expires_at: Any = Field(default=None, json_schema_extra={
        "type": ["string", "null"]},
        description="Optional ISO 8601 timestamp when this memory stops "
                    "being active (TTL).")
    idempotency_key: Any = Field(default=None, json_schema_extra={
        "type": ["string", "null"]},
        description="Client idempotency token (1-128 characters). A second "
                    "fact with the same key in the same scope answers "
                    "`duplicate` naming the first row instead of writing.")
    valid_from: Any = Field(default=None, json_schema_extra={
        "type": ["string", "null"]},
        description="Optional ISO 8601 timestamp when this fact became true "
                    "(default: now).")
    valid_to: Any = Field(default=None, json_schema_extra={
        "type": ["string", "null"]},
        description="Optional ISO 8601 timestamp when this fact stops being "
                    "true (usually set by a later supersede).")
    source: Any = Field(default=None, json_schema_extra={
        "type": ["string", "null"]},
        description="Where the fact came from (max 200 characters).")
    agent_id: Any = Field(default=None, json_schema_extra={
        "type": ["string", "null"]},
        description="Identifier of the writing agent/client (max 200 "
                    "characters).")
    confidence: Any = Field(default=None, json_schema_extra={
        "type": ["number", "null"]},
        description="How sure the source is of this fact (0.0-1.0).")
    tags: Any = Field(default=None, json_schema_extra={
        "type": ["array", "null"], "items": {"type": "string"}},
        description="Free-form tags (trimmed, lower-cased, at most 32).")
    pinned: Any = Field(default=None, json_schema_extra={"type": ["boolean", "null"]},
                        description="Keep this fact in every `recall` answer of "
                                    "its scope: a pinned fact is backfilled into "
                                    "the page even when a relevance or decay "
                                    "floor would drop it.")
    fuzzy_threshold: Any = Field(default=None, json_schema_extra={"type": ["number", "null"]},
                                 description="Similarity threshold 0.0-1.0 for semantic dedup.")
    dedupe: Any = Field(default=None, json_schema_extra={
        "type": ["string", "null"], "enum": ["exact", "semantic", "off", None],
    }, description="Duplicate detection: 'exact' (content hash), 'semantic' (vector similarity) or 'off'.")
    links: Any = Field(default=None, json_schema_extra={
        "type": ["array", "null"],
        "items": {"type": "object",
                   "properties": {"to_entity": {"type": "string"},
                                   "relation_type": {"type": "string"},
                                   "weight": {"type": "number"},
                                   "context": {"type": "string"},
                                   "from_entity": {"type": "string"}},
                   "required": ["to_entity", "relation_type"]},
    }, description="Graph edges created in the same step: each item needs "
                   "'to_entity' and 'relation_type'; 'from_entity' defaults "
                   "to the fact's entity. A malformed edge is a per-fact "
                   "error; an edge runtime failure keeps the fact and warns "
                   "(links failed: ...).")


_BATCH_FACT_FIELDS = frozenset(BatchFact.model_fields)


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


def _as_bool(value: Any, name: str) -> bool:
    """Coerce a batch flag strictly: only real booleans (or absence) pass.

    A batch entry is raw JSON, so ``bool("false")`` would silently turn a
    string into ``True``. A flag that is not a boolean is an argument error
    for that entry instead of an unintended write.
    """
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    raise ValueError(
        f"{name} must be a boolean (true or false), got {type(value).__name__}")


def _norm_idempotency_key(value: Any) -> str | None:
    """Validate a client idempotency token: trimmed, 1-128 characters."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(
            f"idempotency_key must be a string (got {type(value).__name__})")
    key = value.strip()
    if not key:
        raise ValueError("idempotency_key must not be empty")
    if len(key) > 128:
        raise ValueError("idempotency_key must be at most 128 characters")
    return key


def _hidden_state_reason(db: MemoryStore, memory_id: str,
                         tables: list[str]) -> str | None:
    """Why an existing row cannot be updated or touched.

    A plain "NOT FOUND" hides that the id *does* exist but is superseded,
    forgotten or expired - the caller then keeps looking for the wrong
    problem. This returns the actionable reason, or None for an unknown id.
    """
    found = lifecycle_row(db, memory_id, tables)
    if found is None:
        return None
    _, row = found
    superseded_by = row.get("superseded_by")
    deleted_at = row.get("deleted_at")
    expires_at = row.get("expires_at")
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
    valid_from: Any = Field(default=None, json_schema_extra={
        "type": ["string", "null"]},
        description="New ISO 8601 timestamp when the fact became true.")
    valid_to: Any = Field(default=None, json_schema_extra={
        "type": ["string", "null"]},
        description="New ISO 8601 timestamp when the fact stops being true "
                    "(mutually exclusive with clear_valid_to).")
    clear_valid_to: Any = Field(default=None, json_schema_extra={
        "type": ["boolean", "null"]},
        description="Reopen the validity window (valid_to = NULL).")
    source: Any = Field(default=None, json_schema_extra={
        "type": ["string", "null"]},
        description="Where the fact came from (max 200 characters).")
    agent_id: Any = Field(default=None, json_schema_extra={
        "type": ["string", "null"]},
        description="Identifier of the writing agent (max 200 characters).")
    confidence: Any = Field(default=None, json_schema_extra={
        "type": ["number", "null"]},
        description="How sure the source is of this fact (0.0-1.0).")
    tags: Any = Field(default=None, json_schema_extra={
        "type": ["array", "null"], "items": {"type": "string"}},
        description="New tags (trimmed, lower-cased, at most 32).")
    tags_mode: Any = Field(default=None, json_schema_extra={
        "type": ["string", "null"], "enum": ["replace", "merge", None]},
        description="'replace' (default) or 'merge' (add to the stored tags).")
    clear_tags: Any = Field(default=None, json_schema_extra={
        "type": ["boolean", "null"]},
        description="Remove every tag from the memory.")
    pinned: Any = Field(default=None, json_schema_extra={
        "type": ["boolean", "null"]},
        description="Set or clear the always-include flag.")
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
        str | None,
        Field(description=MEMORY_TYPE_HELP + " Omit it when the fact replaces "
                          "another memory: the replaced memory's type is "
                          "carried over, so a decision stays a decision.")] = None,
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
    contradiction_mode: Annotated[
        Literal["supersede", "warn"],
        Field(description="What a confirmed contradiction does: 'supersede' "
                          "(default) replaces the old memory, 'warn' stores "
                          "the fact and reports the pair without "
                          "invalidating anything. Only the strongest "
                          "`max_contradiction_supersedes` pairs at or above "
                          "`contradiction_min_confidence` are replaced.")] = "supersede",
    contradiction_min_confidence: Annotated[
        float,
        Field(description="Minimum NLI confidence (0.0-1.0, default 0.70) for "
                          "an automatic supersede. Weaker pairs are stored as "
                          "`kept` and only reported.",
              ge=0.0, le=1.0)] = CONTRADICTION_AUTO_MIN_CONFIDENCE,
    max_contradiction_supersedes: Annotated[
        int,
        Field(description="Maximum number of contradicting memories this "
                          "write may invalidate (default 1). Further pairs "
                          "stay active and are reported in "
                          "`contradictions.kept`.",
              ge=0)] = CONTRADICTION_AUTO_MAX_SUPERSEDES,
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
    idempotency_key: Annotated[
        str | None,
        Field(description="Optional client idempotency token (1-128 "
                          "characters). A second `remember` with the same key "
                          "in the same scope returns `duplicate`, names the "
                          "first memory and writes nothing (including no "
                          "supersede side effects). Different scopes have "
                          "independent key spaces.")] = None,
    valid_from: Annotated[
        str | None,
        Field(description="Optional ISO 8601 timestamp when this fact became "
                          "true (default: now). Backdate a fact that was true "
                          "earlier so `as_of` searches can find it.")] = None,
    valid_to: Annotated[
        str | None,
        Field(description="Optional ISO 8601 timestamp when this fact stops "
                          "being true. Usually set by a later supersede; omit "
                          "for a fact that is still valid.")] = None,
    source: Annotated[
        str | None,
        Field(description="Where the fact came from (document, URL, person, "
                          "system); trimmed, at most 200 characters.")] = None,
    agent_id: Annotated[
        str | None,
        Field(description="Identifier of the agent/client that wrote the "
                          "fact; trimmed, at most 200 characters.")] = None,
    confidence: Annotated[
        float | None,
        Field(ge=0.0, le=1.0,
              description="How sure the source is of this fact (0.0-1.0).")] = None,
    tags: Annotated[
        list[str] | None,
        Field(description="Free-form tags (trimmed, lower-cased, at most 32, "
                          "1-64 characters each). Filter with tags/tags_any "
                          "on the read tools.")] = None,
    pinned: Annotated[
        bool,
        Field(description="Keep this fact in every `recall` answer of its "
                          "scope: a pinned fact is backfilled into the page "
                          "even when a relevance or decay floor would drop "
                          "it. Structural filters (entity, tags, provenance, "
                          "project) still apply.")] = False,
    links: Annotated[
        list[dict[str, Any]] | None,
        Field(description="Optional graph edges to create in the same call "
                          "(instead of a separate `relate` round trip). Each "
                          "item needs 'to_entity' and 'relation_type'; "
                          "optional 'from_entity' (default: the memory's "
                          "entity), 'weight' (0.0-1.0) and 'context'.")] = None,
) -> dict[str, Any]:
    """Store a memory with its embedding.

    `scope` defaults to "project" when a project is given, else "global";
    scope="global" writes the shared table even inside a project.
    Replace an existing fact with `supersede` (id), `supersede_content` (exact
    wording) or `supersede_last` (best-matching fact of the entity, or of the
    scope without one). `supersede_last` refuses a different topic: the fact is
    stored additionally and a warning is returned. Metadata of a replaced row is
    carried over unless `metadata_mode="replace"`.
    `dedupe`: "exact" (default), "semantic" (vector, `fuzzy_threshold`) or
    "off". A duplicate names the matched mode and similarity.
    `on_near_duplicate` handles a reworded copy below the dedupe threshold:
    "warn" (default, store and warn), "supersede" or "reject" (nothing stored).
    The alias `auto_supersede_near_duplicate` means "supersede".
    `check_contradictions` NLI-checks the nearest memories; `contradiction_mode`,
    `contradiction_min_confidence` and `max_contradiction_supersedes` bound what
    may be superseded. A project write that invalidates a global fact reports
    `cross_scope_superseded` plus a warning.
    `links` creates graph edges in the same call (`relate` semantics,
    `from_entity` defaults to the memory's entity); edge failures warn and never
    lose the fact. `metadata` must be an object; a JSON-encoded object is
    accepted. `idempotency_key` makes a retry safe: a replay stores nothing and
    answers `duplicate` with `idempotent_replay: true`.
    `pinned=True` marks the fact for always-include: `recall(include_pinned=True)`
    backfills it into the page even when a relevance or decay floor would drop
    it (structural filters still apply).

    Returns the standard envelope: {"ok", "status", "id", "error", "warnings"}.
    """
    logger.info("remember entity=%s mem_type=%s importance=%s check_contradictions=%s fuzzy=%s scope=%s dedupe=%s",
                entity, mem_type, importance, check_contradictions, fuzzy_threshold, scope, dedupe)
    raw_project = project
    try:
        project = _norm_project(project)
        entity = _norm_entity(entity)
    except ValueError as e:
        return fail(STATUS_ERROR, str(e))
    if mem_type is not None and mem_type not in MEMORY_TYPES:
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
    if contradiction_mode not in ("supersede", "warn"):
        return fail(STATUS_ERROR,
                    "contradiction_mode must be 'supersede' or 'warn'")
    try:
        contradiction_min_confidence = float(contradiction_min_confidence)
    except (TypeError, ValueError):
        return fail(STATUS_ERROR,
                    "contradiction_min_confidence must be a number between "
                    "0.0 and 1.0")
    if not 0.0 <= contradiction_min_confidence <= 1.0:
        return fail(STATUS_ERROR,
                    "contradiction_min_confidence must be between 0.0 and 1.0")
    if not isinstance(max_contradiction_supersedes, int) \
            or max_contradiction_supersedes < 0:
        return fail(STATUS_ERROR,
                    "max_contradiction_supersedes must be an integer >= 0")
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
            parse_expires_at(expires_at)
        except ValueError as e:
            return fail(STATUS_ERROR, str(e))
    try:
        valid_from = parse_iso_timestamp(valid_from, "valid_from")
        valid_to = parse_iso_timestamp(valid_to, "valid_to")
        source = normalize_optional_text(source, "source")
        agent_id = normalize_optional_text(agent_id, "agent_id")
        confidence = validate_confidence(confidence)
        tags = normalize_tags(tags)
    except ValueError as e:
        return fail(STATUS_ERROR, str(e))
    if valid_from is not None and valid_to is not None \
            and valid_to < valid_from:
        return fail(STATUS_ERROR,
                    "valid_to must not be earlier than valid_from")
    if metadata is not None:
        metadata, meta_error = _coerce_metadata(metadata)
        if meta_error:
            return fail(STATUS_ERROR, meta_error)
    try:
        idempotency_key = _norm_idempotency_key(idempotency_key)
    except ValueError as e:
        return fail(STATUS_ERROR, str(e))
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
    emb = embed([content])[0]
    db = MemoryStore()
    try:
        outcome = write_one(db, WriteIntent(
            content=content,
            project=store_project,
            scope=effective_scope,
            entity=entity,
            mem_type=mem_type,
            importance=importance,
            metadata=meta,
            metadata_mode=metadata_mode,
            expires_at=expires_at,
            dedupe=dedupe,
            fuzzy_threshold=fuzzy_threshold,
            supersede=supersede,
            supersede_content=supersede_content,
            supersede_last=supersede_last,
            on_near_duplicate=near_policy,
            check_contradictions=check_contradictions,
            contradiction_mode=contradiction_mode,
            contradiction_min_confidence=contradiction_min_confidence,
            max_contradiction_supersedes=max_contradiction_supersedes,
            links=links,
            idempotency_key=idempotency_key,
            embedding=emb,
            valid_from=valid_from,
            valid_to=valid_to,
            source=source,
            agent_id=agent_id,
            confidence=confidence,
            tags=tags,
            pinned=bool(pinned),
        ))
        if outcome.status == STATUS_DUPLICATE:
            if not outcome.persist:
                db.rollback()
                return ok(
                    STATUS_DUPLICATE, id=outcome.id,
                    duplicate_of=outcome.duplicate_of,
                    duplicate_mode=outcome.duplicate_mode,
                    similarity=outcome.similarity,
                    scope=outcome.scope,
                    project=outcome.project,
                    warnings=[_near_rejected_warning(outcome)])
            db.commit()
            warnings = [(
                outcome.warnings[0] if outcome.idempotent_replay else
                f"memory already exists; nothing stored "
                f"({outcome.duplicate_detail})")]
            if outcome.created_scope:
                warnings.append(_remember_scope_warning(store_project))
            warnings.extend(outcome.duplicate_supersede_warnings)
            if outcome.cross_scope_warning:
                warnings.append(outcome.cross_scope_warning)
            warnings.extend(outcome.link_warnings)
            payload: dict[str, Any] = {
                "duplicate_mode": outcome.duplicate_mode,
                "similarity": outcome.similarity,
                "scope": outcome.scope,
                "project": outcome.project,
                # The row that blocked the write. For an exact/near duplicate
                # `id` already IS that row, but the explicit key keeps the
                # batch entries and the single call symmetric.
                "duplicate_of": outcome.duplicate_of,
            }
            if outcome.idempotent_replay:
                payload["idempotent_replay"] = True
            if outcome.superseded:
                payload["superseded"] = outcome.superseded
                if outcome.cross_scope_superseded:
                    payload["cross_scope_superseded"] = (
                        outcome.cross_scope_superseded)
            if outcome.links:
                payload["links"] = outcome.links
            return ok(STATUS_DUPLICATE, id=outcome.id, warnings=warnings,
                      **payload)

        db.commit()
        # remember reports every invalidated row in one list; remember_batch
        # keeps NLI supersedes separate. Render the merged view here.
        superseded = outcome.superseded + [
            c["existing_id"] for c in outcome.auto_superseded
            if c["existing_id"] not in outcome.superseded]

        warnings = list(outcome.guard_warnings)
        warnings.extend(outcome.link_warnings)
        if outcome.created_scope:
            warnings.append(_remember_scope_warning(store_project))
        if outcome.ttl_warning:
            warnings.append(outcome.ttl_warning)
        if outcome.superseded_match:
            match = outcome.superseded_match
            detail = ("the same content" if match["mode"] == "exact"
                      else f"a similar content (similarity "
                           f"{match['similarity']})")
            warnings.append(
                f"this fact matches a superseded memory "
                f"({match['id']}) that was replaced by "
                f"{match['superseded_by']} ({detail} was already "
                f"invalidated). The new fact was stored as active, so both "
                f"versions are reachable now; verify this revert is intended"
            )
        if outcome.auto_near:
            warnings.append(_auto_supersede_warning(
                outcome.near_duplicate, outcome.near_conflict))
        elif (outcome.near_duplicate
              and outcome.near_duplicate["id"] not in superseded):
            confirmed = _confirmed_conflicts(outcome)
            warnings.append(_near_duplicate_warning(
                outcome.near_duplicate, entity, outcome.near_conflict,
                new_content=content,
                explicit_supersede=supersede or supersede_content,
                contradicted=bool(confirmed),
                contradiction_confidence=confirmed.get(
                    outcome.near_duplicate["id"])))
        if outcome.cross_scope_warning:
            warnings.append(outcome.cross_scope_warning)
        if outcome.cross_duplicate_warning:
            warnings.append(outcome.cross_duplicate_warning)
        contradictions = (outcome.contradictions or {}).get("superseded", [])
        kept_contradictions = (outcome.contradictions or {}).get("kept", [])
        if contradictions:
            warnings.append(
                f"superseded {len(contradictions)} contradicting memory/facts")
        if kept_contradictions:
            kept_ids = ", ".join(
                c["existing_id"] for c in kept_contradictions[:5])
            mode = (outcome.contradictions or {}).get("mode", "supersede")
            # Executable first: an agent copy-pastes supersede='<id>' into the
            # next remember call; the report path stays as the alternative.
            warnings.append(
                f"{len(kept_contradictions)} possible contradiction(s) were "
                f"kept active (contradiction_mode={mode}): {kept_ids}"
                + (" ..." if len(kept_contradictions) > 5 else "")
                + f"; verify them and resolve explicitly: supersede="
                  f"'{kept_contradictions[0]['existing_id']}' (one supersede "
                  "argument per kept id, or let contradictions_report"
                  "(resolve='newest') decide same-entity pairs)")
        if not contradictions and not kept_contradictions \
                and outcome.nli_unavailable:
            warnings.append("contradiction check skipped: NLI provider is not configured")
        elif outcome.contradiction_skip:
            warnings.append(
                f"contradiction check skipped: {outcome.contradiction_skip}")
        unjudged = (outcome.contradictions or {}).get("unjudged") or []
        if unjudged:
            warnings.append(
                f"contradiction check incomplete: NLI could not judge "
                f"{len(unjudged)} of the candidate memories "
                f"({(unjudged[0].get('reason') or 'NLI provider failure')}); "
                f"those pairs are unverified (contradictions.unjudged)")
        if check_contradictions and not entity and not outcome.nli_unavailable \
                and not outcome.contradiction_skip:
            warnings.append(
                "no entity set: the contradiction check compared the fact "
                "against the nearest memories across all entities")
        if superseded and not check_contradictions:
            warnings.append(f"superseded {len(superseded)} memory/facts")
        if outcome.metadata_carried:
            carried_detail = ", ".join(
                f"{key}={outcome.metadata.get(key)!r}"
                for key in sorted(outcome.metadata_carried))
            warnings.append(
                "metadata carried over from the replaced memories (these "
                "values come from the old fact; verify they still apply): "
                + carried_detail)

        payload = {"superseded": superseded,
                   # The scope the fact really landed in. An agent that asked
                   # for a project but wrote to the shared scope has to know
                   # that: later reads and deletes need that name.
                   "scope": outcome.scope,
                   "project": outcome.project}
        if outcome.cross_scope_superseded:
            payload["cross_scope_superseded"] = outcome.cross_scope_superseded
        if outcome.cross_scope_duplicates:
            payload["cross_scope_duplicates"] = outcome.cross_scope_duplicates
        if outcome.contradictions is not None:
            payload["contradictions"] = outcome.contradictions
        if outcome.links:
            payload["links"] = outcome.links
        return ok(STATUS_STORED, id=outcome.id, warnings=warnings, **payload)
    except Exception as e:
        db.rollback()
        logger.exception("remember failed")
        return fail(STATUS_ERROR, str(e))
    finally:
        db.close()


def _confirmed_conflicts(outcome: Any) -> dict[str, float]:
    """Confirmed NLI contradictions of a write, keyed by existing id."""
    payload = outcome.contradictions or {}
    found: dict[str, float] = {}
    for key in ("superseded", "kept"):
        for pair in payload.get(key) or []:
            confidence = float(pair.get("confidence")
                               or pair.get("similarity") or 0.0)
            if confidence >= 0.7:
                found[pair["existing_id"]] = confidence
    return found


def _remember_scope_warning(store_project: str | None) -> str:
    return (
        f"created a new project scope {store_project!r}; it did not exist "
        f"before this write (check the name if this was a typo, otherwise "
        f"the scope now appears in init_context(project='*'))")


def _batch_scope_warning(store_project: str | None) -> str:
    return (
        f"created a new project scope {store_project!r}; it did not exist "
        f"before this write (check the name if this was a typo)")


def _near_rejected_warning(outcome: Any) -> str:
    near_dup = outcome.near_duplicate
    return (
        f"near-duplicate rejected: active memory {near_dup['id']} "
        f"matches this content with similarity "
        f"{near_dup['similarity']}"
        + (", and a number or negation differs - this may be a "
           "contradiction rather than a reworded copy"
           if outcome.near_conflict else "")
        + "; nothing was stored (pass on_near_duplicate="
          "'supersede' to replace it or 'warn' to store it "
          "anyway)")


def _auto_supersede_warning(near_dup: dict, conflict: bool) -> str:
    return (
        f"auto-supersede: replaced near-duplicate "
        f"{near_dup['id']} (similarity {near_dup['similarity']}) "
        f"instead of storing a reworded copy"
        + ("; a number or negation differs, so verify this was an "
           "intended update and not a contradiction"
           if conflict else ""))


class _EntryRejected(Exception):
    """A batch entry the pipeline refused without an error (near-reject).

    Raising inside `db.attempt` rolls the entry's savepoint back, exactly
    like the hand-written ROLLBACK in the old loop, and carries the outcome
    to the entry renderer.
    """

    def __init__(self, outcome: Any):
        super().__init__("entry rejected")
        self.outcome = outcome


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
    contradiction_mode: Annotated[
        Literal["supersede", "warn"],
        Field(description="Default contradiction policy for every fact "
                          "(per-fact 'contradiction_mode' wins): 'supersede' "
                          "replaces the old memory, 'warn' only reports it.")] = "supersede",
    contradiction_min_confidence: Annotated[
        float,
        Field(description="Default minimum NLI confidence (0.0-1.0, 0.70) for "
                          "an automatic supersede.",
              ge=0.0, le=1.0)] = CONTRADICTION_AUTO_MIN_CONFIDENCE,
    max_contradiction_supersedes: Annotated[
        int,
        Field(description="Default maximum number of memories one fact may "
                          "invalidate (1). Further pairs stay active and are "
                          "reported in `contradictions.kept`.",
              ge=0)] = CONTRADICTION_AUTO_MAX_SUPERSEDES,
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
    """Store many facts in one call with per-entry savepoints.

    Each fact needs `content`. Per fact (otherwise the top-level value wins):
    `project`, `scope`, `entity`, `mem_type`, `importance`, `metadata`,
    `supersede`, `supersede_content`, `supersede_last`, `dedupe`,
    `fuzzy_threshold`, `on_near_duplicate`, `contradiction_mode`,
    `contradiction_min_confidence`, `max_contradiction_supersedes`,
    `expires_at`, `idempotency_key`, `valid_from`, `valid_to`, `source`,
    `agent_id`, `confidence`, `tags` and `links` (same edges as `remember`).
    A malformed edge is a per-fact error; a runtime edge failure keeps the fact
    and warns ("links failed: ..."). Rejected or duplicate facts store no
    edges. An unknown per-fact key is refused in that entry only.
    `check_contradictions` and `skip_duplicates` apply to every fact.
    Invalid facts are reported individually; the rest are stored (`partial`).
    A stored entry reports its edges in `links`, failed ones in
    `link_warnings`, and a duplicate names `duplicate_of`; an idempotency
    replay adds `idempotent_replay: true`. A global fact invalidated from a
    project entry reports `cross_scope_superseded` and a warning.
    `count` is the entry count, `succeeded`/`failed` split the outcomes.
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
        unknown = sorted(set(fact) - _BATCH_FACT_FIELDS)
        if unknown:
            prepared.append((None, (
                "unknown argument(s) "
                + ", ".join(repr(key) for key in unknown)
                + " for fact; accepted arguments are: "
                + ", ".join(sorted(_BATCH_FACT_FIELDS)))))
            continue
        mem_type = fact.get("mem_type")
        if mem_type is not None and mem_type not in MEMORY_TYPES:
            prepared.append((None, f"Invalid mem_type. Choose from: "
                                   f"{', '.join(sorted(MEMORY_TYPES))}"))
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
        flinks = fact.get("links")
        if flinks is not None:
            # A malformed edge is a per-fact argument error, exactly like the
            # single `remember(links=...)` path - never a silently skipped
            # edge list.
            per_link_errors = _validate_links(flinks, fentity)
            if per_link_errors:
                prepared.append((None, "; ".join(per_link_errors)))
                continue
        try:
            ft = fact.get("fuzzy_threshold", fuzzy_threshold)
            if ft is not None:
                ft = float(ft)
        except (TypeError, ValueError):
            prepared.append((None,
                             ("fuzzy_threshold must be a number between 0.0 "
                              "and 1.0")))
            continue
        if ft is not None and not 0.0 <= ft <= 1.0:
            prepared.append((None, "fuzzy_threshold must be between 0.0 and 1.0"))
            continue
        fact_dedupe = fact.get("dedupe") or dedupe
        if fact_dedupe not in DEDUPE_MODES:
            prepared.append((None,
                             (f"invalid dedupe {fact_dedupe!r}; choose from "
                              f"{', '.join(sorted(DEDUPE_MODES))}")))
            continue
        fact_near = fact.get("on_near_duplicate") or on_near_duplicate
        if fact_near not in ("warn", "supersede", "reject"):
            prepared.append((None,
                             ("on_near_duplicate must be 'warn', 'supersede' "
                              "or 'reject'")))
            continue
        try:
            fact_auto_near = _as_bool(
                fact.get("auto_supersede_near_duplicate"),
                "auto_supersede_near_duplicate")
            fact_supersede_last = _as_bool(
                fact.get("supersede_last"), "supersede_last")
            fact_pinned = _as_bool(fact.get("pinned"), "pinned")
        except ValueError as e:
            prepared.append((None, str(e)))
            continue
        if fact_auto_near and fact_near == "warn":
            fact_near = "supersede"
        fact_mode = fact.get("contradiction_mode") or contradiction_mode
        if fact_mode not in ("supersede", "warn"):
            prepared.append((None,
                             ("contradiction_mode must be 'supersede' or "
                              "'warn'")))
            continue
        fact_min_conf = fact.get("contradiction_min_confidence")
        if fact_min_conf is None:
            fact_min_conf = contradiction_min_confidence
        if not isinstance(fact_min_conf, (int, float)) \
                or not 0.0 <= float(fact_min_conf) <= 1.0:
            prepared.append((None,
                             ("contradiction_min_confidence must be between "
                              "0.0 and 1.0")))
            continue
        fact_max_supersedes = fact.get("max_contradiction_supersedes")
        if fact_max_supersedes is None:
            fact_max_supersedes = max_contradiction_supersedes
        if not isinstance(fact_max_supersedes, int) or fact_max_supersedes < 0:
            prepared.append((None,
                             "max_contradiction_supersedes must be >= 0"))
            continue
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
                parse_expires_at(fexpires)
            except ValueError:
                prepared.append((None,
                                 "expires_at must be an ISO 8601 timestamp"))
                continue
        try:
            fvalid_from = parse_iso_timestamp(
                fact.get("valid_from"), "valid_from")
            fvalid_to = parse_iso_timestamp(
                fact.get("valid_to"), "valid_to")
            fsource = normalize_optional_text(fact.get("source"), "source")
            fagent = normalize_optional_text(fact.get("agent_id"), "agent_id")
            fconfidence = validate_confidence(fact.get("confidence"))
            ftags = normalize_tags(fact.get("tags"))
        except ValueError as e:
            prepared.append((None, str(e)))
            continue
        if fvalid_from is not None and fvalid_to is not None \
                and fvalid_to < fvalid_from:
            prepared.append((None,
                             "valid_to must not be earlier than valid_from"))
            continue
        try:
            fkey = _norm_idempotency_key(fact.get("idempotency_key"))
        except ValueError as e:
            prepared.append((None, str(e)))
            continue
        prepared.append(({
            "content": content,
            "mem_type": mem_type,
            "importance": importance,
            "project": fp if fscope == "project" else None,
            "scope": fscope,
            "entity": fentity,
            "fuzzy_threshold": ft,
            "dedupe": fact_dedupe,
            "metadata": meta,
            "supersede": fact.get("supersede"),
            "supersede_content": fact.get("supersede_content"),
            "supersede_last": fact_supersede_last,
            "on_near_duplicate": fact_near,
            "contradiction_mode": fact_mode,
            "contradiction_min_confidence": float(fact_min_conf),
            "max_contradiction_supersedes": fact_max_supersedes,
            "expires_at": fexpires,
            "ttl_warning": _ttl_past_warning(fexpires),
            "links": flinks,
            "idempotency_key": fkey,
            "valid_from": fvalid_from,
            "valid_to": fvalid_to,
            "source": fsource,
            "agent_id": fagent,
            "confidence": fconfidence,
            "tags": ftags,
            "pinned": fact_pinned,
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

    db = MemoryStore()
    results: list[dict] = []
    try:
        for i, (spec, err) in enumerate(prepared):
            if spec is None:
                raw = facts[i].get("content", "") if isinstance(facts[i], dict) else ""
                results.append({"id": None, "content": raw, "status": "error",
                                "error": err, "warnings": []})
                continue
            store_project = spec["project"]
            try:
                with db.attempt(f"fact_{i}"):
                    outcome = write_one(db, WriteIntent(
                        content=spec["content"],
                        project=store_project,
                        scope=spec["scope"],
                        entity=spec["entity"],
                        mem_type=spec["mem_type"],
                        importance=spec["importance"],
                        metadata=spec["metadata"],
                        expires_at=spec.get("expires_at"),
                        dedupe=spec.get("dedupe") or dedupe,
                        fuzzy_threshold=spec["fuzzy_threshold"],
                        supersede=spec.get("supersede"),
                        supersede_content=spec.get("supersede_content"),
                        supersede_last=spec.get("supersede_last"),
                        on_near_duplicate=spec["on_near_duplicate"],
                        check_contradictions=check_contradictions,
                        contradiction_mode=spec["contradiction_mode"],
                        contradiction_min_confidence=spec[
                            "contradiction_min_confidence"],
                        max_contradiction_supersedes=spec[
                            "max_contradiction_supersedes"],
                        links=spec["links"],
                        embedding=embeddings[i],
                        skip_duplicates=skip_duplicates,
                        probe_nli_config=False,
                        report_superseded_match=False,
                        idempotency_key=spec["idempotency_key"],
                        valid_from=spec["valid_from"],
                        valid_to=spec["valid_to"],
                        source=spec["source"],
                        agent_id=spec["agent_id"],
                        confidence=spec["confidence"],
                        tags=spec["tags"],
                        pinned=spec["pinned"],
                    ))
                    if not outcome.persist:
                        raise _EntryRejected(outcome)
            except _EntryRejected as rejected:
                outcome = rejected.outcome
            except Exception as exc:
                results.append({"id": None, "content": spec["content"],
                                "status": "error", "error": str(exc),
                                "scope": spec["scope"],
                                "project": (store_project or "global"),
                                "warnings": []})
                continue

            if outcome.status == STATUS_DUPLICATE:
                entry: dict[str, Any] = {
                    "id": outcome.id, "content": spec["content"],
                    "status": "duplicate",
                    "duplicate_mode": outcome.duplicate_mode,
                    "similarity": outcome.similarity,
                    "scope": outcome.scope,
                    "project": outcome.project,
                }
                warnings: list[str] = []
                if outcome.idempotent_replay:
                    entry["idempotent_replay"] = True
                    warnings.extend(outcome.warnings)
                if outcome.created_scope:
                    warnings.append(_batch_scope_warning(store_project))
                if not outcome.persist:
                    warnings.append(
                        f"near-duplicate rejected: active memory "
                        f"{outcome.near_duplicate['id']} matches this content "
                        f"with similarity {outcome.near_duplicate['similarity']}"
                        + (", and a number or negation differs - this may be a "
                           "contradiction rather than a reworded copy"
                           if outcome.near_conflict else "")
                        + "; nothing was stored")
                else:
                    warnings.extend(outcome.duplicate_supersede_warnings)
                    if outcome.cross_scope_warning:
                        warnings.append(outcome.cross_scope_warning)
                # Stable shape: every entry exposes `warnings`, so a caller
                # can iterate without a key check.
                entry["warnings"] = warnings
                if outcome.superseded:
                    entry["superseded"] = outcome.superseded
                    if outcome.cross_scope_superseded:
                        entry["cross_scope_superseded"] = (
                            outcome.cross_scope_superseded)
                if outcome.duplicate_of:
                    entry["duplicate_of"] = outcome.duplicate_of
                results.append(entry)
                continue

            entry = {
                "id": outcome.id,
                "content": spec["content"],
                "status": "stored",
                "superseded": outcome.superseded,
                # Where the fact really landed: a batch can mix scopes, so
                # the caller has to be able to see it per fact.
                "scope": outcome.scope,
                "project": outcome.project,
                # Stable shape: a caller can read entry["warnings"] without a
                # key check, also when there is nothing to say.
                "warnings": [],
            }
            if outcome.guard_warnings:
                entry["warnings"] = list(outcome.guard_warnings)
            if outcome.links:
                # Same shape as remember(links=[...]): per-edge status plus
                # the roll-back warnings of failed edges (the fact stays).
                entry["links"] = outcome.links
                entry["warnings"].extend(outcome.link_warnings)
            if outcome.exact_copy:
                entry["warnings"].append(
                    f"identical content is already active as {outcome.exact_copy}; "
                    f"another copy was stored because skip_duplicates=false")
            if outcome.created_scope:
                entry["warnings"].append(_batch_scope_warning(store_project))
            if outcome.ttl_warning:
                entry["warnings"].append(outcome.ttl_warning)
            if outcome.metadata_carried:
                entry["warnings"].append(
                    "metadata carried over from the replaced memories (these "
                    "values come from the old fact; verify they still apply): "
                    + ", ".join(sorted(set(outcome.metadata_carried))))
            if outcome.auto_near:
                entry["warnings"].append(_auto_supersede_warning(
                    outcome.near_duplicate, outcome.near_conflict))
            elif (outcome.near_duplicate
                  and outcome.near_duplicate["id"] not in outcome.superseded):
                # Already replaced through the explicit supersede path: the
                # entry's `superseded` list already names it, so no extra
                # advice is needed.
                confirmed = _confirmed_conflicts(outcome)
                entry["warnings"].append(_near_duplicate_warning(
                    outcome.near_duplicate, spec["entity"],
                    outcome.near_conflict, new_content=spec["content"],
                    contradicted=bool(confirmed),
                    contradiction_confidence=confirmed.get(
                        outcome.near_duplicate["id"])))
            if outcome.cross_scope_superseded:
                entry["cross_scope_superseded"] = (
                    outcome.cross_scope_superseded)
                entry["warnings"].append(outcome.cross_scope_warning)
            if outcome.cross_scope_duplicates:
                entry["cross_scope_duplicates"] = outcome.cross_scope_duplicates
                entry["warnings"].append(outcome.cross_duplicate_warning)
            if check_contradictions:
                entry["contradictions"] = dict(
                    outcome.contradictions
                    or {"found": False, "superseded": [], "kept": []})
                kept = entry["contradictions"].get("kept") or []
                if kept:
                    entry["warnings"].append(
                        f"{len(kept)} possible contradiction(s) were kept "
                        f"active: verify them and resolve explicitly: "
                        f"supersede='{kept[0]['existing_id']}' (one supersede "
                        "argument per kept id, or let contradictions_report"
                        "(resolve='newest') decide same-entity pairs)")
                unjudged = entry["contradictions"].get("unjudged") or []
                if unjudged:
                    unreason = (unjudged[0].get("reason")
                                or "NLI provider failure")
                    entry["warnings"].append(
                        f"contradiction check incomplete: NLI could not judge "
                        f"{len(unjudged)} of the candidate memories "
                        f"({unreason}); those pairs are unverified "
                        "(contradictions.unjudged)")
                if outcome.contradiction_skip:
                    entry["contradictions"]["skipped"] = (
                        outcome.contradiction_skip)
                    entry["warnings"].append(
                        f"contradiction check skipped: "
                        f"{outcome.contradiction_skip}")
                elif not spec["entity"]:
                    # Transparency: the check ran, but across all entities.
                    entry["contradictions"]["cross_entity"] = True
                    entry["warnings"].append(
                        "no entity set: the contradiction check compared the "
                        "fact against the nearest memories across all entities")
            results.append(entry)
        db.commit()
    except Exception as e:
        db.rollback()
        logger.exception("remember_batch failed")
        return fail(STATUS_ERROR, str(e))
    finally:
        db.close()
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

def _restore_blocked_reason(db: MemoryStore, memory_id: str,
                            tables: list[str]) -> tuple[str, str] | None:
    """Status and reason for an existing row that cannot be restored.

    "not found" was a lie for an active memory: the id exists, it just has
    nothing to restore. Such a request answers `conflict`; an unknown id stays
    `not_found`. A superseded history row is a conflict too - a restore never
    undoes a supersede.
    """
    found = lifecycle_row(db, memory_id, tables)
    if found is None:
        return None
    _, row = found
    deleted_at = row.get("deleted_at")
    superseded_by = row.get("superseded_by")
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


def _restore_memory(memory_id: str, project: str | None) -> dict[str, Any]:
    """Clear the soft delete of one memory (inverse of `forget(soft=True)`).

    A real `superseded_by` id is kept: a restored memory cannot resurrect
    next to the fact that replaced it. With a project only that project's
    table is searched (the explicit scope); without one every scope is
    searched and an ambiguous id is refused.
    """
    db = MemoryStore()
    try:
        cur = db.cur
        if project:
            tables = [_mem_table(project)]
            if not _table_exists(cur, tables[0]):
                return fail(STATUS_NOT_FOUND, f"NOT FOUND: {memory_id}")
        else:
            scopes = db.scopes_of(memory_id)
            if len(scopes) > 1:
                db.rollback()
                return fail(STATUS_ERROR,
                            f"ambiguous memory id {memory_id}: it exists in "
                            f"the scopes {scopes}; pass project=<name> to say "
                            f"which one to restore",
                            scopes=scopes)
            tables = _all_mem_tables(cur)
        restored = 0
        restored_table = None
        for table in tables:
            # NULLIF clears only the forgotten sentinel; a `superseded_by` id
            # from a real supersede stays set, so a restored memory cannot
            # resurrect next to the fact that replaced it.
            count = db.restore(memory_id, tables=[table])
            if count:
                restored += count
                restored_table = table
                break
        db.commit()
        if not restored:
            blocked = _restore_blocked_reason(db, memory_id, tables)
            if blocked:
                status, reason = blocked
                return fail(status, reason)
            return fail(STATUS_NOT_FOUND, f"NOT FOUND: {memory_id}")
        scope = _scope_of_table(cur, restored_table) if restored_table else None
    except Exception as e:
        db.rollback()
        return fail(STATUS_ERROR, str(e))
    finally:
        db.close()
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
    valid_from: Annotated[
        str | None,
        Field(description="New ISO 8601 timestamp when this fact became "
                          "true.")] = None,
    valid_to: Annotated[
        str | None,
        Field(description="New ISO 8601 timestamp when this fact stops being "
                          "true. Mutually exclusive with clear_valid_to.")] = None,
    clear_valid_to: Annotated[
        bool,
        Field(description="Reopen the validity window (sets valid_to to "
                          "NULL). Mutually exclusive with valid_to.")] = False,
    source: Annotated[
        str | None,
        Field(description="Where the fact came from (max 200 characters); an "
                          "empty string clears it.")] = None,
    agent_id: Annotated[
        str | None,
        Field(description="Identifier of the writing agent/client (max 200 "
                          "characters); an empty string clears it.")] = None,
    confidence: Annotated[
        float | None,
        Field(ge=0.0, le=1.0,
              description="How sure the source is of this fact (0.0-1.0).")] = None,
    tags: Annotated[
        list[str] | None,
        Field(description="New tags (trimmed, lower-cased, at most 32). "
                          "`tags_mode` decides whether they replace or extend "
                          "the stored tags.")] = None,
    tags_mode: Annotated[
        Literal["replace", "merge"],
        Field(description="'replace' (default) sets exactly `tags`; 'merge' "
                          "adds them to the stored tags.")] = "replace",
    clear_tags: Annotated[
        bool,
        Field(description="Remove every tag from the memory.")] = False,
    pinned: Annotated[
        bool | None,
        Field(description="Set or clear the always-include flag: a pinned "
                          "fact is backfilled into a `recall(include_pinned="
                          "True)` page even when a relevance or decay floor "
                          "would drop it. Omit to leave it unchanged.")] = None,
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
    """Update content, importance, metadata, entity, type or TTL of a memory.

    Without `project` the id is looked up in every scope; an id found in
    several scopes is refused with the scopes named, so pass `project`.
    `content` re-embeds and dedupe-checks the edit (`dedupe`/`fuzzy_threshold`).
    `expires_at` sets a TTL and `clear_expires_at=True` removes it.
    `valid_from`/`valid_to` move the validity window (`clear_valid_to=True`
    reopens it); `source`/`agent_id`/`confidence` update provenance,
    `tags`/`tags_mode`/`clear_tags` the tags and
    `pinned=True`/`pinned=False` the always-include flag.
    `touch=True` only refreshes the decay age; `restore=True` undoes a soft
    delete and cannot be combined. A superseded or expired row answers
    `conflict`; revive an expired row with a future `expires_at` or
    `clear_expires_at`, and a real `superseded_by` is never cleared.
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
    if valid_to is not None and clear_valid_to:
        return fail(STATUS_ERROR,
                    "pass either valid_to or clear_valid_to, not both")
    if tags_mode not in ("replace", "merge"):
        return fail(STATUS_ERROR,
                    "tags_mode must be 'replace' or 'merge'")
    source_provided = source is not None
    agent_provided = agent_id is not None
    try:
        valid_from_value = parse_iso_timestamp(valid_from, "valid_from")
        valid_to_value = parse_iso_timestamp(valid_to, "valid_to")
        source = normalize_optional_text(source, "source")
        agent_id = normalize_optional_text(agent_id, "agent_id")
        confidence_value = validate_confidence(confidence)
        tags_value = normalize_tags(tags)
    except ValueError as e:
        return fail(STATUS_ERROR, str(e))
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
            expires_value = parse_expires_at(expires_at)
        except ValueError as e:
            return fail(STATUS_ERROR, str(e))
    # An expired row is inactive. Editing its content or metadata would
    # silently reactivate a fact whose TTL said "stop using this", so the
    # caller has to revive it explicitly: a future `expires_at` or
    # `clear_expires_at`. A plain `touch` on an expired row is refused too.
    revive_expired = bool(clear_expires_at) or (
        expires_value is not None and not _is_past(expires_value))
    if restore:
        if (content is not None or importance is not None
                or metadata is not None or entity is not None
                or mem_type is not None or expires_at is not None
                or clear_expires_at or touch or clear_keys
                or valid_from is not None or valid_to is not None
                or clear_valid_to or source_provided or agent_provided
                or confidence is not None or tags is not None
                or clear_tags or pinned is not None):
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
    if valid_from is not None:
        sets.append("valid_from = %s")
        params.append(valid_from_value)
    if valid_to is not None:
        sets.append("valid_to = %s")
        params.append(valid_to_value)
    elif clear_valid_to:
        sets.append("valid_to = NULL")
    if source_provided:
        sets.append("source = %s")
        params.append(source)
    if agent_provided:
        sets.append("agent_id = %s")
        params.append(agent_id)
    if confidence_value is not None:
        sets.append("confidence = %s")
        params.append(confidence_value)
    if pinned is not None:
        sets.append("pinned = %s")
        params.append(bool(pinned))

    if sets:
        # Updated memories must also age like new ones for decay scoring.
        sets.append("updated_at = NOW()")
    elif touch:
        # A pure refresh: no content change, only the decay age restarts.
        sets.append("updated_at = NOW()")

    db = MemoryStore()
    try:
        cur = db.cur
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
                db.rollback()
                return fail(STATUS_ERROR,
                            f"ambiguous memory id {memory_id}: it exists in "
                            f"the scopes {scopes}; pass project=<name> to say "
                            f"which one to update",
                            scopes=scopes)
            tables = _all_mem_tables(cur)
        # Locate the memory BEFORE the "nothing to update" check: an unknown
        # id must answer NOT FOUND even when no field was passed, otherwise
        # the validation message suggests the id was seen and is valid.
        active_table = active_table_for(
            db, memory_id, tables, revive_expired=revive_expired)
        if active_table is None:
            reason = _hidden_state_reason(db, memory_id, tables)
            db.rollback()
            if reason:
                return fail(
                    STATUS_CONFLICT,
                    f"memory {memory_id} exists but cannot be updated: {reason}")
            return fail(STATUS_NOT_FOUND, f"NOT FOUND: {memory_id}")
        if metadata_value is not None or clear_keys:
            # Metadata merge and key removal need the stored object. This is
            # the only field that is computed from the existing row; the
            # caller's new keys win, missing ones survive in 'merge' mode.
            current = stored_metadata(db, active_table, memory_id)
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
        if tags_value is not None or clear_tags:
            if tags_mode == "merge":
                merged_tags = list(dict.fromkeys(
                    stored_tags(db, active_table, memory_id)
                    + list(tags_value or [])))
            else:
                merged_tags = list(tags_value or [])
            if len(merged_tags) > MAX_TAGS:
                db.rollback()
                return fail(STATUS_ERROR,
                            f"tags accepts at most {MAX_TAGS} tags "
                            f"(after merging the stored tags)")
            sets.append("tags = %s::text[]")
            params.append(merged_tags)
        if sets and "updated_at = NOW()" not in sets:
            # A metadata-only edit must age like every other update.
            sets.append("updated_at = NOW()")
        if not sets:
            db.rollback()
            return fail(STATUS_ERROR,
                        "nothing to update: pass content, importance, metadata, "
                        "clear_metadata_keys, entity, mem_type, "
                        "expires_at/clear_expires_at, valid_from, "
                        "valid_to/clear_valid_to, source, agent_id, "
                        "confidence, tags/tags_mode/clear_tags, pinned, "
                        "touch=true or restore=true")
        scope = _scope_of_table(cur, active_table)
        # `entity=None` means "keep the current entity", not "untagged": the
        # near-duplicate warning needs the entity the updated row will have.
        near_entity = entity
        if near_entity is None:
            near_entity = stored_entity(db, active_table, memory_id)
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
                db.rollback()
                detail = ("exact content hash" if dup_mode == "exact"
                          else f"semantic match (similarity {dup_similarity})")
                return fail(
                    # Nothing was changed, so `ok: false` is the honest
                    # answer: an agent must not read this as "updated".
                    # `id` names the existing memory that blocked the edit
                    # (same contract as remember); the id the caller tried
                    # to edit is `memory_id`.
                    STATUS_DUPLICATE,
                    f"update not applied: the new content would duplicate "
                    f"active memory {dup_id} ({detail}); nothing was changed",
                    id=dup_id, memory_id=memory_id,
                    duplicate_of=dup_id, applied=False,
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
        if not apply_fields(db, active_table, memory_id, sets, params):
            db.rollback()
            return fail(STATUS_NOT_FOUND, f"NOT FOUND: {memory_id}")
        db.commit()
    except Exception as e:
        db.rollback()
        return fail(STATUS_ERROR, str(e))
    finally:
        db.close()
    warnings = ([_near_duplicate_warning(near_dup, near_entity,
                                         near_conflict, update=True,
                                         new_content=content)]
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
                          "'clear_expires_at', 'valid_from', 'valid_to', "
                          "'clear_valid_to', 'source', 'agent_id', "
                          "'confidence', 'tags', 'tags_mode', 'clear_tags', "
                          "'touch', 'restore' and 'project'. Invalid entries "
                          "are reported individually and the rest is still "
                          "applied.")],
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

    Each entry works like `update_memory`: `content` re-embeds, then
    `importance`/`metadata`/`entity`/`mem_type`, TTL, validity
    (`valid_from`/`valid_to`/`clear_valid_to`), provenance (`source`,
    `agent_id`, `confidence`) and tags (`tags`/`tags_mode`/`clear_tags`) are
    applied. An expired row answers `conflict` unless revived with a future
    `expires_at` or `clear_expires_at`. `touch=true` refreshes decay and
    combines with other fields; `restore=true` undoes a soft delete and is
    exclusive.
    An unknown id answers `not_found`, an id in several scopes without a
    `project` answers `error` with the `scopes`; one bad entry never aborts the
    rest. Answers `count` + `results`, `status="partial"` on failures.
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
        try:
            entry_clear = _as_bool(entry.get("clear_expires_at"),
                                   "clear_expires_at")
            entry_clear_valid_to = _as_bool(entry.get("clear_valid_to"),
                                            "clear_valid_to")
            entry_clear_tags = _as_bool(entry.get("clear_tags"), "clear_tags")
            entry_restore = _as_bool(entry.get("restore"), "restore")
            entry_touch = _as_bool(entry.get("touch"), "touch")
        except ValueError as exc:
            error = str(exc)
            entry_clear = entry_clear_valid_to = entry_clear_tags = False
            entry_restore = entry_touch = False
        if error is None and entry_expires is not None and entry_clear:
            error = "pass either expires_at or clear_expires_at, not both"
        elif error is None and entry_expires is not None:
            try:
                fields["expires_at"] = parse_expires_at(entry_expires)
            except ValueError:
                error = "expires_at must be an ISO 8601 timestamp"
        elif error is None and entry_clear:
            fields["expires_at"] = None
        entry_ttl_warning = (_ttl_past_warning(entry_expires)
                             if entry_expires is not None and error is None
                             else None)
        entry_valid_from = entry.get("valid_from")
        entry_valid_to = entry.get("valid_to")
        if error is None and entry_valid_to is not None \
                and entry_clear_valid_to:
            error = "pass either valid_to or clear_valid_to, not both"
        if error is None and entry_valid_from is not None:
            try:
                fields["valid_from"] = parse_iso_timestamp(
                    entry_valid_from, "valid_from")
            except ValueError as exc:
                error = str(exc)
        if error is None and entry_valid_to is not None:
            try:
                fields["valid_to"] = parse_iso_timestamp(
                    entry_valid_to, "valid_to")
            except ValueError as exc:
                error = str(exc)
        elif error is None and entry_clear_valid_to:
            fields["valid_to"] = None
        for name, label in (("source", "source"), ("agent_id", "agent_id")):
            raw_text = entry.get(name)
            if error is None and raw_text is not None:
                try:
                    fields[name] = normalize_optional_text(raw_text, label)
                except ValueError as exc:
                    error = str(exc)
        entry_confidence = entry.get("confidence")
        if error is None and entry_confidence is not None:
            try:
                fields["confidence"] = validate_confidence(entry_confidence)
            except ValueError as exc:
                error = str(exc)
        entry_pinned = entry.get("pinned")
        if error is None and entry_pinned is not None:
            try:
                fields["pinned"] = _as_bool(entry_pinned, "pinned")
            except ValueError as exc:
                error = str(exc)
        entry_tags = entry.get("tags")
        entry_tags_mode = entry.get("tags_mode") or "replace"
        if error is None and entry_tags_mode not in ("replace", "merge"):
            error = "tags_mode must be 'replace' or 'merge'"
        if error is None and entry_tags is not None:
            try:
                entry_tags = normalize_tags(entry_tags)
            except ValueError as exc:
                error = str(exc)
        if error is None and entry_restore and (
                fields or entry_touch or entry_clear_keys
                or entry_tags is not None or entry_clear_tags):
            error = ("restore=true cannot be combined with other update "
                     "fields")
        if error is None and not fields and not entry_restore \
                and not entry_touch and not entry_clear_keys \
                and entry_tags is None and not entry_clear_tags:
            error = ("nothing to update: pass content, importance, metadata, "
                     "clear_metadata_keys, entity, mem_type, "
                     "expires_at/clear_expires_at, valid_from, "
                     "valid_to/clear_valid_to, source, agent_id, confidence, "
                     "tags/tags_mode/clear_tags, pinned, touch=true or "
                     "restore=true")
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
                              "tags": entry_tags,
                              "tags_mode": entry_tags_mode,
                              "clear_tags": entry_clear_tags,
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

    db = MemoryStore()
    results: list[dict] = []
    try:
        cur = db.cur

        def _undo(savepoint: str) -> None:
            # ROLLBACK TO leaves the savepoint on the stack; without the
            # release a long batch accumulates one savepoint per failed entry.
            db.rollback_to(savepoint)
            db.release(savepoint)

        for i, (spec, err) in enumerate(prepared):
            if spec is None:
                raw = updates[i]
                raw_id = raw.get("memory_id") if isinstance(raw, dict) else None
                results.append({"id": raw_id, "status": "error", "error": err,
                                "warnings": []})
                continue
            memory_id = spec["memory_id"]
            entry_project = spec["project"]
            savepoint = f"update_{i}"
            db.open_savepoint(savepoint)
            try:
                if spec.get("restore"):
                    if entry_project:
                        restore_tables = [_mem_table(entry_project)]
                        if not _table_exists(cur, restore_tables[0]):
                            _undo(savepoint)
                            results.append({
                                "id": memory_id, "status": "not_found",
                                "error": f"NOT FOUND: {memory_id}",
                                "warnings": []})
                            continue
                    else:
                        scopes = _memory_scopes(cur, memory_id)
                        if len(scopes) > 1:
                            _undo(savepoint)
                            results.append({
                                "id": memory_id, "status": "error",
                                "error": (f"ambiguous memory id {memory_id}: "
                                          f"it exists in the scopes {scopes}; "
                                          f"pass project=<name> to say which "
                                          f"one to restore"),
                                "scopes": scopes, "warnings": []})
                            continue
                        restore_tables = _all_mem_tables(cur)
                    restored_table, restored = restore_rows(
                        db, memory_id, restore_tables)
                    if not restored:
                        blocked = _restore_blocked_reason(
                            db, memory_id, restore_tables)
                        _undo(savepoint)
                        results.append({
                            "id": memory_id,
                            "status": blocked[0] if blocked else "not_found",
                            "error": (blocked[1] if blocked
                                      else f"NOT FOUND: {memory_id}"),
                            "warnings": []})
                        continue
                    scope = _scope_of_table(cur, restored_table)
                    db.release(savepoint)
                    results.append({"id": memory_id,
                                    "status": STATUS_RESTORED,
                                    "project": scope, "warnings": []})
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
                        _undo(savepoint)
                        results.append({
                            "id": memory_id, "status": "error",
                            "error": (f"ambiguous memory id {memory_id}: it "
                                      f"exists in the scopes {scopes}; pass "
                                      f"project=<name> to say which one to "
                                      f"update"),
                            "scopes": scopes, "warnings": []})
                        continue
                    tables = _all_mem_tables(cur)

                entry_fields = spec["fields"]
                revive_expired = (
                    "expires_at" in entry_fields
                    and (entry_fields["expires_at"] is None
                         or not _is_past(entry_fields["expires_at"])))
                active_table = active_table_for(
                    db, memory_id, tables, revive_expired=revive_expired)
                if active_table is None:
                    reason = _hidden_state_reason(db, memory_id, tables)
                    _undo(savepoint)
                    error = (f"memory {memory_id} exists but cannot be "
                             f"updated: {reason}" if reason
                             else f"NOT FOUND: {memory_id}")
                    results.append({
                        "id": memory_id,
                        "status": "conflict" if reason else "not_found",
                        "error": error, "warnings": []})
                    continue
                scope = _scope_of_table(cur, active_table)
                if spec["metadata_mode"] == "merge" \
                        or spec["clear_metadata_keys"]:
                    current = stored_metadata(db, active_table, memory_id)
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
                if spec["tags"] is not None or spec["clear_tags"]:
                    if spec["tags_mode"] == "merge":
                        merged_tags = list(dict.fromkeys(
                            stored_tags(db, active_table, memory_id)
                            + list(spec["tags"] or [])))
                    else:
                        merged_tags = list(spec["tags"] or [])
                    if len(merged_tags) > MAX_TAGS:
                        _undo(savepoint)
                        results.append({
                            "id": memory_id, "status": "error",
                            "error": (f"tags accepts at most {MAX_TAGS} tags "
                                      f"(after merging the stored tags)"),
                            "warnings": []})
                        continue
                    sets.append("tags = %s::text[]")
                    params.append(merged_tags)
                near_entity = spec["fields"].get("entity")
                if near_entity is None:
                    near_entity = stored_entity(db, active_table, memory_id)
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
                        db.release(savepoint)
                        results.append({
                            "id": dup_id, "memory_id": memory_id,
                            "status": "duplicate",
                            "ok": False,
                            "applied": False,
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
                if not apply_fields(db, active_table, memory_id, sets,
                                    params):
                    _undo(savepoint)
                    results.append({"id": memory_id, "status": "not_found",
                                    "error": f"NOT FOUND: {memory_id}",
                                    "warnings": []})
                    continue
                db.release(savepoint)
            except Exception as exc:
                _undo(savepoint)
                results.append({"id": memory_id, "status": "error",
                                "error": str(exc), "warnings": []})
                continue
            entry: dict[str, Any] = {"id": memory_id,
                                     "status": STATUS_UPDATED,
                                     "project": scope, "warnings": []}
            if near_dup:
                entry["warnings"].append(
                    _near_duplicate_warning(
                        near_dup, near_entity, near_conflict, update=True,
                        new_content=spec["fields"]["content"]))
            if spec.get("ttl_warning"):
                entry["warnings"].append(spec["ttl_warning"])
            results.append(entry)
        db.commit()
    except Exception as e:
        db.rollback()
        logger.exception("update_batch failed")
        return fail(STATUS_ERROR, str(e))
    finally:
        db.close()
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



def _hard_delete_window() -> tuple[dict[str, str], str]:
    """Timestamp a hard delete stays undoable and the matching warning."""
    until = datetime.now(timezone.utc) + timedelta(days=HARD_DELETE_GRACE_DAYS)
    iso = until.isoformat(timespec="seconds")
    return (
        {"undo_until": iso, "purge_after": iso},
        f"recoverable until {iso}, purged afterwards",
    )


def _forget_by_selector(
    db: MemoryStore,
    cur,
    project: str | None,
    scope_name: str,
    explicit_scope: bool,
    memory_id: str | None,
    memory_ids: list[str] | None,
    entity_name: str | None,
    metadata_filter: Any,
    contains: str | None,
    *,
    soft: bool,
    dry_run: bool,
) -> dict[str, Any]:
    """Remove the active memories matched by `metadata_filter`/`contains`."""
    if metadata_filter is not None and contains is not None:
        db.rollback()
        return fail(STATUS_ERROR,
                    "metadata_filter and contains are mutually exclusive")
    if memory_id or memory_ids or entity_name:
        db.rollback()
        return fail(STATUS_ERROR,
                    "metadata_filter and contains cannot be combined with "
                    "memory_id, memory_ids or entity")
    if not explicit_scope:
        db.rollback()
        return fail(STATUS_ERROR,
                    "metadata_filter and contains remove a selected set of "
                    "memories: pass project=<name> to name the scope "
                    "(the shared scope is addressed with project=\"global\")")
    if metadata_filter is not None:
        if isinstance(metadata_filter, str):
            try:
                metadata_filter = json.loads(metadata_filter)
            except json.JSONDecodeError:
                db.rollback()
                return fail(STATUS_ERROR,
                            "metadata_filter must be an object, not a string")
        if not isinstance(metadata_filter, dict):
            db.rollback()
            return fail(STATUS_ERROR, "metadata_filter must be an object")
        if not metadata_filter:
            db.rollback()
            return fail(STATUS_ERROR, "metadata_filter must not be empty")
    if contains is not None:
        if not isinstance(contains, str) or not contains.strip():
            db.rollback()
            return fail(STATUS_ERROR, "contains must be a non-empty string")
        contains = contains.strip()

    counts = {"memories": 0, "entities": 0, "relations": 0}
    dry_status = "would_soft_forget" if soft else "would_delete"
    live_status = STATUS_SOFT_FORGOTTEN if soft else STATUS_DELETED
    reason = "metadata_filter" if metadata_filter is not None else "contains"
    table = _mem_table(project)
    if not _table_exists(cur, table):
        db.rollback()
        return ok(
            STATUS_OK, id=None, project=scope_name, counts=counts,
            count=0, scopes_affected=[], bulk_results=[], dry_run=dry_run,
            warnings=[f"no memories matched {reason} in scope {scope_name!r}"],
            message=f"no memories matched {reason} in {scope_name}",
        )

    conditions = [ACTIVE_ROW_SQL]
    params: list[Any] = []
    if metadata_filter is not None:
        meta_conditions, meta_params = _extra_conditions(None, metadata_filter)
        conditions.extend(meta_conditions)
        params.extend(meta_params)
    if contains is not None:
        conditions.append("content ILIKE %s")
        params.append(f"%{escape_like(contains)}%")
    where = " AND ".join(conditions)
    rows = db.all(
        f"SELECT id, content FROM {table} WHERE {where} "
        f"ORDER BY updated_at DESC",
        params,
    )
    scope_hit = _scope_of_table(cur, table)
    bulk_results: list[dict] = []
    for row in rows:
        removed, _touched = db.forget_rows(
            row["id"], tables=[table], soft=soft, dry_run=dry_run)
        if not removed:
            continue
        counts["memories"] += removed
        preview = " ".join(str(row.get("content") or "").split())
        if len(preview) > 120:
            preview = preview[:117] + "..."
        bulk_results.append({
            "id": row["id"],
            "content_preview": preview,
            "status": dry_status if dry_run else live_status,
            "count": removed,
            "scopes": [scope_hit],
            "reason": reason,
        })

    if not bulk_results:
        db.rollback()
        return ok(
            STATUS_OK, id=None, project=scope_name, counts=counts,
            count=0, scopes_affected=[], bulk_results=[], dry_run=dry_run,
            warnings=[f"no memories matched {reason} in scope {scope_name!r}"],
            message=f"no memories matched {reason} in {scope_name}",
        )

    if dry_run:
        db.rollback()
    else:
        if not soft:
            clear_dangling_supersede(cur)
        db.commit()
    warnings: list[str] = []
    extra: dict[str, Any] = {}
    if not soft and not dry_run:
        window, warning = _hard_delete_window()
        extra.update(window)
        warnings.append(warning)
    return ok(
        STATUS_OK if dry_run else live_status, id=None, project=scope_name,
        counts=counts, count=len(bulk_results),
        scopes_affected=[scope_hit], bulk_results=bulk_results,
        dry_run=dry_run, **extra, warnings=warnings,
        message=(f"{'Would remove' if dry_run else 'Removed'} "
                 f"{counts['memories']} memory/memories matching {reason} "
                 f"in {scope_name}"),
    )


def _forget_by_query(
    db: MemoryStore,
    cur,
    project: str | None,
    scope_name: str,
    explicit_scope: bool,
    memory_id: str | None,
    memory_ids: list[str] | None,
    entity_name: str | None,
    metadata_filter: Any,
    contains: str | None,
    query: str,
    limit: int,
    min_similarity: float | None,
    *,
    soft: bool,
    dry_run: bool,
) -> dict[str, Any]:
    """Remove the active memories that semantically match `query`."""
    if not isinstance(query, str) or not query.strip():
        db.rollback()
        return fail(STATUS_ERROR, "query must be a non-empty string")
    query = query.strip()
    if metadata_filter is not None or contains is not None:
        db.rollback()
        return fail(STATUS_ERROR,
                    "query and metadata_filter/contains are mutually exclusive")
    if memory_id or memory_ids or entity_name:
        db.rollback()
        return fail(STATUS_ERROR,
                    "query cannot be combined with memory_id, memory_ids or "
                    "entity")
    if not explicit_scope:
        db.rollback()
        return fail(STATUS_ERROR,
                    "query removes a selected set of memories: pass "
                    "project=<name> to name the scope (the shared scope is "
                    "addressed with project=\"global\")")
    if not isinstance(limit, int) or limit < 1:
        db.rollback()
        return fail(STATUS_ERROR, "limit must be a positive integer")
    if min_similarity is not None and not 0.0 <= min_similarity <= 1.0:
        db.rollback()
        return fail(STATUS_ERROR,
                    "min_similarity must be between 0.0 and 1.0")

    counts = {"memories": 0, "entities": 0, "relations": 0}
    dry_status = "would_soft_forget" if soft else "would_delete"
    live_status = STATUS_SOFT_FORGOTTEN if soft else STATUS_DELETED
    table = _mem_table(project)
    scope_hit = _scope_of_table(cur, table)
    if not _table_exists(cur, table):
        db.rollback()
        return ok(
            STATUS_OK, id=None, project=scope_name, counts=counts,
            count=0, scopes_affected=[], bulk_results=[], dry_run=dry_run,
            query=query,
            warnings=["no memories matched the query"],
            message=f"no memories matched the query in {scope_name}",
        )

    try:
        query_embedding = embed([query], mode="query")[0]
    except Exception as e:
        db.rollback()
        return fail(STATUS_ERROR, f"query embedding failed: {e}")

    rows = _search_memories(cur, SearchRequest(
        query_embedding=query_embedding,
        query_text=query,
        project=project,
        tables=[table],
        limit=limit,
        min_similarity=min_similarity,
        exclude_relation_context=True,
    ))

    bulk_results: list[dict] = []
    for row in rows:
        removed, _touched = db.forget_rows(
            row["id"], tables=[table], soft=soft, dry_run=dry_run)
        if not removed:
            continue
        counts["memories"] += removed
        preview = " ".join(str(row.get("content") or "").split())
        if len(preview) > 200:
            preview = preview[:197] + "..."
        similarity = row.get("_similarity")
        bulk_results.append({
            "id": row["id"],
            "content_preview": preview,
            "similarity": (round(float(similarity), 4)
                           if similarity is not None else 0.0),
            "status": dry_status if dry_run else live_status,
            "count": removed,
            "scopes": [scope_hit],
            "reason": "query",
        })

    if not bulk_results:
        db.rollback()
        return ok(
            STATUS_OK, id=None, project=scope_name, counts=counts,
            count=0, scopes_affected=[], bulk_results=[], dry_run=dry_run,
            query=query,
            warnings=["no memories matched the query"],
            message=f"no memories matched the query in {scope_name}",
        )

    if dry_run:
        db.rollback()
    else:
        db.commit()
    warnings: list[str] = []
    extra: dict[str, Any] = {}
    if not soft and not dry_run:
        window, warning = _hard_delete_window()
        extra.update(window)
        warnings.append(warning)
    return ok(
        STATUS_OK if dry_run else live_status, id=None, project=scope_name,
        counts=counts, count=len(bulk_results),
        scopes_affected=[scope_hit], bulk_results=bulk_results,
        dry_run=dry_run, query=query, **extra, warnings=warnings,
        message=(f"{'Would remove' if dry_run else 'Removed'} "
                 f"{counts['memories']} memory/memories matching the query "
                 f"in {scope_name}"),
    )


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
    metadata_filter: Annotated[
        dict[str, Any] | None,
        Field(description="Remove the active memories whose metadata contains "
                          "these key/value pairs (same comparison as "
                          "`recall`). Needs an explicit `project` and cannot "
                          "be combined with ids or `entity`.")] = None,
    contains: Annotated[
        str | None,
        Field(description="Remove the active memories whose content contains "
                          "this literal text (case-insensitive). Needs an "
                          "explicit `project` and cannot be combined with ids "
                          "or `entity`.")] = None,
    query: Annotated[
        str | None,
        Field(description="Remove the active memories that semantically match "
                          "this query. Needs an explicit `project` and cannot "
                          "be combined with ids, `entity`, `metadata_filter` "
                          "or `contains`.")] = None,
    limit: Annotated[
        int,
        Field(description="Maximum number of query matches to remove "
                          "(default 20).")] = 20,
    min_similarity: Annotated[
        float | None,
        Field(description="Minimum cosine similarity for a query match "
                          "(0.0-1.0, default 0.85).")] = 0.85,
) -> dict[str, Any]:
    """Remove memories by id(s), entity, `metadata_filter`, `contains` or `query`.

    `soft=True` (default) hides a row via `deleted_at` and a forgotten sentinel;
    a hard delete marks the row and keeps it recoverable until `undo_until`
    (`purge_after`), after which `prune_memories` removes it. Without `project`
    the id/entity is looked up in every scope, and an entity delete also
    removes its graph node. `entity="*"` empties one project (project
    required).

    An id is unique per scope: an id found in several scopes is refused with the
    scopes named, and an explicit `project` (including "global") bounds the
    removal to that table. `memory_ids` is a bulk form; each id is reported in
    `bulk_results` (status `soft_forgotten`/`deleted`, or `would_*` with
    `dry_run`). `memory_id` and `memory_ids` are mutually exclusive.

    `metadata_filter`, `contains` and `query` remove a matched set and need an
    explicit `project` ("global" for the shared table); they cannot be combined
    with ids, `entity` or each other. `counts` splits
    memories/entities/relations, an entity delete without a project reports
    `scopes_affected`, and a selector that matches nothing answers `ok` with a
    warning.
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
    selector_used = (metadata_filter is not None or contains is not None
                     or query is not None)
    if not memory_id and not memory_ids and not entity_name and not selector_used:
        return fail(STATUS_ERROR, "memory_id, memory_ids, entity, "
                                  "metadata_filter, contains or query is required")
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
    db = MemoryStore()
    hard_extra: dict[str, Any] = {}
    hard_warnings: list[str] = []
    if not soft and not dry_run:
        hard_extra, hard_warning = _hard_delete_window()
        hard_warnings = [hard_warning]
    counts = {"memories": 0, "entities": 0, "relations": 0}
    scopes: list[str] = []
    affected_scopes: list[str] = []
    bulk_results: list[dict] = []
    try:
        cur = db.cur
        if query is not None:
            return _forget_by_query(
                db, cur, project, scope_name, explicit_scope,
                memory_id, memory_ids, entity_name,
                metadata_filter, contains, query, limit, min_similarity,
                soft=soft, dry_run=dry_run,
            )
        if selector_used:
            return _forget_by_selector(
                db, cur, project, scope_name, explicit_scope,
                memory_id, memory_ids, entity_name,
                metadata_filter, contains, soft=soft, dry_run=dry_run,
            )
        # An explicit scope (a project name, or the reserved "global" that
        # _norm_project maps to None) must narrow the lookup to that one
        # table. A plain `if not project` test widened project="global" to
        # every scope and removed copies in unrelated projects.
        limited_tables = None
        if explicit_scope:
            limited_tables = [_mem_table(project)]
        if memory_ids:
            # Bulk forget: process multiple memory ids in one call
            tables = limited_tables or _all_mem_tables(cur)
            if project and not _table_exists(cur, tables[0]):
                return fail(STATUS_NOT_FOUND, "NOT FOUND: project table does not exist")
            
            # A dry run must not look like a completed deletion: the per-id
            # status says "would_*", and each entry names the scope(s) that
            # hold the id. Without the scope a cross-scope count of 2 read
            # like two rows in one project.
            dry_status = "would_soft_forget" if soft else "would_delete"
            live_status = "soft_forgotten" if soft else "deleted"
            for mid in memory_ids:
                if not mid or not isinstance(mid, str):
                    bulk_results.append({"id": mid, "status": "error", "error": "invalid memory id"})
                    continue
                removed, touched = db.forget_rows(
                    mid, tables=tables, soft=soft, dry_run=dry_run)
                if removed:
                    counts["memories"] += removed
                    bulk_results.append({
                        "id": mid,
                        "status": dry_status if dry_run else live_status,
                        "count": removed,
                        "scopes": [_scope_of_table(cur, table)
                                   for table in touched],
                    })
                else:
                    bulk_results.append({"id": mid, "status": "not_found"})
            
            if not dry_run:
                if not soft:
                    clear_dangling_supersede(cur)
                db.commit()
            else:
                db.rollback()
            
            status = STATUS_OK if dry_run else (STATUS_SOFT_FORGOTTEN if soft else STATUS_DELETED)
            return ok(status, counts=counts, dry_run=dry_run, bulk_results=bulk_results,
                      **hard_extra, warnings=hard_warnings,
                      message=f"{'Would remove' if dry_run else ('Soft-forgotten' if soft else 'Hard-deleted')} "
                              f"{counts['memories']} memory/memories from {len(memory_ids)} id(s)")
        if memory_id:
            tables = limited_tables or _all_mem_tables(cur)
            if project and not _table_exists(cur, tables[0]):
                return fail(STATUS_NOT_FOUND, f"NOT FOUND: {memory_id}")
            scopes = db.scopes_of(memory_id)
            if not explicit_scope and len(scopes) > 1:
                # The same id can live in several scopes (an import copies
                # the id). Removing it from all of them at once would
                # delete more than the caller asked for.
                db.rollback()
                return fail(STATUS_ERROR,
                            f"ambiguous memory id {memory_id}: it exists in "
                            f"the scopes {scopes}; pass project=<name> to say "
                            f"which one to remove",
                            scopes=scopes)
            removed, _touched = db.forget_rows(
                memory_id, tables=tables, soft=soft, dry_run=dry_run)
            counts["memories"] += removed
            if not counts["memories"]:
                db.rollback()
                return fail(STATUS_NOT_FOUND, f"NOT FOUND: {memory_id}")
            if not soft and not dry_run:
                clear_dangling_supersede(cur)
        if entity_name:
            graph = GraphStore(cur=cur)
            if purge_all:
                # A whole scope goes: its memories, its graph nodes and the
                # entity/relation mirror rows of that scope.
                table = _mem_table(project)
                if _table_exists(cur, table):
                    # The preview and the real call must agree even about the
                    # rows a soft delete leaves alone: `memories` counts what
                    # this call hides/deletes, `already_forgotten` names the
                    # rows that are already hidden.
                    removed, table_counts = db.forget_scope(
                        project, soft=soft, dry_run=dry_run)
                    counts["memories"] += removed
                    counts["active_memories"] = table_counts["active"]
                    counts["already_forgotten"] = table_counts["forgotten"]
                mirror = graph.scope_counts(project)
                counts["entities"] = mirror["entities"]
                counts["relations"] = mirror["relations"]
                if not dry_run:
                    graph.delete_scope_nodes(project)
                if dry_run:
                    db.rollback()
                else:
                    if not soft:
                        clear_dangling_supersede(cur)
                    db.commit()
                if not (counts["memories"] or counts["entities"]
                        or counts["relations"]):
                    db.rollback()
                    return fail(STATUS_NOT_FOUND,
                                f"no memories in scope {scope_name}")
                status = STATUS_OK if dry_run else (
                    STATUS_SOFT_FORGOTTEN if soft else STATUS_DELETED)
                return ok(status, id=None, entity="*", project=scope_name,
                          counts=counts, dry_run=dry_run,
                          **hard_extra, warnings=hard_warnings,
                          message=(
                              f"{'Would remove' if dry_run else 'Soft-forgotten' if soft else 'Hard-deleted'} "
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
            preview = graph.entity_delete_preview(entity_name, project)
            counts["entities"] = preview["entities"]
            counts["relations"] = preview["relations"]
            affected_scopes = preview["scopes"]
            if not dry_run:
                graph.delete_entity(entity_name, project)

            # Only the selected scope's memories: forgetting an entity in one
            # project must not wipe the global facts of the same entity.
            tables = (_all_mem_tables(cur) if not project
                      else [_mem_table(project)])
            for tbl in dict.fromkeys(tables):
                if not _table_exists(cur, tbl):
                    continue
                memories_before = counts["memories"]
                counts["memories"] += forget_entity_rows(
                    db, tbl, entity_name, soft=soft, dry_run=dry_run)
                counts["memories"] += forget_relation_context_rows(
                    db, tbl, entity_name, soft=soft, dry_run=dry_run)
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
                        hidden += count_entity_rows_incl_context(
                            db, tbl, entity_name)
                if hidden:
                    db.rollback()
                    return ok(
                        STATUS_OK, id=None, entity=entity_name,
                        project=scope_name, counts=counts, dry_run=dry_run,
                        message=(f"nothing to forget: all {hidden} "
                                 f"memory/memories of {entity_name!r} are "
                                 f"already soft-forgotten"))
                db.rollback()
                return fail(STATUS_NOT_FOUND, f"NOT FOUND: {entity_name}")
        if dry_run:
            db.rollback()
        else:
            if entity_name and not soft:
                clear_dangling_supersede(cur)
            db.commit()
    except Exception as e:
        db.rollback()
        return fail(STATUS_ERROR, str(e))
    finally:
        db.close()
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
                f"forget(entity=...) without `project` touched multiple "
                f"scopes ({', '.join(affected_scopes)}); pass project=<name> "
                f"to restrict the removal to one scope")
    if dry_run:
        return ok(STATUS_OK, id=memory_id, entity=entity_name, counts=counts,
                  project=(scope_name if project else None),
                  dry_run=True, **extra, warnings=warnings,
                  message=f"Would remove {target} (nothing was changed)")
    return ok(status, id=memory_id, entity=entity_name, counts=counts,
              project=(scope_name if project else None),
              dry_run=False, **extra, **hard_extra,
              warnings=warnings + hard_warnings,
              message=f"{'Soft-forgotten' if soft else 'Hard-deleted'} {target}")


def merge_memories(
    memory_ids: Annotated[
        list[str],
        Field(description="2-20 ids of active memories to merge into one new "
                          "memory.")],
    content: Annotated[
        str,
        Field(description="Content of the merged memory (non-empty).")],
    project: Annotated[
        str | None,
        Field(description="Scope of the merge. Without it the scope of the "
                          "source ids is used; mixed scopes are refused.")] = None,
    entity: Annotated[
        str | None,
        Field(description="Entity of the merged memory; defaults to the "
                          "sources' shared entity.")] = None,
    importance: Annotated[
        float | None,
        Field(description="Importance of the merged memory (0.0-1.0); defaults "
                          "to the highest importance of the sources.")] = None,
    metadata: Annotated[
        dict[str, Any] | None,
        Field(description="Metadata of the merged memory. With "
                          "`metadata_mode='merge'` (default) it is combined "
                          "with the sources' metadata and these keys win.")] = None,
    metadata_mode: Annotated[
        Literal["merge", "replace"],
        Field(description="'merge' (default) carries the sources' metadata "
                          "(later ids win) and lets explicit keys override "
                          "it; 'replace' stores only `metadata`.")] = "merge",
    mem_type: Annotated[
        str | None,
        Field(description=MEMORY_TYPE_HELP + " Defaults to the sources' shared "
                                             "type, else 'fact'.")] = None,
    dedupe: Annotated[
        Literal["exact", "semantic", "off"],
        Field(description="Duplicate check for the merged content against "
                          "other active memories: 'exact' (default), "
                          "'semantic' or 'off'.")] = "exact",
    fuzzy_threshold: Annotated[
        float | None,
        Field(description="Similarity threshold for `dedupe='semantic'` "
                          "(default 0.85).")] = None,
    dry_run: Annotated[
        bool,
        Field(description="True only reports the merged memory and the "
                          "sources it would supersede; nothing is written.")] = False,
) -> dict[str, Any]:
    """Merge several active memories into one new fact.

    All sources must live in the same scope and be active (not superseded,
    soft-forgotten or expired). The new memory is written first and then every
    source is marked as superseded by it, in one transaction: a failed write
    leaves the sources untouched. `metadata_mode='merge'` (default) combines
    the sources' metadata (later ids win) and lets explicitly passed keys
    override it; graph-edge metadata (`relation_context`, `from`, `to`,
    `relation_type`, `weight`) is never carried onto a real fact.
    Provenance follows the same conservative rule: `confidence` is the highest
    source confidence, `source`/`agent_id` are only carried when every source
    agrees, and `tags` are the union of the sources' tags.
    """
    warnings: list[str] = []
    if not isinstance(memory_ids, list) or not 2 <= len(memory_ids) <= 20:
        return fail(STATUS_ERROR, "memory_ids must contain 2-20 ids")
    if any(not isinstance(mid, str) or not mid.strip() for mid in memory_ids):
        return fail(STATUS_ERROR, "memory_ids must contain non-empty strings")
    ids = list(dict.fromkeys(mid.strip() for mid in memory_ids))
    if len(ids) != len(memory_ids):
        return fail(STATUS_ERROR, "memory_ids must not contain duplicates")
    if not isinstance(content, str) or not content.strip():
        return fail(STATUS_ERROR, "content must be a non-empty string")
    content = content.strip()
    if dedupe not in DEDUPE_MODES:
        return fail(STATUS_ERROR,
                    f"dedupe must be one of {', '.join(sorted(DEDUPE_MODES))}")
    if metadata_mode not in ("merge", "replace"):
        return fail(STATUS_ERROR, "metadata_mode must be 'merge' or 'replace'")
    if importance is not None and not 0.0 <= float(importance) <= 1.0:
        return fail(STATUS_ERROR, "importance must be between 0.0 and 1.0")
    try:
        metadata_obj, metadata_error = (
            (None, None) if metadata is None else _coerce_metadata(metadata))
    except ValueError as e:
        return fail(STATUS_ERROR, str(e))
    if metadata_error:
        return fail(STATUS_ERROR, metadata_error)
    if metadata_mode == "replace":
        merged_meta = dict(metadata_obj or {})
    else:
        merged_meta = {}
    raw_project = project
    explicit_scope = (raw_project is not None
                      and str(raw_project).strip() != "")
    try:
        project = _norm_project(project)
    except ValueError as e:
        return fail(STATUS_ERROR, str(e))

    db = MemoryStore()
    try:
        cur = db.cur
        scopes_by_id = {mid: db.scopes_of(mid) for mid in ids}
        missing = [mid for mid, scopes in scopes_by_id.items() if not scopes]
        if missing:
            db.rollback()
            return fail(STATUS_NOT_FOUND,
                        f"NOT FOUND: {', '.join(missing)}", missing=missing)
        if explicit_scope:
            scope_name = project or "global"
            absent = [mid for mid, scopes in scopes_by_id.items()
                      if scope_name not in scopes]
            if absent:
                db.rollback()
                return fail(
                    STATUS_NOT_FOUND,
                    f"NOT FOUND in scope {scope_name!r}: {', '.join(absent)}",
                    missing=absent)
            if any(len(scopes_by_id[mid]) > 1 for mid in ids):
                warnings.append(
                    "some source ids also exist in another scope; only the "
                    f"copies in {scope_name!r} were merged")
        else:
            all_scopes = {scope for scopes in scopes_by_id.values()
                          for scope in scopes}
            if len(all_scopes) > 1:
                db.rollback()
                return fail(
                    STATUS_ERROR,
                    f"the source ids live in several scopes "
                    f"({', '.join(sorted(all_scopes))}); pass project=<name> "
                    f"to merge the copies of one scope",
                    scopes=sorted(all_scopes))
            scope_name = next(iter(all_scopes))
        # The sources name the scope; a project omitted from the call must not
        # redirect the merge to the shared table (and report every id as
        # missing).
        merge_project = None if scope_name == "global" else scope_name
        table = _mem_table(merge_project)
        if not _table_exists(cur, table):
            db.rollback()
            return fail(STATUS_NOT_FOUND, "the source scope has no memory table")
        rows = db.rows_by_ids(
            ids, table=table,
            columns="id, content, mem_type, entity, importance, metadata, "
                    "superseded_by, deleted_at, expires_at, confidence, "
                    "source, agent_id, tags")
        by_id = {row["id"]: row for row in rows}
        invalid: dict[str, str] = {}
        for mid in ids:
            row = by_id.get(mid)
            if row is None:
                invalid[mid] = "not_found"
            elif row["deleted_at"] is not None:
                invalid[mid] = "soft_forgotten"
            elif (row["superseded_by"] is not None
                  and row["superseded_by"] != SENTINEL_FORGOTTEN):
                invalid[mid] = "superseded"
            elif _is_past(row["expires_at"]):
                invalid[mid] = "expired"
        if invalid:
            db.rollback()
            named = ", ".join(f"{mid} ({reason})"
                              for mid, reason in invalid.items())
            return fail(STATUS_CONFLICT,
                        f"cannot merge inactive memories: {named}",
                        invalid=invalid)
        ordered = [by_id[mid] for mid in ids]

        if not entity:
            entities = {row["entity"] for row in ordered}
            if len(entities) == 1:
                entity = next(iter(entities))
            elif any(entities):
                warnings.append("the sources had different entities; the "
                                "merged memory is untagged")
        if not mem_type:
            types = {row["mem_type"] for row in ordered}
            mem_type = next(iter(types)) if len(types) == 1 else "fact"
        elif mem_type not in MEMORY_TYPES:
            db.rollback()
            return fail(STATUS_ERROR,
                        f"invalid mem_type {mem_type!r}; valid types are "
                        f"{', '.join(MEMORY_TYPES)}")
        if importance is None:
            importance = max(float(row["importance"] or 0.0)
                             for row in ordered)
        if metadata_mode == "merge":
            for row in ordered:
                meta = row["metadata"] or {}
                if isinstance(meta, str):
                    try:
                        meta = json.loads(meta)
                    except json.JSONDecodeError:
                        meta = {}
                for key, value in dict(meta).items():
                    if key not in _GRAPH_METADATA_KEYS:
                        merged_meta[key] = value
            merged_meta.update(metadata_obj or {})
        else:
            merged_meta = dict(metadata_obj or {})

        # Provenance of a merge: a confidence is only an upper bound, so the
        # strongest source wins; a source/agent is only carried when every
        # source agrees, otherwise the merged fact has no single origin.
        known_confidence = [float(row["confidence"]) for row in ordered
                            if row.get("confidence") is not None]
        merged_confidence = (max(known_confidence) if known_confidence
                             else None)
        source_values = {row.get("source") for row in ordered}
        if len(source_values) == 1:
            merged_source = next(iter(source_values))
        else:
            merged_source = None
            if any(source_values):
                warnings.append("the sources had different sources; the "
                                "merged memory has no source")
        agent_values = {row.get("agent_id") for row in ordered}
        if len(agent_values) == 1:
            merged_agent = next(iter(agent_values))
        else:
            merged_agent = None
            if any(agent_values):
                warnings.append("the sources had different agent ids; the "
                                "merged memory has no agent_id")
        merged_tags: list[str] = []
        for row in ordered:
            for tag in row.get("tags") or []:
                if tag not in merged_tags:
                    merged_tags.append(tag)
        if len(merged_tags) > MAX_TAGS:
            db.rollback()
            return fail(STATUS_ERROR,
                        f"the sources carry {len(merged_tags)} distinct tags; "
                        f"a memory accepts at most {MAX_TAGS}")

        emb = embed([content])[0]
        duplicate = _deduplicate(
            cur, content, merge_project, embedding_vec=emb,
            fuzzy_threshold=fuzzy_threshold, dedupe=dedupe)
        if duplicate is not None:
            dup_id, mode, similarity = duplicate
            if dup_id not in ids:
                db.rollback()
                return fail(
                    STATUS_DUPLICATE,
                    f"the merged content already exists as {dup_id}",
                    id=dup_id, duplicate_of=dup_id, duplicate_mode=mode,
                    similarity=similarity)
        if dry_run:
            db.rollback()
            return ok(
                STATUS_OK, id=None, dry_run=True,
                would_create={"content": content, "entity": entity,
                              "mem_type": mem_type,
                              "importance": importance,
                              "project": scope_name,
                              "metadata": merged_meta,
                              "confidence": merged_confidence,
                              "source": merged_source,
                              "agent_id": merged_agent,
                              "tags": merged_tags},
                would_supersede=ids, merged_ids=ids,
                warnings=warnings,
                message=f"Would merge {len(ids)} memories into one")
        new_id = _store_memory(
            cur, content, emb, mem_type, entity, importance, merged_meta,
            merge_project, expires_at=None,
            scope=("global" if scope_name == "global" else "project"),
            source=merged_source, agent_id=merged_agent,
            confidence=merged_confidence, tags=merged_tags)
        db.supersede_many(ids, new_id, table=table)
        db.commit()
        return ok(
            STATUS_MERGED, id=new_id, merged_ids=ids, superseded=ids,
            count=len(ids), project=scope_name, warnings=warnings,
            message=f"Merged {len(ids)} memories into {new_id}")
    except Exception as e:
        db.rollback()
        return fail(STATUS_ERROR, str(e))
    finally:
        db.close()
