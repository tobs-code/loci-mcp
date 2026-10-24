import json
import logging
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

from mcp_server.config import (
    CONTRADICTION_AUTO_MAX_SUPERSEDES,
    CONTRADICTION_AUTO_MIN_CONFIDENCE,
    DEDUPE_MODES,
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
from mcp_server.graph_store import GraphStore
from mcp_server.inference_gateway import embed
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
from mcp_server.scoring import _content_hash
from mcp_server.storage import (
    _deduplicate,
    _find_near_duplicate,
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
)
from mcp_server.write_pipeline import (
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
    expires_at: Any = Field(default=None, json_schema_extra={
        "type": ["string", "null"]},
        description="Optional ISO 8601 timestamp when this memory stops "
                    "being active (TTL).")
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
                "runbook", "context", "observation", "convention", "command",
                "note"] | None,
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
            embedding=emb,
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
            warnings = [(f"memory already exists; nothing stored "
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
            }
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
            warnings.append(_near_duplicate_warning(
                outcome.near_duplicate, entity, outcome.near_conflict,
                new_content=content,
                explicit_supersede=supersede or supersede_content))
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
            warnings.append(
                f"{len(kept_contradictions)} possible contradiction(s) were "
                f"kept active (contradiction_mode={mode}): {kept_ids}"
                + (" ..." if len(kept_contradictions) > 5 else "")
                + "; verify them and resolve explicitly with "
                  "contradictions_report(resolve='newest') or supersede=<id>")
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
    """Store multiple facts at once. Each dict needs 'content'.

    Optional keys: mem_type (one of the memory types), entity, importance,
    metadata (a JSON object - a JSON-encoded string of an object is accepted),
    scope ("project" or "global"; 'project' requires a project), project,
    supersede (memory id
    or content), supersede_content, supersede_last, on_near_duplicate
    ("warn"/"supersede"/"reject"),
    auto_supersede_near_duplicate, expires_at (ISO 8601 TTL timestamp),
    fuzzy_threshold, dedupe ("exact", "semantic", or "off"),
    contradiction_mode ("supersede"/"warn"), contradiction_min_confidence
    (0.0-1.0), max_contradiction_supersedes (>= 0), and links
    ([{to_entity, relation_type, weight?, context?, from_entity?}] - graph
    edges created with the fact, 'from_entity' defaults to the fact's
    entity); a malformed edge is a per-fact error and an edge runtime failure
    keeps the fact and warns ("links failed: ...").
    A stored entry reports its edges in `links` (same shape as remember) and
    failed edges inside `link_warnings`; a fact that was REJECTED (a
    near-duplicate reject or a duplicate) stores no edges.
    The last three contradiction keys
    override the top-level defaults per fact.
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
        mem_type = fact.get("mem_type")
        if mem_type is not None and mem_type not in MEMORY_TYPES:
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
        flinks = fact.get("links")
        if flinks is not None:
            # A malformed edge is a per-fact argument error, exactly like the
            # single `remember(links=...)` path - never a silently skipped
            # edge list.
            per_link_errors = _validate_links(flinks, fentity)
            if per_link_errors:
                prepared.append((None, "; ".join(per_link_errors)))
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
        fact_mode = fact.get("contradiction_mode") or contradiction_mode
        if fact_mode not in ("supersede", "warn"):
            prepared.append((None,
                             ("contradiction_mode must be 'supersede' or "
                              "'warn'")))
            continue
        fact_min_conf = fact.get(
            "contradiction_min_confidence", contradiction_min_confidence)
        if not isinstance(fact_min_conf, (int, float)) \
                or not 0.0 <= float(fact_min_conf) <= 1.0:
            prepared.append((None,
                             ("contradiction_min_confidence must be between "
                              "0.0 and 1.0")))
            continue
        fact_max_supersedes = fact.get(
            "max_contradiction_supersedes", max_contradiction_supersedes)
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
            "contradiction_mode": fact_mode,
            "contradiction_min_confidence": float(fact_min_conf),
            "max_contradiction_supersedes": fact_max_supersedes,
            "expires_at": fexpires,
            "ttl_warning": _ttl_past_warning(fexpires),
            "links": flinks,
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
                                "error": err})
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
                    ))
                    if not outcome.persist:
                        raise _EntryRejected(outcome)
            except _EntryRejected as rejected:
                outcome = rejected.outcome
            except Exception as exc:
                results.append({"id": None, "content": spec["content"],
                                "status": "error", "error": str(exc),
                                "scope": spec["scope"],
                                "project": (store_project or "global")})
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
                if warnings:
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
                entry["warnings"].append(_near_duplicate_warning(
                    outcome.near_duplicate, spec["entity"],
                    outcome.near_conflict, new_content=spec["content"]))
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
                        f"active; verify them and resolve explicitly with "
                        f"contradictions_report(resolve='newest') or "
                        f"supersede=<id>")
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
    An expired row is inactive and answers `conflict` like a superseded one;
    revive it deliberately with a future `expires_at` or `clear_expires_at`.
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
        if sets and "updated_at = NOW()" not in sets:
            # A metadata-only edit must age like every other update.
            sets.append("updated_at = NOW()")
        if not sets:
            db.rollback()
            return fail(STATUS_ERROR,
                        "nothing to update: pass content, importance, metadata, "
                        "clear_metadata_keys, entity, mem_type, "
                        "expires_at/clear_expires_at, touch=true or "
                        "restore=true")
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
    TTL (`expires_at`/`clear_expires_at`) are applied as given. An expired
    row answers `conflict` unless the entry revives it with a future
    `expires_at` or `clear_expires_at`. `touch=true`
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
                fields["expires_at"] = parse_expires_at(entry_expires)
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

    db = MemoryStore()
    results: list[dict] = []
    try:
        cur = db.cur
        for i, (spec, err) in enumerate(prepared):
            if spec is None:
                raw = updates[i]
                raw_id = raw.get("memory_id") if isinstance(raw, dict) else None
                results.append({"id": raw_id, "status": "error", "error": err})
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
                            db.rollback_to(savepoint)
                            results.append({
                                "id": memory_id, "status": "not_found",
                                "error": f"NOT FOUND: {memory_id}"})
                            continue
                    else:
                        scopes = _memory_scopes(cur, memory_id)
                        if len(scopes) > 1:
                            db.rollback_to(savepoint)
                            results.append({
                                "id": memory_id, "status": "error",
                                "error": (f"ambiguous memory id {memory_id}: "
                                          f"it exists in the scopes {scopes}; "
                                          f"pass project=<name> to say which "
                                          f"one to restore"),
                                "scopes": scopes})
                            continue
                        restore_tables = _all_mem_tables(cur)
                    restored_table, restored = restore_rows(
                        db, memory_id, restore_tables)
                    if not restored:
                        blocked = _restore_blocked_reason(
                            db, memory_id, restore_tables)
                        db.rollback_to(savepoint)
                        results.append({
                            "id": memory_id,
                            "status": blocked[0] if blocked else "not_found",
                            "error": (blocked[1] if blocked
                                      else f"NOT FOUND: {memory_id}")})
                        continue
                    scope = _scope_of_table(cur, restored_table)
                    db.release(savepoint)
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
                        db.rollback_to(savepoint)
                        results.append({
                            "id": memory_id, "status": "error",
                            "error": (f"ambiguous memory id {memory_id}: it "
                                      f"exists in the scopes {scopes}; pass "
                                      f"project=<name> to say which one to "
                                      f"update"),
                            "scopes": scopes})
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
                    db.rollback_to(savepoint)
                    error = (f"memory {memory_id} exists but cannot be "
                             f"updated: {reason}" if reason
                             else f"NOT FOUND: {memory_id}")
                    results.append({
                        "id": memory_id,
                        "status": "conflict" if reason else "not_found",
                        "error": error})
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
                    db.rollback_to(savepoint)
                    results.append({"id": memory_id, "status": "not_found",
                                    "error": f"NOT FOUND: {memory_id}"})
                    continue
                db.release(savepoint)
            except Exception as exc:
                db.rollback_to(savepoint)
                results.append({"id": memory_id, "status": "error",
                                "error": str(exc)})
                continue
            entry: dict[str, Any] = {"id": memory_id,
                                     "status": STATUS_UPDATED,
                                     "project": scope}
            if near_dup:
                entry["warnings"] = [
                    _near_duplicate_warning(
                        near_dup, near_entity, near_conflict, update=True,
                        new_content=spec["fields"]["content"])]
            if spec.get("ttl_warning"):
                entry.setdefault("warnings", []).append(spec["ttl_warning"])
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
    not; a found id carries the scope(s) it was removed from, and with
    `dry_run=True` its status is `would_soft_forget`/`would_delete` instead
    of `soft_forgotten`/`deleted`. `memory_id` and `memory_ids` are mutually
    exclusive.

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
    db = MemoryStore()
    counts = {"memories": 0, "entities": 0, "relations": 0}
    scopes: list[str] = []
    affected_scopes: list[str] = []
    bulk_results: list[dict] = []
    try:
        cur = db.cur
        if memory_ids:
            # Bulk forget: process multiple memory ids in one call
            tables = _all_mem_tables(cur) if not project else [_mem_table(project)]
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
                      message=f"{'Would remove' if dry_run else ('Soft-forgotten' if soft else 'Deleted')} "
                              f"{counts['memories']} memory/memories from {len(memory_ids)} id(s)")
        if memory_id:
            tables = _all_mem_tables(cur) if not project else [_mem_table(project)]
            if project and not _table_exists(cur, tables[0]):
                return fail(STATUS_NOT_FOUND, f"NOT FOUND: {memory_id}")
            scopes = db.scopes_of(memory_id)
            if not project and len(scopes) > 1:
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
              dry_run=False, **extra, warnings=warnings,
              message=f"{'Soft-forgotten' if soft else 'Deleted'} {target}")
