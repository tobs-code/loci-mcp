import json
import logging
from datetime import datetime
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
from mcp_server.inference_client import embed, nli_configured
from mcp_server.scopes import (
    _all_mem_tables,
    _ensure_project_table,
    _mem_table,
    _memory_scopes,
    _norm_entity,
    _norm_project,
    _scope_key,
    _scope_label,
    _scope_of_table,
    _table_exists,
)
from mcp_server.scoring import _content_hash
from mcp_server.storage import (
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
                            description="Importance in 0.0-1.0 (default 1.0); scales decay weighting.")
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
                                   description="Exact content of the memory this fact replaces.")
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


def _near_duplicate_warning(near_dup: dict) -> str:
    """Warning for a write that dedupe did not match but that still reads
    like a reworded copy of an active memory (remember + remember_batch)."""
    return (
        f"near-duplicate: active memory {near_dup['id']} matches this content "
        f"with similarity {near_dup['similarity']} - the write looks like a "
        f"reworded copy; pass supersede={near_dup['id']} to replace it, or "
        f"dedupe='off' to store without this warning"
    )


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
                          description="Replacement metadata object.")
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
        Field(description="Importance in 0.0-1.0 (default 1.0). Scales the decay "
                          "weight, so low-importance memories fade faster.")] = 1.0,
    metadata: Annotated[
        dict[str, Any] | None,
        Field(description="Free-form JSON object stored with the memory.")] = None,
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
        Field(description="Exact content of the memory this fact replaces.")] = None,
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

    Returns the standard envelope: {"ok", "status", "id", "error", "warnings"}.
    """
    logger.info("remember entity=%s mem_type=%s importance=%s check_contradictions=%s fuzzy=%s scope=%s dedupe=%s",
                entity, mem_type, importance, check_contradictions, fuzzy_threshold, scope, dedupe)
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
    if metadata is not None and not isinstance(metadata, dict):
        return fail(STATUS_ERROR, "metadata must be an object")
    effective_scope = scope or ("project" if project else "global")
    if effective_scope == "project" and not project:
        return fail(STATUS_ERROR, "project required when scope='project'")
    store_project = project if effective_scope == "project" else None

    meta = metadata or {}
    emb = embed([content])[0]
    conn = get_conn()
    try:
        cur = conn.cursor()
        if store_project:
            _ensure_project_table(cur, store_project)

        existing = _deduplicate(cur, content, store_project, emb, fuzzy_threshold,
                                dedupe=dedupe)
        if existing:
            existing_id, mode, similarity = existing
            detail = ("exact content hash" if mode == "exact"
                      else f"semantic match (similarity {similarity})")
            if mode == "exact" and dedupe == "semantic":
                detail += "; the content hash short-circuits the semantic check"
            warnings = [f"memory already exists; nothing stored ({detail})"]
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
                        supersede_content, supersede_last, emb)
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
            return ok(STATUS_DUPLICATE, id=existing_id,
                      warnings=warnings, **payload)

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
                supersede_last, emb
            )
        except ValueError as e:
            conn.rollback()
            return fail(STATUS_ERROR, str(e))

        # The near-duplicate policy turns the warning into an action. `reject`
        # writes nothing at all - the caller explicitly asked not to fork a
        # reworded copy and expects a duplicate answer instead.
        auto_near: dict | None = None
        if near_dup and near_policy == "reject" \
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
                    f"{near_dup['similarity']}; nothing was stored (pass "
                    f"on_near_duplicate='supersede' to replace it or 'warn' "
                    f"to store it anyway)"])
        if near_dup and near_policy == "supersede" \
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

        conn.commit()

        warnings: list[str] = []
        warnings.extend(guard_warnings)
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
                f"instead of storing a reworded copy")
        elif near_dup and near_dup["id"] in superseded:
            # The explicit supersede path already replaced this memory:
            # telling the agent to also pass supersede=<id> would be stale
            # advice, so the near-duplicate is reported as handled.
            warnings.append(
                f"near-duplicate {near_dup['id']} (similarity "
                f"{near_dup['similarity']}) was replaced by the supersede "
                f"target of this write")
        elif near_dup:
            warnings.append(_near_duplicate_warning(near_dup))
        if cross_scope_superseded:
            warnings.append(_cross_scope_warning(cross_scope_superseded))
        if contradictions:
            warnings.append(
                f"superseded {len(contradictions)} contradicting memory/facts")
        elif nli_unavailable:
            warnings.append("contradiction check skipped: NLI provider is not configured")
        elif contradiction_skip:
            warnings.append(f"contradiction check skipped: {contradiction_skip}")
        if check_contradictions and not entity and not nli_unavailable \
                and not contradiction_skip and not contradictions:
            # Independent of the block above: even a clean scan must say that
            # it compared the fact against nearest memories across all
            # entities when no entity was set. Hiding it behind `elif` made
            # the broader comparison silent.
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
        return ok(STATUS_STORED, id=mem_id, warnings=warnings, **payload)
    except Exception as e:
        conn.rollback()
        logger.exception("remember failed")
        return fail(STATUS_ERROR, str(e))
    finally:
        conn.close()


def _supersede_targets(cur, store_project: str | None, entity: str | None,
                       supersede: str | None, supersede_content: str | None,
                       supersede_last: bool,
                       embedding: list[float] | None = None
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
        if found:
            targets.append(found)
        else:
            warnings.append(
                f"supersede_content skipped: no active memory with exactly "
                f"this content ({supersede_content[:80]!r}) in the target "
                f"scope; nothing was replaced"
            )
    if supersede_last:
        if entity:
            found = _find_last_for_entity(cur, entity, store_project, embedding)
        else:
            # No entity: the best match across the whole scope. The
            # similarity guard still decides - a scope-wide fallback must
            # not replace an unrelated memory either.
            found = _find_last_in_scope(cur, store_project, embedding)
            if found:
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
            target_id, old_content, similarity = found
            topic_floor = (SUPERSEDE_SIMILARITY_MIN if entity
                           else max(SUPERSEDE_SIMILARITY_MIN,
                                    NEAR_DUPLICATE_THRESHOLD))
            if similarity is not None and similarity < topic_floor:
                pool = repr(entity) if entity else "the scope"
                warnings.append(
                    f"supersede_last skipped: no memory of "
                    f"{pool}"
                    f" is about the same topic (best match {old_content!r} "
                    f"with similarity {similarity} < {topic_floor}); the new "
                    f"fact was stored as an additional memory"
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
                               embedding: list[float] | None
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
        supersede_last, embedding)
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
        Field(description="Default scope for all facts (per-fact scope wins).")]= None,
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
    metadata, scope ("project" or "global"), project, supersede (memory id
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
            importance = float(fact.get("importance", 1.0))
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
                prepared.append((None, "metadata must be valid JSON"))
                continue
        if not isinstance(meta, dict):
            prepared.append((None, "metadata must be an object"))
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
                if store_project and store_project not in ensured:
                    _ensure_project_table(cur, store_project)
                    ensured.add(store_project)

                near_dup = None
                if skip_duplicates:
                    fact_dedupe = spec.get("dedupe") or dedupe
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
                                        embeddings[i]))
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
                    embeddings[i],
                )
                # Per-fact near-duplicate policy (same semantics as in
                # `remember`): reject stores nothing, supersede replaces.
                fact_near_policy = spec["on_near_duplicate"]
                if near_dup and fact_near_policy == "reject" \
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
                            f"similarity {near_dup['similarity']}; nothing "
                            f"was stored"],
                    })
                    continue
                if near_dup and fact_near_policy == "supersede" \
                        and near_dup["id"] not in superseded:
                    superseded.append(near_dup["id"])

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
                     "project": (store_project or "global")}
            if guard_warnings:
                entry["warnings"] = guard_warnings
            if near_dup and spec["on_near_duplicate"] == "supersede":
                entry.setdefault("warnings", []).append(
                    f"auto-supersede: replaced near-duplicate "
                    f"{near_dup['id']} (similarity {near_dup['similarity']}) "
                    f"instead of storing a reworded copy")
            elif near_dup and near_dup["id"] in superseded:
                # Already replaced through the explicit supersede path.
                pass
            elif near_dup:
                entry.setdefault("warnings", []).append(
                    _near_duplicate_warning(near_dup))
            if cross_scope_superseded:
                entry["cross_scope_superseded"] = cross_scope_superseded
                entry.setdefault("warnings", []).append(
                    _cross_scope_warning(cross_scope_superseded))
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
    if failed and len(failed) == len(results):
        return fail(STATUS_ERROR, "no fact could be stored",
                    count=len(results), results=results)
    status = STATUS_PARTIAL if failed else "ok"
    warnings = [f"{len(failed)} fact(s) failed"] if failed else []
    return ok(status, count=len(results), results=results, warnings=warnings)


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
        dict | None,
        Field(description="Replacement metadata object.")] = None,
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
) -> dict[str, Any]:
    """Update an existing memory's content, importance, metadata, entity, type or TTL.

    Without `project`, the memory is looked up in every scope. A memory id is
    unique per scope, so an id that exists in several scopes (an import copies
    the id) is refused: the response names the scopes, and `project` says
    which one to update.
    `expires_at` sets a new TTL (`clear_expires_at=True` removes it).
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
    expires_value: datetime | None = None
    if expires_at is not None:
        try:
            expires_value = datetime.fromisoformat(
                str(expires_at).replace("Z", "+00:00"))
        except ValueError:
            return fail(STATUS_ERROR,
                        "expires_at must be an ISO 8601 timestamp "
                        "(e.g. '2027-01-15' or '2027-01-15T12:00:00Z')")
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
    if metadata is not None:
        if not isinstance(metadata, dict):
            return fail(STATUS_ERROR, "metadata must be an object")
        sets.append("metadata = %s::jsonb")
        params.append(json.dumps(metadata))
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
            conn.rollback()
            return fail(STATUS_NOT_FOUND, f"NOT FOUND: {memory_id}", id=memory_id)
        if not sets:
            conn.rollback()
            return fail(STATUS_ERROR,
                        "nothing to update: pass content, importance, metadata, "
                        "entity, mem_type or expires_at/clear_expires_at",
                        id=memory_id)
        scope = _scope_of_table(cur, active_table)
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
                    STATUS_DUPLICATE, id=memory_id, duplicate_of=dup_id,
                    duplicate_mode=dup_mode, similarity=dup_similarity,
                    project=scope,
                    warnings=[f"update not applied: the new content would "
                              f"duplicate active memory {dup_id} ({detail}); "
                              f"nothing was changed"])
            near_dup = _find_near_duplicate(
                cur, content, dedupe_project, emb, exclude_id=memory_id)
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
    warnings = ([_near_duplicate_warning(near_dup)] if near_dup else [])
    return ok(STATUS_UPDATED, id=memory_id, project=scope, warnings=warnings)


def update_batch(
    updates: Annotated[
        list[BatchUpdateInput],
        Field(description="Updates to apply. Each entry needs 'memory_id'; "
                          "optional 'content', 'importance', 'metadata', "
                          "'entity', 'mem_type', 'expires_at', "
                          "'clear_expires_at' and 'project'. Invalid "
                          "entries are reported individually and the rest is "
                          "still applied.")],
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
    TTL (`expires_at`/`clear_expires_at`) are applied as given. An unknown id answers `not_found` in its entry, an id
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
            if not isinstance(metadata, dict):
                error = "metadata must be an object"
            else:
                fields["metadata"] = metadata
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
        if error is None and not fields:
            error = ("nothing to update: pass content, importance, metadata, "
                     "entity, mem_type or expires_at/clear_expires_at")
        if error:
            prepared.append((None, error))
        else:
            prepared.append(({"memory_id": memory_id,
                              "project": entry_project,
                              "fields": fields,
                              "dedupe": entry_dedupe,
                              "fuzzy_threshold": entry_fuzzy}, None))

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
                sets: list[str] = []
                params: list[Any] = []
                for name, value in spec["fields"].items():
                    if name == "content":
                        sets += ["content = %s", "content_hash = %s",
                                 "embedding = %s::vector"]
                        params += [value, _content_hash(value), embeddings[i]]
                    elif name == "metadata":
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
                    cur.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                    results.append({"id": memory_id, "status": "not_found",
                                    "error": f"NOT FOUND: {memory_id}"})
                    continue
                scope = _scope_of_table(cur, active_table)
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
                            "id": memory_id, "status": "duplicate",
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
                entry["warnings"] = [_near_duplicate_warning(near_dup)]
            results.append(entry)
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.exception("update_batch failed")
        return fail(STATUS_ERROR, str(e))
    finally:
        conn.close()
    failed = [r for r in results if r.get("status") != STATUS_UPDATED]
    if failed and len(failed) == len(results):
        return fail(STATUS_ERROR, "no memory could be updated",
                    count=len(results), results=results)
    status = STATUS_PARTIAL if failed else STATUS_OK
    warnings = [f"{len(failed)} update(s) failed"] if failed else []
    return ok(status, count=len(results), results=results, warnings=warnings)


def touch_memory(
    memory_id: Annotated[
        str,
        Field(description="Id of the memory to refresh.")],
    project: Annotated[
        str | None,
        Field(description="Project scope. Without it the id is looked up in "
                          "every scope.")] = None,
    importance: Annotated[
        float | None,
        Field(description="Optional new importance in 0.0-1.0; leave unset to "
                          "keep the current value.")] = None,
) -> dict[str, Any]:
    """Refresh a memory: reset its decay age without changing its content.

    Sets `updated_at` to now, so the recency half of the cognitive decay
    starts over: a fact that was just confirmed or re-read stops fading with
    its original age (the half-life is 90 days). Optionally sets
    `importance` in the same call (importance also slows decay).

    `recall` bumps `access_count`, which already slows decay when a fact is
    read. `touch_memory` is for the opposite direction: "this is still true
    and important" without a search. Content, embedding and `access_count`
    are untouched - use `update_memory` to change what the memory says.

    Without `project` the id is looked up in every scope; an id that exists
    in several scopes is refused (pass `project`).
    """
    logger.info("touch_memory id=%s project=%s has_importance=%s",
                memory_id, project, importance is not None)
    memory_id = (memory_id or "").strip()
    if not memory_id:
        return fail(STATUS_ERROR, "memory_id must not be empty")
    try:
        project = _norm_project(project)
    except ValueError as e:
        return fail(STATUS_ERROR, str(e))

    sets: list[str] = []
    params: list[Any] = []
    if importance is not None:
        try:
            importance = float(importance)
        except (TypeError, ValueError):
            return fail(STATUS_ERROR,
                        "importance must be a number between 0.0 and 1.0")
        if not 0.0 <= importance <= 1.0:
            return fail(STATUS_ERROR,
                        "importance must be between 0.0 and 1.0")
        sets.append("importance = %s")
        params.append(importance)
    sets.append("updated_at = NOW()")

    conn = get_conn()
    try:
        cur = conn.cursor()
        if project:
            tables = [_mem_table(project), "memories"]
        else:
            scopes = _memory_scopes(cur, memory_id)
            if len(scopes) > 1:
                conn.rollback()
                return fail(STATUS_ERROR,
                            f"ambiguous memory id {memory_id}: it exists in "
                            f"the scopes {scopes}; pass project=<name> to say "
                            f"which one to touch",
                            id=memory_id, scopes=scopes)
            tables = _all_mem_tables(cur)
        # Only an active memory can be refreshed: a superseded or forgotten
        # row is invisible to reads, so refreshing it would be a no-op an
        # agent cannot verify.
        active = False
        for table in tables:
            cur.execute(
                f"SELECT 1 FROM {table} WHERE id = %s "
                "AND superseded_by IS NULL AND deleted_at IS NULL",
                (memory_id,))
            if cur.fetchone():
                active = True
                break
        if not active:
            conn.rollback()
            return fail(STATUS_NOT_FOUND, f"NOT FOUND: {memory_id}",
                        id=memory_id)
        touched_table = None
        for table in tables:
            cur.execute(
                f"UPDATE {table} SET {', '.join(sets)} "
                "WHERE id = %s AND superseded_by IS NULL "
                "AND deleted_at IS NULL",
                params + [memory_id])
            if cur.rowcount:
                touched_table = table
                break
        if touched_table is None:
            conn.rollback()
            return fail(STATUS_NOT_FOUND, f"NOT FOUND: {memory_id}",
                        id=memory_id)
        scope = _scope_of_table(cur, touched_table)
        cur.execute(f"SELECT updated_at FROM {touched_table} WHERE id = %s",
                    (memory_id,))
        row = cur.fetchone()
        touched_at = row[0].isoformat() if row and row[0] else None
        conn.commit()
    except Exception as e:
        conn.rollback()
        return fail(STATUS_ERROR, str(e), id=memory_id)
    finally:
        conn.close()
    return ok(STATUS_UPDATED, id=memory_id, project=scope,
              updated_at=touched_at)


def restore_memory(
    memory_id: Annotated[
        str | None,
        Field(description="Id of the soft-forgotten memory to restore.")] = None,
    project: Annotated[
        str | None,
        Field(description="Project scope. Without it the id is looked up in "
                          "every scope.")] = None,
    memory_ids: Annotated[
        list[str] | None,
        Field(description="Ids of several soft-forgotten memories to restore "
                          "in one call. Each id is validated individually; "
                          "mutually exclusive with `memory_id`.")] = None,
) -> dict[str, Any]:
    """Bring a soft-forgotten memory back.

    `forget(soft=True)` hides a memory without deleting it. This tool reverses
    that: it clears `deleted_at` (and the forgotten sentinel) so the memory is
    visible to `recall`/`list_memories` again. Only soft-forgotten memories can
    be restored; hard-deleted rows and active memories are reported as
    `not_found`. A memory that was superseded before it was forgotten keeps
    `superseded_by`: the restore clears the forgotten sentinel, never the
    invalidation - it cannot resurrect an old fact next to the one that
    replaced it.

    Without `project` the id is looked up in every scope. An id that exists in
    several scopes is refused; pass `project` to select one.
    Pass `memory_ids` to restore several ids in one call; each entry is
    validated individually and reported in `results`.
    """
    logger.info("restore_memory id=%s ids=%s project=%s", memory_id,
                len(memory_ids) if isinstance(memory_ids, list) else None,
                project)
    if memory_id and memory_ids is not None:
        return fail(STATUS_ERROR,
                    "pass either memory_id or memory_ids, not both")
    if memory_ids is not None:
        if not isinstance(memory_ids, list) or not memory_ids:
            return fail(STATUS_ERROR, "memory_ids must be a non-empty list")
        results = [restore_memory(memory_id=mid, project=project)
                   for mid in memory_ids]
        restored = [r for r in results if r.get("ok")]
        failed = [r for r in results if not r.get("ok")]
        status = STATUS_RESTORED if not failed else STATUS_PARTIAL
        warnings = ([f"{len(failed)} id(s) could not be restored"]
                    if failed else [])
        return ok(status, count=len(results), restored=len(restored),
                  results=results, warnings=warnings)
    memory_id = (memory_id or "").strip()
    if not memory_id:
        return fail(STATUS_ERROR, "memory_id must not be empty")
    try:
        project = _norm_project(project)
    except ValueError as e:
        return fail(STATUS_ERROR, str(e))
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
            return fail(
                STATUS_NOT_FOUND,
                f"NOT FOUND: {memory_id} (not soft-forgotten or does not exist)",
                id=memory_id)
        scope = _scope_of_table(cur, restored_table) if restored_table else None
    except Exception as e:
        conn.rollback()
        return fail(STATUS_ERROR, str(e), id=memory_id)
    finally:
        conn.close()
    return ok(STATUS_RESTORED, id=memory_id, project=scope)


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
        return fail(STATUS_ERROR, "memory_id, memory_ids or entity_name is required")
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
                                 "restore_memory; use drop_project to remove "
                                 "the scope completely)" if soft and not dry_run else "")
                          ))
            # The scope filter must match the delete path exactly: without a
            # project the entity is removed from *every* scope, so the preview
            # has to count every scope as well. Otherwise `dry_run` reports
            # "NOT FOUND" for a graph node that the real call deletes.
            scope_clause = " AND project = %s" if project else ""
            scope_args: tuple[Any, ...] = ((entity_name, scope) if project
                                            else (entity_name,))
            cur.execute(
                f"SELECT id FROM entities WHERE name = %s{scope_clause}",
                scope_args,
            )
            entity_ids = [r[0] for r in cur.fetchall()]
            counts["entities"] = len(entity_ids)
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
            if not (counts["memories"] or counts["entities"] or counts["relations"]):
                conn.rollback()
                return fail(STATUS_NOT_FOUND, f"NOT FOUND: {entity_name}")
        if dry_run:
            conn.rollback()
        else:
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
    if dry_run:
        return ok(STATUS_OK, id=memory_id, entity=entity_name, counts=counts,
                  dry_run=True, **extra,
                  message=f"Would remove {target} (nothing was changed)")
    return ok(status, id=memory_id, entity=entity_name, counts=counts,
              dry_run=False, **extra,
              message=f"{'Soft-forgotten' if soft else 'Deleted'} {target}")
