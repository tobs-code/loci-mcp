import json
import logging
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

from mcp_server.config import (
    DEDUPE_MODES,
    MEMORY_TYPE_HELP,
    MEMORY_TYPES,
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
from mcp_server.inference_client import embed
from mcp_server.inference_client import nli_configured
from mcp_server.scopes import (
    _all_mem_tables,
    _ensure_project_table,
    _mem_table,
    _memory_scopes,
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
    _find_superseded_match,
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
                                description="Replace the newest memory of the same entity. "
                                            "Only when the new fact is about the same "
                                            "topic (similarity >= 0.5); otherwise the "
                                            "fact is stored additionally.")
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
                          "(default 0.9). Only used by dedupe='semantic'.")] = None,
    dedupe: Annotated[
        Literal["exact", "semantic", "off"],
        Field(description="Duplicate detection: 'exact' (content hash, default), "
                          "'semantic' (also vector similarity) or 'off'.")] = "exact",
    supersede_content: Annotated[
        str | None,
        Field(description="Exact content of the memory this fact replaces.")] = None,
    supersede_last: Annotated[
        bool,
        Field(description="Replace the newest memory of the same entity. Only "
                          "replaces it when the new fact is about the same "
                          "topic (similarity >= 0.5); otherwise the fact is "
                          "stored additionally and a warning is returned.")] = False,
) -> dict[str, Any]:
    """Store a memory (fact, decision, learning, preference, ...).

    Automatically embedded for vector search. Attach to an entity for graph connectivity.
    If `check_contradictions` is true, returns warnings for conflicting existing facts.
    Use `supersede` to replace an existing memory by ID, `supersede_content` to
    replace the fact with the same wording, or `supersede_last` to replace the
    newest fact of the same entity. `supersede_last` refuses to replace a fact
    that is about a different topic (it would destroy it); the new fact is
    stored additionally and the response carries a warning.
    `dedupe` selects how duplicates are detected: "exact" (default, content
    hash), "semantic" (also vector similarity, see `fuzzy_threshold`) or "off".
    A duplicate answer names the mode that matched ("exact" or "semantic") and
    the measured similarity.
    `scope` defaults to "project" when a project is given, else "global".
    scope="global" stores in the shared table even when a project is given.

    Returns the standard envelope: {"ok", "status", "id", "error", "warnings"}.
    """
    logger.info("remember entity=%s mem_type=%s importance=%s check_contradictions=%s fuzzy=%s scope=%s dedupe=%s",
                entity, mem_type, importance, check_contradictions, fuzzy_threshold, scope, dedupe)
    try:
        project = _norm_project(project)
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
            return ok(STATUS_DUPLICATE, id=existing_id,
                      warnings=[f"memory already exists; nothing stored "
                                f"({detail})"],
                      duplicate_mode=mode, similarity=similarity,
                      scope=effective_scope,
                      project=(store_project or "global"))

        # No active duplicate, but the content may match a memory that was
        # superseded earlier. Storing it would silently re-activate an
        # outdated fact, so report the match (without blocking the write).
        superseded_match = _find_superseded_match(
            cur, content, store_project, emb, fuzzy_threshold, dedupe=dedupe)

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

        mem_id = _store_memory(cur, content, emb, mem_type, entity,
                               importance, meta, store_project, None,
                               scope=effective_scope)

        for old_id in superseded:
            _supersede(cur, store_project, mem_id, old_id)

        for c in contradictions:
            eid = c["existing_id"]
            if eid not in superseded:
                _supersede(cur, store_project, mem_id, eid)
                superseded.append(eid)

        # Keep context when a replaced fact is superseded: keys of the old
        # metadata that the new memory does not set are merged into it, so a
        # superseding fact does not silently drop provenance (tickets, ...)
        # on the previous fact. Keys set explicitly by the caller win.
        carried_keys: list[str] = []
        if superseded:
            table = _mem_table(store_project)
            for old_id in superseded:
                cur.execute(
                    f"SELECT metadata FROM {table} "
                    f"WHERE id = %s AND deleted_at IS NULL",
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
        if superseded and not check_contradictions:
            warnings.append(f"superseded {len(superseded)} memory/facts")
        if carried_keys:
            warnings.append(
                "metadata carried over from the replaced memories: "
                + ", ".join(sorted(carried_keys)))
        payload: dict[str, Any] = {"superseded": superseded,
                                   # The scope the fact really landed in. An
                                   # agent that asked for a project but wrote
                                   # to the shared scope has to know that:
                                   # later reads and deletes need that name.
                                   "scope": effective_scope,
                                   "project": (store_project or "global")}
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
        if not entity:
            raise ValueError("supersede_last requires an entity")
        found = _find_last_for_entity(cur, entity, store_project, embedding)
        if found:
            target_id, old_content, similarity = found
            if (similarity is not None
                    and similarity < SUPERSEDE_SIMILARITY_MIN):
                warnings.append(
                    f"supersede_last skipped: no memory of entity {entity!r} "
                    f"is about the same topic (best match {old_content!r} "
                    f"with similarity {similarity} < "
                    f"{SUPERSEDE_SIMILARITY_MIN}); the new fact was stored "
                    f"as an additional memory"
                )
            else:
                targets.append(target_id)
    # Keep order, drop duplicates.
    return list(dict.fromkeys(targets)), warnings


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
) -> dict[str, Any]:
    """Store multiple facts at once. Each dict needs 'content'.

    Optional keys: mem_type (one of the memory types), entity, importance,
    metadata, scope ("project" or "global"), project, supersede (memory id
    or content), supersede_last, fuzzy_threshold, and dedupe ("exact",
    "semantic", or "off").
    Set skip_duplicates=False to allow storing identical content multiple times.
    Set fuzzy_threshold (0.0-1.0) for semantic dedup across all facts.
    Per-fact 'fuzzy_threshold' overrides the top-level value.
    `dedupe` selects how duplicates are detected: "exact" (default, content
    hash), "semantic" (also vector similarity) or "off".
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
        ft = fact.get("fuzzy_threshold", fuzzy_threshold)
        if ft is not None and not 0.0 <= ft <= 1.0:
            prepared.append((None, "fuzzy_threshold must be between 0.0 and 1.0"))
            continue
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
        prepared.append(({
            "content": content,
            "mem_type": mem_type,
            "importance": importance,
            "project": fp if fscope == "project" else None,
            "scope": fscope,
            "entity": fact.get("entity"),
            "fuzzy_threshold": ft,
            "dedupe": fact.get("dedupe"),
            "metadata": meta,
            "supersede": fact.get("supersede"),
            "supersede_content": fact.get("supersede_content"),
            "supersede_last": bool(fact.get("supersede_last")),
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

                if skip_duplicates:
                    fact_dedupe = spec.get("dedupe") or dedupe
                    existing = _deduplicate(cur, spec["content"], store_project,
                                            embeddings[i], spec["fuzzy_threshold"],
                                            dedupe=fact_dedupe)
                    if existing:
                        cur.execute(f"RELEASE SAVEPOINT {savepoint}")
                        results.append({"id": existing[0], "content": spec["content"],
                                        "status": "duplicate",
                                        "duplicate_mode": existing[1],
                                        "similarity": existing[2],
                                        "scope": spec["scope"],
                                        "project": (store_project or "global")})
                        continue

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

                mid = _store_memory(
                    cur, spec["content"], embeddings[i], spec["mem_type"],
                    spec["entity"], spec["importance"], spec["metadata"],
                    store_project, scope=spec["scope"],
                )

                for old_id in superseded:
                    _supersede(cur, store_project, mid, old_id)

                auto_superseded = []
                for c in contradictions:
                    if c["existing_id"] in superseded:
                        continue
                    _supersede(cur, store_project, mid, c["existing_id"])
                    auto_superseded.append(c)
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
) -> dict[str, Any]:
    """Update an existing memory's content, importance, metadata, entity or type.

    Without `project`, the memory is looked up in every scope. A memory id is
    unique per scope, so an id that exists in several scopes (an import copies
    the id) is refused: the response names the scopes, and `project` says
    which one to update.
    """
    logger.info("update_memory id=%s has_content=%s has_importance=%s has_metadata=%s",
                memory_id, content is not None, importance is not None, metadata is not None)
    try:
        project = _norm_project(project)
    except ValueError as e:
        return fail(STATUS_ERROR, str(e))
    sets: list[str] = []
    params: list[Any] = []

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
        entity = (entity or "").strip()
        if not entity:
            return fail(STATUS_ERROR, "entity must not be empty")
        sets.append("entity = %s")
        params.append(entity)
    if mem_type is not None:
        if mem_type not in MEMORY_TYPES:
            return fail(STATUS_ERROR,
                        f"Invalid mem_type. Choose from: "
                        f"{', '.join(sorted(MEMORY_TYPES))}")
        sets.append("mem_type = %s")
        params.append(mem_type)

    if sets:
        # Updated memories must also age like new ones for decay scoring.
        sets.append("updated_at = NOW()")

    conn = get_conn()
    try:
        cur = conn.cursor()
        if project:
            # The project copy and its global twin are updated together.
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
        active = False
        for table in tables:
            cur.execute(
                f"SELECT 1 FROM {table} WHERE id = %s "
                "AND superseded_by IS NULL AND deleted_at IS NULL",
                (memory_id,),
            )
            if cur.fetchone():
                active = True
                break
        if not active:
            conn.rollback()
            return fail(STATUS_NOT_FOUND, f"NOT FOUND: {memory_id}", id=memory_id)
        if not sets:
            conn.rollback()
            return fail(STATUS_ERROR,
                        "nothing to update: pass content, importance, metadata, "
                        "entity or mem_type", id=memory_id)
        updated = 0
        updated_table = None
        for table in tables:
            cur.execute(
                f"UPDATE {table} SET {', '.join(sets)} "
                "WHERE id = %s AND superseded_by IS NULL AND deleted_at IS NULL",
                params + [memory_id],
            )
            if cur.rowcount:
                updated += cur.rowcount
                updated_table = table
                break
        conn.commit()
        if not updated:
            return fail(STATUS_NOT_FOUND, f"NOT FOUND: {memory_id}", id=memory_id)
        scope = _scope_of_table(cur, updated_table) if updated_table else None
    except Exception as e:
        conn.rollback()
        return fail(STATUS_ERROR, str(e), id=memory_id)
    finally:
        conn.close()
    return ok(STATUS_UPDATED, id=memory_id, project=scope)


def restore_memory(
    memory_id: Annotated[
        str,
        Field(description="Id of the soft-forgotten memory to restore.")],
    project: Annotated[
        str | None,
        Field(description="Project scope. Without it the id is looked up in "
                          "every scope.")] = None,
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
    """
    logger.info("restore_memory id=%s project=%s", memory_id, project)
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
    """Remove a fact by memory_id or delete an entity and all its memories.

    The entity can be passed as `entity` or `entity_name`.
    When soft=True (default), hides the memory by setting deleted_at and a
    forgotten sentinel; it remains available only for database-level recovery.
    Without `project`, the id/entity is looked up in every table; deleting an
    entity also removes its memories from all scopes and its graph node.

    A `memory_id` is unique per scope, not globally: if the same id exists in
    several scopes (an import copies the id), the call is refused and the
    scopes are named - pass `project` to say which one to remove.

    `entity="*"` removes every memory of the given project (memories and
    graph nodes); `project` is required for that, so a stray call cannot wipe
    the whole database. `dry_run=True` reports the counts without changing
    anything.
    """
    entity_name = entity or entity_name
    logger.info("forget memory_id=%s entity_name=%s soft=%s dry_run=%s",
                memory_id, entity_name, soft, dry_run)
    raw_project = project
    try:
        project = _norm_project(project)
    except ValueError as e:
        return fail(STATUS_ERROR, str(e))
    if not memory_id and not entity_name:
        return fail(STATUS_ERROR, "memory_id or entity_name is required")
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
    try:
        cur = conn.cursor()
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
    target = memory_id or entity_name
    extra: dict[str, Any] = {}
    if memory_id and scopes:
        extra["scopes"] = scopes
    if dry_run:
        return ok(STATUS_OK, id=memory_id, entity=entity_name, counts=counts,
                  dry_run=True, **extra,
                  message=f"Would remove {target} (nothing was changed)")
    return ok(status, id=memory_id, entity=entity_name, counts=counts,
              dry_run=False, **extra,
              message=f"{'Soft-forgotten' if soft else 'Deleted'} {target}")
