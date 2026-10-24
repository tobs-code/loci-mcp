"""The one owner of a write's order.

`remember` and `remember_batch` used to spell out the same sequence - dedupe,
supersede targets, near-duplicate policy, contradiction check, insert, links -
with slightly different ordering and warning assembly. This module keeps that
order in one place: an adapter validates its arguments, builds a `WriteIntent`
and renders the returned `WriteOutcome` into the tool's own envelope.

The pipeline never commits and never opens a connection. The caller owns the
transaction (one write, or one savepoint per batch entry) and the wording of
its own warnings.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from mcp_server.config import (
    CONTRADICTION_AUTO_MAX_SUPERSEDES,
    CONTRADICTION_AUTO_MIN_CONFIDENCE,
    MEMORY_TYPES,
    NEAR_DUPLICATE_THRESHOLD,
    SUPERSEDE_SIMILARITY_MIN,
)
from mcp_server.duplicates import (
    content_conflict as _content_conflict,
)
from mcp_server.duplicates import (
    same_topic as _shares_subject,
)
from mcp_server.envelope import STATUS_DUPLICATE, STATUS_STORED
from mcp_server.graph_store import GraphStore
from mcp_server.inference_gateway import embed, nli_configured
from mcp_server.lifecycle import parse_expires_at
from mcp_server.scope_catalog import (
    _all_mem_tables,
    _mem_table,
    _safe_ident,
    _scope_of_table,
)
from mcp_server.scoring import _content_hash
from mcp_server.storage import (
    ACTIVE_ROW_SQL,
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
    _supersede,
)
from mcp_server.store import MemoryStore

logger = logging.getLogger(__name__)


@dataclass
class WriteIntent:
    """Everything one write needs, already validated by its adapter."""

    content: str
    project: str | None = None
    scope: Literal["global", "project"] | None = None
    entity: str | None = None
    mem_type: str | None = None
    importance: float = 0.5
    metadata: dict | None = None
    metadata_mode: Literal["merge", "replace"] = "merge"
    expires_at: str | None = None
    dedupe: Literal["exact", "semantic", "off"] = "exact"
    fuzzy_threshold: float | None = None
    supersede: str | None = None
    supersede_content: str | None = None
    supersede_last: bool = False
    on_near_duplicate: Literal["warn", "supersede", "reject"] = "warn"
    check_contradictions: bool = False
    # What a confirmed contradiction does: `supersede` replaces the old
    # memory (default), `warn` stores the fact and reports the pair without
    # invalidating anything. Only the strongest `max_contradiction_supersedes`
    # pairs at or above `contradiction_min_confidence` are replaced, so one
    # noisy NLI run cannot wipe several correct facts.
    contradiction_mode: Literal["supersede", "warn"] = "supersede"
    contradiction_min_confidence: float = CONTRADICTION_AUTO_MIN_CONFIDENCE
    max_contradiction_supersedes: int = CONTRADICTION_AUTO_MAX_SUPERSEDES
    links: list[dict] | None = None
    # Batch tools pre-embed every fact in one call; passing the vector keeps
    # that batching and its single "embedding failed" answer intact.
    embedding: list[float] | None = None
    # remember_batch allows duplicates unless asked otherwise; remember always
    # stops at a duplicate.
    skip_duplicates: bool = True
    # remember probes the provider before using NLI; remember_batch does not.
    probe_nli_config: bool = True
    # remember reports a revert of a superseded fact; a batch entry does not.
    report_superseded_match: bool = True


@dataclass
class WriteOutcome:
    """What the pipeline did, plus the pieces each adapter renders itself."""

    status: str
    id: str | None
    scope: str
    project: str
    content: str
    warnings: list[str] = field(default_factory=list)
    superseded: list[str] = field(default_factory=list)
    cross_scope_superseded: list[str] = field(default_factory=list)
    cross_scope_duplicates: list[dict] = field(default_factory=list)
    contradictions: dict | None = None
    links: list[dict] = field(default_factory=list)
    duplicate_mode: str | None = None
    duplicate_of: str | None = None
    similarity: float | None = None
    metadata_carried: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    error: str | None = None
    # Rendering details the adapters need but must not recompute.
    created_scope: bool = False
    ttl_warning: str | None = None
    exact_copy: str | None = None
    duplicate_detail: str | None = None
    duplicate_supersede_warnings: list[str] = field(default_factory=list)
    guard_warnings: list[str] = field(default_factory=list)
    link_warnings: list[str] = field(default_factory=list)
    near_duplicate: dict | None = None
    near_conflict: bool = False
    auto_near: bool = False
    auto_superseded: list[dict] = field(default_factory=list)
    superseded_match: dict | None = None
    contradiction_skip: str | None = None
    nli_unavailable: bool = False
    cross_scope_warning: str | None = None
    cross_duplicate_warning: str | None = None
    persist: bool = True


def write_one(db: MemoryStore, intent: WriteIntent) -> WriteOutcome:
    """Store one fact in the shared write order.

    Raises on a database or embedding failure: the caller owns the transaction
    and therefore the rollback (the whole write, or one batch savepoint).
    """
    cur = db.cur
    content = intent.content
    effective_scope = intent.scope or ("project" if intent.project else "global")
    store_project = intent.project if effective_scope == "project" else None
    entity = intent.entity
    meta = dict(intent.metadata or {})
    near_policy = intent.on_near_duplicate

    emb = intent.embedding
    if emb is None:
        emb = embed([content])[0]
    ttl_warning = _ttl_past_warning(intent.expires_at)

    created_scope = db.ensure_scope(store_project)

    exact_copy: str | None = None
    if not intent.skip_duplicates and intent.dedupe != "off":
        # "store it anyway" still has to say that an identical active row
        # already exists. _find_near_duplicate excludes exact hashes, so only
        # this explicit check can report it.
        existing_exact = _deduplicate(
            cur, content, store_project, None, dedupe="exact")
        if existing_exact:
            exact_copy = existing_exact[0]

    existing = None
    if intent.skip_duplicates:
        existing = _deduplicate(cur, content, store_project, emb,
                                intent.fuzzy_threshold, dedupe=intent.dedupe)
    if existing:
        existing_id, mode, similarity = existing
        detail = ("exact content hash" if mode == "exact"
                  else f"semantic match (similarity {similarity})")
        if mode == "exact" and intent.dedupe == "semantic":
            detail += "; the content hash short-circuits the semantic check"
        superseded_now: list[str] = []
        extra_warnings: list[str] = []
        cross_now: list[str] = []
        if intent.supersede or intent.supersede_content or intent.supersede_last:
            # A duplicate must not swallow a supersede instruction: the fact
            # is already stored, but the caller still asked to invalidate a
            # memory. Apply the replacement through the existing id.
            (superseded_now, _self_targets, extra_warnings,
             cross_now) = _apply_duplicate_supersede(
                cur, store_project, existing_id, entity, intent.supersede,
                intent.supersede_content, intent.supersede_last, emb, content)
        link_results, link_warnings = _apply_remember_links(
            cur, intent.links, entity, store_project)
        cross_warning = _cross_scope_warning(cross_now) if cross_now else None
        return WriteOutcome(
            status=STATUS_DUPLICATE,
            id=existing_id,
            scope=effective_scope,
            project=(store_project or "global"),
            content=content,
            warnings=([f"memory already exists; nothing stored ({detail})"]
                      + extra_warnings
                      + ([cross_warning] if cross_warning else [])
                      + link_warnings),
            superseded=superseded_now,
            cross_scope_superseded=cross_now,
            links=link_results,
            duplicate_mode=mode,
            similarity=similarity,
            metadata=meta,
            created_scope=created_scope,
            duplicate_detail=detail,
            duplicate_supersede_warnings=extra_warnings,
            link_warnings=link_warnings,
            cross_scope_warning=cross_warning,
        )

    superseded_match = None
    if intent.report_superseded_match:
        # No active duplicate, but the content may match a memory that was
        # superseded earlier. Storing it would silently re-activate an
        # outdated fact, so report the match (without blocking the write).
        superseded_match = _find_superseded_match(
            cur, content, store_project, emb, intent.fuzzy_threshold,
            dedupe=intent.dedupe)

    if intent.skip_duplicates:
        compute_near = intent.dedupe != "off" or near_policy != "warn"
    else:
        # skip_duplicates=false means "store duplicates anyway", so the
        # near-duplicate warning only runs when a policy needs the match.
        compute_near = near_policy != "warn"
    near_dup = (_find_near_duplicate(cur, content, store_project, emb)
                if compute_near else None)

    contradictions: list[dict] = []
    contradiction_skip: str | None = None
    nli_unavailable = (intent.check_contradictions
                       and intent.probe_nli_config
                       and nli_configured() is False)
    if intent.check_contradictions and not nli_unavailable:
        try:
            contradictions = _check_contradictions(
                cur, content, entity, store_project, emb)
        except Exception as e:
            # A failed NLI must not kill the write: the fact is stored
            # unencumbered, but the skipped check is surfaced loudly.
            logger.warning("contradiction check failed: %s", e)
            contradictions = []
            contradiction_skip = f"NLI unavailable: {e}"

    # Resolve the replaced memories before storing: "supersede_last" must see
    # the state before this fact exists.
    superseded, guard_warnings = _supersede_targets(
        cur, store_project, entity, intent.supersede, intent.supersede_content,
        intent.supersede_last, emb, content)

    mem_type = _resolve_mem_type(cur, intent.mem_type, superseded,
                                 store_project)

    cross_duplicates = [
        dup for dup in _cross_scope_duplicates(cur, content, store_project)
        if dup["id"] not in superseded
    ]

    # The near-duplicate policy turns the warning into an action. `reject`
    # writes nothing at all - the caller explicitly asked not to fork a
    # reworded copy and expects a duplicate answer instead.
    same_subject = near_dup is not None and _same_subject_entity(near_dup, entity)
    near_conflict = bool(
        same_subject and _content_conflict(content, near_dup.get("content") or ""))
    if (near_dup and same_subject and near_policy == "reject"
            and near_dup["id"] not in superseded):
        return WriteOutcome(
            status=STATUS_DUPLICATE,
            id=near_dup["id"],
            scope=effective_scope,
            project=(store_project or "global"),
            content=content,
            superseded=[],
            links=[],
            duplicate_of=near_dup["id"],
            duplicate_mode="near",
            similarity=near_dup["similarity"],
            metadata=meta,
            created_scope=created_scope,
            ttl_warning=ttl_warning,
            near_duplicate=near_dup,
            near_conflict=near_conflict,
            persist=False,
        )

    auto_near: dict | None = None
    if (near_dup and same_subject and near_policy == "supersede"
            and near_dup["id"] not in superseded):
        auto_near = near_dup
        superseded.append(near_dup["id"])

    mem_id = db.insert(content, emb, mem_type, entity,
                       intent.importance, meta, project=store_project,
                       scope=effective_scope, expires_at=intent.expires_at)

    for old_id in superseded:
        db.mark_superseded(old_id, mem_id, project=store_project)

    auto_superseded: list[dict] = []
    contradiction_kept: list[dict] = []
    contradiction_min_confidence = max(
        0.0, min(1.0, intent.contradiction_min_confidence))
    max_contradiction_supersedes = max(
        0, int(intent.max_contradiction_supersedes))
    if contradictions:
        eligible: list[dict] = []
        for c in contradictions:
            confidence = float(c.get("confidence",
                                     c.get("similarity", 0.0)) or 0.0)
            c["confidence"] = round(confidence, 3)
            if c["existing_id"] in superseded:
                continue
            if confidence < contradiction_min_confidence:
                contradiction_kept.append(c)
            else:
                eligible.append(c)
        eligible.sort(key=lambda c: c["confidence"], reverse=True)
        if intent.contradiction_mode == "warn":
            contradiction_kept.extend(eligible)
        else:
            auto_superseded = eligible[:max_contradiction_supersedes]
            contradiction_kept.extend(
                eligible[max_contradiction_supersedes:])
            for c in auto_superseded:
                db.mark_superseded(c["existing_id"], mem_id,
                                   project=store_project)

    # A project write that invalidates a SHARED (global) fact hides that fact
    # in every scope, not only in this project.
    replaced_ids = list(dict.fromkeys(
        superseded + [c["existing_id"] for c in auto_superseded]))
    cross_scope_superseded = (
        [old for old in replaced_ids
         if _scope_of_id(cur, old, store_project) == "global"]
        if store_project else [])

    # Keep context when a replaced fact is superseded: keys of the old
    # metadata that the new memory does not set are merged into it. Keys set
    # explicitly by the caller win.
    carried_keys: list[str] = []
    if replaced_ids and intent.metadata_mode == "merge":
        for old_id in replaced_ids:
            for key, value in _read_metadata(cur, store_project, old_id).items():
                if key not in meta and key not in carried_keys:
                    meta[key] = value
                    carried_keys.append(key)
        if carried_keys:
            cur.execute(
                f"UPDATE {_mem_table(store_project)} "
                "SET metadata = %s::jsonb WHERE id = %s",
                (json.dumps(meta), mem_id))

    # The graph edges belong to the SAME transaction as the memory: one commit
    # stores the fact and its links together, and a savepoint keeps an edge
    # failure from losing the fact.
    link_results, link_warnings = _apply_remember_links(
        cur, intent.links, entity, store_project)

    near_warning: str | None = None
    if auto_near:
        near_warning = (
            f"auto-supersede: replaced near-duplicate "
            f"{near_dup['id']} (similarity {near_dup['similarity']}) "
            f"instead of storing a reworded copy"
            + ("; a number or negation differs, so verify this was an "
               "intended update and not a contradiction"
               if near_conflict else ""))
    elif near_dup and near_dup["id"] not in superseded:
        # A near-duplicate that some other part of this write already replaced
        # (explicit supersede or the contradiction check) needs no extra
        # advice: the `superseded` list already names it.
        near_warning = _near_duplicate_warning(
            near_dup, entity, near_conflict, new_content=content)

    cross_warning = (_cross_scope_warning(cross_scope_superseded)
                     if cross_scope_superseded else None)
    cross_duplicate_warning = (_cross_scope_duplicate_warning(cross_duplicates)
                               if cross_duplicates else None)

    contradictions_payload: dict[str, Any] | None = None
    if intent.check_contradictions:
        contradictions_payload = {
            "found": bool(contradictions),
            # Only what this write really invalidated; `kept` names the pairs
            # that stayed active on purpose (mode=warn, below the confidence
            # floor, or beyond the supersede budget).
            "superseded": auto_superseded,
            "kept": contradiction_kept,
            "mode": intent.contradiction_mode,
            "min_confidence": contradiction_min_confidence,
            "max_supersedes": max_contradiction_supersedes,
        }
        skip = (contradiction_skip
                or ("NLI provider is not configured"
                    if nli_unavailable else None))
        if skip:
            contradictions_payload["skipped"] = skip
        if not entity and not skip:
            # Transparency: the check ran, but beyond the entity tag.
            contradictions_payload["cross_entity"] = True

    warnings: list[str] = []
    warnings.extend(guard_warnings)
    warnings.extend(link_warnings)
    if ttl_warning:
        warnings.append(ttl_warning)
    if near_warning:
        warnings.append(near_warning)
    if cross_warning:
        warnings.append(cross_warning)
    if cross_duplicate_warning:
        warnings.append(cross_duplicate_warning)

    return WriteOutcome(
        status=STATUS_STORED,
        id=mem_id,
        scope=effective_scope,
        project=(store_project or "global"),
        content=content,
        warnings=warnings,
        superseded=superseded,
        cross_scope_superseded=cross_scope_superseded,
        cross_scope_duplicates=cross_duplicates,
        contradictions=contradictions_payload,
        links=link_results,
        metadata_carried=carried_keys,
        metadata=meta,
        created_scope=created_scope,
        ttl_warning=ttl_warning,
        exact_copy=exact_copy,
        guard_warnings=guard_warnings,
        link_warnings=link_warnings,
        near_duplicate=near_dup,
        near_conflict=near_conflict,
        auto_near=auto_near is not None,
        auto_superseded=auto_superseded,
        superseded_match=superseded_match,
        contradiction_skip=contradiction_skip,
        nli_unavailable=nli_unavailable,
        cross_scope_warning=cross_warning,
        cross_duplicate_warning=cross_duplicate_warning,
    )


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
            GraphStore.validate_weight(weight)
        except (TypeError, ValueError) as e:
            errors.append(f"links[{i}]: weight: {e}")
        context = link.get("context")
        if context is not None and not isinstance(context, str):
            errors.append(f"links[{i}]: 'context' must be a string")
    return errors


def _prepare_links(links: list[dict], default_entity: str | None) -> list[dict]:
    """Normalize `links` items into GraphStore edge dicts."""
    edges: list[dict] = []
    for link in links:
        edges.append({
            "from_entity": (_norm_str(link.get("from_entity"))
                            or default_entity),
            "to_entity": link.get("to_entity").strip()
            if isinstance(link.get("to_entity"), str) else "",
            "relation_type": link.get("relation_type"),
            "weight": GraphStore.validate_weight(link.get("weight", 1.0)),
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
        graph = GraphStore(cur=cur)
        edges_added = True
        for edge in _prepare_links(links, default_entity):
            for lr in graph.relate([edge], store_project):
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


def _mem_type_of(cur, memory_id: str,
                 store_project: str | None) -> str | None:
    """The mem_type of one stored memory (project first, then global)."""
    cur.execute(
        f"SELECT mem_type FROM {_mem_table(store_project)} WHERE id = %s",
        (memory_id,))
    row = cur.fetchone()
    if not row and store_project:
        cur.execute("SELECT mem_type FROM memories WHERE id = %s", (memory_id,))
        row = cur.fetchone()
    value = row[0] if row else None
    return value if value in MEMORY_TYPES else None


def _resolve_mem_type(cur, requested: str | None, targets: list[str],
                      store_project: str | None) -> str:
    """Resolve the type of a new fact.

    An explicitly passed type always wins. Without one, the type of the
    replaced memory is carried over: turning a `decision` into a plain
    `fact` just because the caller omitted `mem_type` lost the kind of the
    memory on every update.
    """
    if requested is not None:
        return requested
    for target_id in targets:
        inherited = _mem_type_of(cur, target_id, store_project)
        if inherited:
            return inherited
    return "fact"


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
        if not found and store_project:
            # A global fact is replaced by its exact content from inside a
            # project too (like `supersede=<id>` already does with ids); the
            # replacement hides the global fact in every scope, which the
            # response names. The exact content is the identity of the
            # target, so an entity tag must not block the fallback: the same
            # text without (or with another) tag is still the intended fact.
            c_hash = _content_hash((supersede_content or "").strip())
            found = _by_content_hash(cur, "memories", c_hash, entity)
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


def _same_subject_entity(near_dup: dict, entity: str | None) -> bool:
    """Whether the near-duplicate match and the new fact share an entity.

    Entity tagging is the only reliable subject signal available before an
    NLI call: two memories about staging and production are often nearly
    identical in wording but describe different systems. An untagged write
    and an untagged match count as the same subject (the historical
    behaviour); one tagged and one untagged does not.
    """
    return _norm_str(near_dup.get("entity")) == _norm_str(entity)


def _near_duplicate_warning(near_dup: dict, entity: str | None = None,
                            conflict: bool = False,
                            update: bool = False,
                            new_content: str | None = None) -> str:
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
    if (new_content is not None and near_dup.get("content")
            and not _shares_subject(
                new_content, near_dup["content"], entity,
                near_dup.get("similarity"))):
        # Same entity tag, but the saved fact is about another attribute
        # (primary vs legacy region). The shared template alone is no reason
        # to retire a different subject's fact.
        preview = (near_dup.get("content") or "").strip()
        if len(preview) > 120:
            preview = preview[:117] + "..."
        if update:
            return (
                f"near-duplicate with a different subject: active memory "
                f"{near_dup['id']} matches the updated content with similarity "
                f"{near_dup['similarity']}, but the two share no subject term "
                f"- they describe different subjects/attributes. This update "
                f"did not touch that memory; only supersede it in a separate "
                f"remember call (supersede={near_dup['id']}) if you are sure "
                f"it must be replaced. Matched content: {preview!r}"
            )
        return (
            f"near-duplicate with a different subject: active memory "
            f"{near_dup['id']} matches this content with similarity "
            f"{near_dup['similarity']}, but the two share no subject term - "
            f"they describe different subjects/attributes and were stored as "
            f"separate facts. Do not pass supersede={near_dup['id']} merely "
            f"because the wording is close. Matched content: {preview!r}"
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
        stamp = parse_expires_at(value)
    except (TypeError, ValueError):
        return None
    if stamp <= datetime.now(UTC):
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
