import logging
from typing import Annotated, Any, Literal, NotRequired, TypedDict

from pydantic import Field

from mcp_server.store import MemoryStore
from mcp_server.envelope import (
    STATUS_DELETED,
    STATUS_ERROR,
    STATUS_NOT_FOUND,
    STATUS_OK,
    fail,
    ok,
)
from mcp_server.graph_store import GraphStore
from mcp_server.scope_catalog import (
    _norm_entity,
    _norm_project,
    _project_exists,
    _retrieval_tables,
    _safe_ident,
    _scope_key,
    _table_exists,
)

logger = logging.getLogger(__name__)


class RelationEdge(TypedDict):
    from_entity: Annotated[str, Field(description="Source entity.")]
    to_entity: Annotated[str, Field(description="Target entity.")]
    relation_type: Annotated[str, Field(description="Relation label.")]
    weight: NotRequired[Annotated[float, Field(
        description="Edge strength in 0.0-1.0 (default 1.0).")]]
    context: NotRequired[Annotated[str | None, Field(
        description="Sentence describing the relation; stored as a memory.")]]
    project: NotRequired[Annotated[str | None, Field(
        description="Project scope for this edge; overrides the tool-level "
                    "project.")]]


def _graph_scope_exists(graph: GraphStore, project: str | None) -> bool:
    """True when the scope has a memory table or graph data of its own."""
    if project is None:
        return True
    if _project_exists(graph.cur, project):
        return True
    graph.cur.execute(
        "SELECT 1 FROM entities WHERE project = %s LIMIT 1", (project,))
    return graph.cur.fetchone() is not None


def _unknown_graph_project_warning(graph: GraphStore,
                                   project: str | None) -> str | None:
    """Warn when a named graph scope does not exist at all.

    A graph read does not fall back to the global graph when a project is
    given, so an empty answer would otherwise look like an empty project.
    """
    if project is None or _graph_scope_exists(graph, project):
        return None
    return (f"project {project!r} does not exist; no relations were searched "
            f"- check the name with init_context(project='*')")


def _entity_not_found_hint(graph: GraphStore, entity_name: str,
                           project: str | None) -> str:
    """Explain why an entity was not found in the searched graph.

    Agents cannot see the scope of the graph they searched, so the message
    names the scope that was used and points to the likely cause.
    """
    scope = _scope_key(project)
    if not _graph_scope_exists(graph, project):
        return (f"project {project!r} does not exist in the graph; check the "
                f"name with init_context(project='*')")
    other_scopes = graph.scopes_with_entity(entity_name)
    memory_only = False
    for table in _retrieval_tables(graph.cur, project):
        if not _table_exists(graph.cur, table):
            continue
        graph.cur.execute(
            f"SELECT 1 FROM {table} WHERE entity = %s "
            "AND superseded_by IS NULL AND deleted_at IS NULL LIMIT 1",
            (entity_name,))
        if graph.cur.fetchone():
            memory_only = True
            break
    if memory_only:
        return (f"entity not found in the {scope} graph: {entity_name} "
                f"(it exists as a memory-only entity; create a relation with "
                f"`relate` to add it to the graph)")
    if other_scopes:
        return (f"entity not found in the {scope} graph: {entity_name} "
                f"(it exists in: {', '.join(sorted(other_scopes))} - pass "
                f"project=<name> to search that graph)")
    if project is None:
        return (f"entity not found in the {scope} graph: {entity_name} "
                f"(without project the global graph is searched; pass "
                f"project=<name> if the entity lives in a project)")
    return (f"entity not found in the {scope} graph: {entity_name} "
            f"(the entity may live in another project; pass that project)")


def relate(
    from_entity: Annotated[
        str,
        Field(description="Source entity; created automatically if unknown.")],
    to_entity: Annotated[
        str,
        Field(description="Target entity; created automatically if unknown.")],
    relation_type: Annotated[
        str,
        Field(description="Relation label, e.g. 'depends_on'. Only letters, "
                          "digits and '_'.")],
    context: Annotated[
        str | None,
        Field(description="Sentence describing the relation; it is stored as a "
                          "memory of the relation.")] = None,
    project: Annotated[
        str | None,
        Field(description="Project scope of the graph. Without it the global "
                          "graph is used.")] = None,
    weight: Annotated[
        float | None,
        Field(description="Strength of the edge in 0.0-1.0. Omit it to keep "
                          "an existing edge's weight (a new edge defaults to "
                          "1.0); passing a value always re-weights the edge. "
                          "It scales the retrieval boost of graph_boost.")] = None,
) -> dict[str, Any]:
    """Create a directed relation between two entities in the AGE graph.

    `weight` is the strength of the edge in 0.0-1.0; it scales the retrieval
    boost that `graph_boost` gives to the connected memories. Repeating a
    call without `weight` is idempotent and preserves the stored weight
    instead of resetting a deliberate 0.2 edge to the default 1.0.
    """
    logger.info("relate %s ->[%s]-> %s weight=%s", from_entity, relation_type, to_entity, weight)
    try:
        project = _norm_project(project)
    except ValueError as e:
        return fail(STATUS_ERROR, str(e))
    if not relation_type:
        return fail(STATUS_ERROR, "relation_type is required")

    db = MemoryStore()
    try:
        results = GraphStore(db).relate([{
            "from_entity": from_entity,
            "to_entity": to_entity,
            "relation_type": relation_type,
            "weight": weight,
            "context": context,
        }], project)
        db.commit()
    except Exception as e:
        db.rollback()
        return fail(STATUS_ERROR, str(e))
    finally:
        db.close()

    edge = results[0]
    warnings: list[str] = []
    if edge.get("self_loop"):
        warnings.append(
            f"self-loop: {edge['from_entity']!r} points to itself via "
            f"{edge['relation_type']!r}; graph traversal cannot leave that "
            "node through this edge")
    return ok(edge["status"], **{
        "from_entity": edge["from_entity"],
        "to_entity": edge["to_entity"],
        "relation_type": edge["relation_type"],
        "weight": edge["weight"],
        "context_memory": edge["context_memory"],
        "context_reused": edge.get("context_reused", False),
        "warnings": warnings,
    })


def relate_batch(
    edges: Annotated[
        list[RelationEdge],
        Field(description="Edges to create/update. Each needs from_entity, "
                          "to_entity and relation_type; weight, context and "
                          "project are optional.")],
    project: Annotated[
        str | None,
        Field(description="Project scope for all edges (a per-edge project "
                          "wins).")] = None,
) -> dict[str, Any]:
    """Create or update multiple directed relations in one transaction.

    Each edge needs: from_entity, to_entity, relation_type.
    Optional: weight (omit to preserve an existing edge's weight; a new edge
    defaults to 1.0), context (stored as a memory), project (overrides the
    tool-level project for that edge).
    Entities are auto-created; existing edges are updated (MERGE), so the
    call is idempotent and both directions of a relation can coexist.
    """
    logger.info("relate_batch count=%s project=%s", len(edges or []), project)
    try:
        project = _norm_project(project)
    except ValueError as e:
        return fail(STATUS_ERROR, str(e))
    if not edges:
        return fail(STATUS_ERROR, "edges must not be empty: pass at least one "
                                  "edge or omit the call")
    # Validate per edge before the all-or-nothing transaction: an invalid
    # edge in a long batch used to abort the whole call with an error that
    # named neither the offending edge nor its position.
    for i, edge in enumerate(edges):
        try:
            if not _norm_entity(edge.get("from_entity")) \
                    or not _norm_entity(edge.get("to_entity")):
                raise ValueError(
                    "each edge needs 'from_entity' and 'to_entity'")
            _safe_ident(edge.get("relation_type") or "", "relation_type")
            if edge.get("weight") is not None:
                GraphStore.validate_weight(edge.get("weight"))
            _norm_project(edge.get("project", project))
        except (TypeError, ValueError) as e:
            return fail(STATUS_ERROR, f"edge {i}: {e}")
    db = MemoryStore()
    try:
        graph = GraphStore(db)
        # Edges may target different scopes; keep the caller's order.
        indexed: dict[Any, list[tuple[int, dict]]] = {}
        for i, edge in enumerate(edges):
            indexed.setdefault(_norm_project(edge.get("project", project)),
                               []).append((i, edge))
        results: list[Any] = [None] * len(edges)
        # Sorted scope order: `_ensure_entity` takes a per-scope advisory lock,
        # so two batches covering the same scopes in opposite insertion order
        # would otherwise deadlock. Results still land at the caller's index.
        for scope, pairs in sorted(indexed.items(),
                                   key=lambda item: item[0] or ""):
            try:
                batch = graph.relate([edge for _, edge in pairs], scope)
            except Exception as e:
                positions = [pos for pos, _ in pairs]
                raise ValueError(
                    f"edge(s) at index {positions} failed: {e}") from e
            for (pos, _), result in zip(pairs, batch):
                results[pos] = result
        db.commit()
    except Exception as e:
        db.rollback()
        logger.exception("relate_batch failed")
        return fail(STATUS_ERROR, str(e))
    finally:
        db.close()
    warnings: list[str] = []
    self_loops = [r for r in results if r.get("self_loop")]
    if self_loops:
        warnings.append(
            f"{len(self_loops)} self-loop edge(s) stored (an entity points to "
            "itself); they are valid but cannot expand a graph traversal")
    return ok(STATUS_OK, count=len(results), results=results,
              warnings=warnings)


def unrelate(
    from_entity: Annotated[
        str,
        Field(description="Source entity of the edge to delete.")],
    to_entity: Annotated[
        str,
        Field(description="Target entity of the edge to delete.")],
    relation_type: Annotated[
        str,
        Field(description="Relation label of the edge to delete.")],
    project: Annotated[
        str | None,
        Field(description="Project scope of the graph.")] = None,
    keep_context: Annotated[
        bool,
        Field(description="Keep the relation-context memory that `relate` "
                          "stored for this edge. By default it is "
                          "soft-forgotten together with the edge, so a "
                          "deleted relation does not keep describing a "
                          "connection that no longer exists.")] = False,
) -> dict[str, Any]:
    """Delete a single directed relation between two entities.

    The context sentence of the edge is soft-forgotten too unless
    `keep_context=True`: it describes an edge that no longer exists.
    """
    logger.info("unrelate %s -[%s]-> %s project=%s",
                from_entity, relation_type, to_entity, project)
    try:
        project = _norm_project(project)
        from_entity = _norm_entity(from_entity)
        to_entity = _norm_entity(to_entity)
    except ValueError as e:
        return fail(STATUS_ERROR, str(e))
    if not from_entity or not to_entity:
        return fail(STATUS_ERROR, "from_entity and to_entity are required")
    if not relation_type:
        return fail(STATUS_ERROR, "relation_type is required")
    try:
        _safe_ident(relation_type, "relation_type")
    except ValueError as e:
        return fail(STATUS_ERROR, str(e))

    db = MemoryStore()
    try:
        result = GraphStore(db).unrelate(
            from_entity, to_entity, relation_type, project,
            keep_context=keep_context)
        db.commit()
    except Exception as e:
        db.rollback()
        return fail(STATUS_ERROR, str(e))
    finally:
        db.close()

    if not result["deleted"]:
        return fail(STATUS_NOT_FOUND,
                    f"no {relation_type} edge from {from_entity} to {to_entity}")
    return ok(STATUS_DELETED, from_entity=from_entity, to_entity=to_entity,
              relation_type=relation_type,
              context_memories_removed=result["context_memories_removed"],
              context_kept=keep_context,
              orphan_nodes_removed=result["orphans_removed"])


def find_connections(
    entity_name: Annotated[
        str | None,
        Field(description="Entity to traverse from. Alias: `entity`.")] = None,
    direction: Annotated[
        Literal["outgoing", "incoming", "both"],
        Field(description="Direction to traverse (default: both). Each result "
                          "reports its own direction.")] = "both",
    relation_type: Annotated[
        str | None,
        Field(description="Only follow edges of this relation type.")] = None,
    limit: Annotated[
        int,
        Field(description="Maximum number of results (default 20).")] = 20,
    offset: Annotated[
        int,
        Field(description="Skip this many connections (default 0). Use it "
                          "with `limit` to page; the answer reports `total` "
                          "and `has_more`.")] = 0,
    project: Annotated[
        str | None,
        Field(description="Project scope of the graph. Without it the global "
                          "graph is traversed.")] = None,
    entity: Annotated[
        str | None,
        Field(description="Entity to traverse from. Alias: `entity_name`.")] = None,
) -> dict[str, Any]:
    """Traverse the graph from an entity to find connected entities.

    The start entity can be passed as `entity` or `entity_name`.
    `direction` is "outgoing", "incoming" or "both"; each result reports its
    own direction. `project` selects that project's graph; without it the
    global graph is traversed. `offset` pages through the connections; the
    answer reports `total` and `has_more`, so a truncated list is never
    mistaken for a complete neighbourhood. Each result carries the edge's
    `context` sentence when one was stored with `relate`.
    """
    if entity and entity_name and entity.strip() != entity_name.strip():
        return fail(
            STATUS_ERROR,
            f"entity and entity_name were both given but differ "
            f"({entity!r} vs {entity_name!r}); pass only one of the aliases")
    entity_name = entity or entity_name
    logger.info("find_connections entity=%s direction=%s relation_type=%s limit=%s project=%s",
                entity_name, direction, relation_type, limit, project)
    try:
        project = _norm_project(project)
        entity_name = _norm_entity(entity_name)
    except ValueError as e:
        return fail(STATUS_ERROR, str(e))
    if not entity_name:
        return fail(STATUS_ERROR, "entity is required")
    if limit < 1:
        return fail(STATUS_ERROR, "limit must be >= 1")
    if offset < 0:
        return fail(STATUS_ERROR, "offset must be >= 0")
    if direction not in ("outgoing", "incoming", "both"):
        return fail(STATUS_ERROR,
                    "direction must be 'outgoing', 'incoming' or 'both'")
    if relation_type:
        try:
            _safe_ident(relation_type, "relation_type")
        except ValueError as e:
            return fail(STATUS_ERROR, str(e))
    db = MemoryStore()
    try:
        graph = GraphStore(db)
        page = graph.connections(entity_name, direction, project,
                                 relation_type, limit, offset)
        if not page["found"]:
            db.rollback()
            return fail(STATUS_NOT_FOUND,
                        _entity_not_found_hint(graph, entity_name, project),
                        entity=entity_name, scope=_scope_key(project))
        total = page["total"]
        results = page["results"]
        db.commit()
    except Exception as e:
        db.rollback()
        return fail(STATUS_ERROR, str(e))
    finally:
        db.close()
    has_more = total > offset + limit
    warnings: list[str] = []
    if offset > 0 and total > 0 and not results:
        warnings.append(
            f"no results on this page: offset={offset} is beyond the "
            f"{total} connection(s); lower the offset")
    return ok(STATUS_OK, count=len(results), total=total, offset=offset,
              has_more=has_more,
              next_offset=(offset + limit) if has_more else None,
              warnings=warnings, results=results)


def list_relations(
    entity: Annotated[
        str | None,
        Field(description="Only relations where this entity is the source or "
                          "the target.")] = None,
    relation_type: Annotated[
        str | None,
        Field(description="Only relations of this type.")] = None,
    project: Annotated[
        str | None,
        Field(description="Graph scope. Without it (or with '*') every scope "
                          "is listed; 'global' lists only the shared graph.")] = None,
    limit: Annotated[
        int,
        Field(description="Maximum number of relations (default 50).")] = 50,
    offset: Annotated[
        int,
        Field(description="Skip this many relations (default 0).")] = 0,
) -> dict[str, Any]:
    """Browse stored graph edges without traversing from one entity.

    `find_connections` answers "what is next to X"; this answers "which
    relations exist here", optionally filtered by one endpoint or a relation
    type. Sorted by weight (strongest first), then by source, target and type.
    """
    logger.info("list_relations entity=%s relation_type=%s project=%s",
                entity, relation_type, project)
    all_scopes = project is None or str(project).strip() == "*"
    try:
        project = None if all_scopes else _norm_project(project)
        entity = _norm_entity(entity) if entity is not None else None
    except ValueError as e:
        return fail(STATUS_ERROR, str(e))
    if relation_type:
        try:
            _safe_ident(relation_type, "relation_type")
        except ValueError as e:
            return fail(STATUS_ERROR, str(e))
    if limit < 1:
        return fail(STATUS_ERROR, "limit must be >= 1")
    if offset < 0:
        return fail(STATUS_ERROR, "offset must be >= 0")

    warnings: list[str] = []
    db = MemoryStore()
    try:
        graph = GraphStore(db)
        unknown = _unknown_graph_project_warning(graph, project)
        if unknown:
            warnings.append(unknown)
        page = graph.relation_page(
            entity=entity, relation_type=relation_type, project=project,
            limit=limit, offset=offset, all_scopes=all_scopes)
        total = page["total"]
        results = page["results"]
        db.commit()
    except Exception as e:
        db.rollback()
        return fail(STATUS_ERROR, str(e))
    finally:
        db.close()

    has_more = total > offset + limit
    if offset > 0 and total > 0 and not results:
        warnings.append(
            f"no results on this page: offset={offset} is beyond the "
            f"{total} relation(s); lower the offset")
    return ok(STATUS_OK, count=len(results), total=total, offset=offset,
              has_more=has_more,
              next_offset=(offset + limit) if has_more else None,
              warnings=warnings, results=results)


def find_paths(from_entity: Annotated[
        str,
        Field(description="Start entity of the path.")],
    to_entity: Annotated[
        str,
        Field(description="End entity of the path.")],
    max_hops: Annotated[
        int,
        Field(description="Maximum number of hops to search (default 5, max "
                          "10).")] = 5,
    project: Annotated[
        str | None,
        Field(description="Project scope of the graph. Without it the global "
                          "graph is used.")] = None,
) -> dict[str, Any]:
    """Find paths between two entities in the graph.

    The canonical collection is ``results``.
    `project` selects that project's graph; without it the global graph is used.
    When no path is found the response says so in `warnings`: an empty
    `results` list then means "no connection", not "nothing looked up".
    """
    logger.info("find_paths from=%s to=%s max_hops=%s project=%s",
                from_entity, to_entity, max_hops, project)
    try:
        project = _norm_project(project)
        from_entity = _norm_entity(from_entity)
        to_entity = _norm_entity(to_entity)
    except ValueError as e:
        return fail(STATUS_ERROR, str(e))
    if not from_entity or not to_entity:
        return fail(STATUS_ERROR, "from_entity and to_entity are required")
    try:
        max_hops = int(max_hops)
    except (TypeError, ValueError):
        return fail(STATUS_ERROR, "max_hops must be an integer")
    if max_hops < 1:
        return fail(STATUS_ERROR, "max_hops must be >= 1")
    warnings: list[str] = []
    if max_hops > 10:
        warnings.append(
            f"max_hops={max_hops} was clamped to 10: longer paths are not "
            "searched")
        max_hops = 10
    db = MemoryStore()
    try:
        graph = GraphStore(db)
        found = graph.paths(from_entity, to_entity, project, max_hops)
        if found["missing"]:
            missing = found["missing"][0]
            db.rollback()
            return fail(STATUS_NOT_FOUND,
                        _entity_not_found_hint(graph, missing, project),
                        entity=missing, scope=found["scope"])
        if from_entity == to_entity:
            db.rollback()
            return ok(STATUS_OK, count=1, warnings=warnings,
                      results=found["results"])
        paths = found["results"]
        scope = found["scope"]
        db.commit()
    except Exception as e:
        db.rollback()
        return fail(STATUS_ERROR, str(e))
    finally:
        db.close()
    if paths:
        shortest_hops = min(path["hops"] for path in paths)
        paths = [path for path in paths if path["hops"] == shortest_hops]
    else:
        warnings.append(
            f"no path found between {from_entity!r} and {to_entity!r} "
            f"(at most {max_hops} hop(s) in the {scope} graph): the two "
            f"entities are not connected")
        return ok(STATUS_OK, count=0, results=[], warnings=warnings)
    return ok(STATUS_OK, count=len(paths), results=paths, warnings=warnings)
