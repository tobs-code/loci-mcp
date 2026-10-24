"""Project scopes: remove them completely.

A project is a scope with its own memory table, its own graph label and its
own rows in the entity/relation mirror. `forget(entity="*")` empties a scope
but leaves the table and the registry entry behind, so the project keeps
showing up in every listing. `drop_project` removes the scope completely.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

from pydantic import Field

from mcp_server.store import MemoryStore
from mcp_server.store_maintenance import (
    drop_scope_table,
    entity_count,
    graph_scopes as graph_scopes_of,
    relation_count,
    relation_scopes as relation_scopes_of,
)
from mcp_server.envelope import STATUS_ERROR, STATUS_NOT_FOUND, STATUS_OK, fail, ok
from mcp_server.graph_store import GraphStore
from mcp_server.lifecycle import counts as lifecycle_counts
from mcp_server.storage import clear_dangling_supersede
from mcp_server.scope_catalog import (
    _mem_table,
    _norm_project,
    _project_names,
    _project_suffix,
    _project_tables,
    _table_exists,
)

logger = logging.getLogger(__name__)



def drop_project(
    project: Annotated[
        str,
        Field(description="Project to remove completely. 'global' is reserved "
                          "and cannot be dropped.")],
    dry_run: Annotated[
        bool,
        Field(description="True only reports what would be removed; nothing is "
                          "deleted.")] = False,
) -> dict[str, Any]:
    """Remove a project scope completely.

    Deletes the project's memory table, its registry entry, its graph nodes
    and its rows in the entity/relation mirror. This is permanent: unlike
    `forget(entity="*")`, which only empties a scope, the project disappears
    from every listing.

    Use `dry_run=True` first if you want to see the counts.

    A scope that only exists in the graph (a project in which `relate` was
    used but no fact was stored) has no memory table; it is removed as well,
    otherwise its graph nodes would stay behind as orphans that no listing
    shows and no tool can delete.

    'global' is the shared scope and cannot be dropped. To empty it, use
    `forget(entity="*", project="global")`.
    """
    logger.info("drop_project project=%s dry_run=%s", project, dry_run)
    raw = str(project or "").strip()
    try:
        name = _norm_project(project)
    except ValueError:
        # A blank name answers with the documented "project is required"
        # failure below instead of picking up the raw-value error.
        name = None
    if not name:
        if raw.lower() == "global":
            return fail(STATUS_ERROR,
                        "'global' is the shared scope and cannot be dropped; "
                        "use forget(entity='*', project='global') to empty it")
        return fail(STATUS_ERROR, "project is required")
    table = _mem_table(name)
    suffix = _project_suffix(name)
    db = MemoryStore()
    counts = {"memories": 0, "entities": 0, "relations": 0}
    try:
        cur = db.cur
        has_table = _table_exists(cur, table)
        if has_table:
            table_counts = lifecycle_counts(cur, table)
            counts["memories"] = table_counts["total"]
            # `forget(entity='*')` can only hide the active rows while the
            # project drop removes the table. Naming both numbers makes the
            # two previews comparable instead of looking contradictory.
            counts["active_memories"] = table_counts["active"]
            counts["already_forgotten"] = table_counts["forgotten"]
        graph = GraphStore(cur=cur)
        mirror = graph.scope_counts(project)
        counts["entities"] = mirror["entities"]
        counts["relations"] = mirror["relations"]
        if not has_table and not counts["entities"] and not counts["relations"]:
            return fail(STATUS_NOT_FOUND,
                        f"unknown project {name!r}: it has neither a memory "
                        f"table nor graph nodes")
        if dry_run:
            db.rollback()
            return ok(STATUS_OK, project=name, counts=counts, dry_run=True,
                      message=(f"Would drop project {name} "
                               f"({counts['memories']} memories, "
                               f"{counts['entities']} entities, "
                               f"{counts['relations']} relations)"))
        # The graph first: the AGE label and the mirror rows belong together.
        graph.delete_scope(project)
        # Back to the public schema before the DDL.
        drop_scope_table(db, table, suffix)
        # A global fact superseded by a fact of this project now points at an
        # id that no longer exists: clear that link so the old fact becomes
        # readable history instead of staying invisible forever.
        clear_dangling_supersede(cur)
        db.commit()
    except Exception as e:
        db.rollback()
        return fail(STATUS_ERROR, str(e))
    finally:
        db.close()
    return ok(STATUS_OK, project=name, counts=counts, dry_run=False,
              message=(f"Dropped project {name} "
                       f"({counts['memories']} memories, "
                       f"{counts['entities']} entities, "
                       f"{counts['relations']} relations)"))

def list_projects(
    include_empty: Annotated[
        bool,
        Field(description="True (default) lists every known scope, including "
                          "ones without memories. False lists only scopes with "
                          "active memories or graph data.")] = True,
    include_counts: Annotated[
        bool,
        Field(description="True (default) reports per-scope memory, entity and "
                          "relation counts. False omits the count keys "
                          "entirely instead of returning zero placeholders; "
                          "the active-memory count is still computed "
                          "internally for include_empty.")] = True,
) -> dict[str, Any]:
    """List every project scope (and the shared global one).

    A project exists as soon as a write created its table (or `relate` created
    graph nodes for it). The shared scope is reported as `global`. This is the
    discovery tool for `recall`, `list_memories` and `init_context`: a caller
    that does not know the project names would otherwise have to guess them
    with `init_context(project='*')`.

    `project` on every row is the value to pass to the other tools (`global`
    for the shared scope). `has_table=False` marks a graph-only scope.

    With `include_counts=True`: `memories` counts the ACTIVE rows (what every
    read shows), while `total_memories` counts every row including superseded,
    forgotten and expired history. `superseded_memories`/`forgotten_memories`
    name the history parts, so the numbers decompose without guessing.
    """
    logger.info("list_projects include_empty=%s", include_empty)
    db = MemoryStore()
    try:
        cur = db.cur
        names = _project_names(cur)
        table_scopes = {
            table: (names.get(table[len("mem_"):], table[len("mem_"):]))
            for table in _project_tables(cur)
        }
        graph_scopes = {s for s in graph_scopes_of(db) if s}
        # Relations have no project column of their own: their scope is the
        # scope of the entities they connect.
        graph_scopes |= {s for s in relation_scopes_of(db) if s}

        scopes = ["global"] + sorted(
            set(table_scopes.values()) | {s for s in graph_scopes
                                           if s != "global"},
            key=str.casefold)
        rows: list[dict[str, Any]] = []
        for scope in scopes:
            table = ("memories" if scope == "global"
                     else _mem_table(scope))
            has_table = scope == "global" or table in table_scopes
            counts = {"memories": 0, "total_memories": 0,
                      "superseded_memories": 0, "forgotten_memories": 0,
                      "entities": 0, "relations": 0}
            active_count = 0
            if has_table:
                table_counts = lifecycle_counts(cur, table)
                active_count = table_counts["active"]
                if include_counts:
                    counts.update({
                        "memories": table_counts["active"],
                        "total_memories": table_counts["total"],
                        "superseded_memories": table_counts["superseded"],
                        "forgotten_memories": table_counts["forgotten"],
                    })
                if include_counts:
                    counts["entities"] = entity_count(db, scope)
                    counts["relations"] = relation_count(db, scope)
            has_graph = scope in graph_scopes
            if include_empty or active_count > 0 or has_graph:
                row = {
                    "project": scope,
                    "scope": "global" if scope == "global" else "project",
                    "has_table": has_table,
                    "has_graph": has_graph,
                }
                if include_counts:
                    row.update(counts)
                rows.append(row)
        db.commit()
    except Exception as e:
        db.rollback()
        return fail(STATUS_ERROR, str(e))
    finally:
        db.close()
    return ok(STATUS_OK, count=len(rows), results=rows, projects=rows)
