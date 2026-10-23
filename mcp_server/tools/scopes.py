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

from mcp_server.db import get_conn
from mcp_server.envelope import STATUS_ERROR, STATUS_NOT_FOUND, STATUS_OK, fail, ok
from mcp_server.storage import clear_dangling_supersede
from mcp_server.scopes import (
    _mem_table,
    _norm_project,
    _project_names,
    _project_suffix,
    _project_tables,
    _scope_label,
    _scope_key,
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
    label = _scope_label(name)
    scope = _scope_key(name)
    conn = get_conn()
    counts = {"memories": 0, "entities": 0, "relations": 0}
    try:
        cur = conn.cursor()
        has_table = _table_exists(cur, table)
        if has_table:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            counts["memories"] = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM entities WHERE project = %s", (scope,))
        counts["entities"] = cur.fetchone()[0]
        cur.execute(
            "SELECT COUNT(*) FROM relations WHERE source_id IN "
            "(SELECT id FROM entities WHERE project = %s) OR target_id IN "
            "(SELECT id FROM entities WHERE project = %s)",
            (scope, scope),
        )
        counts["relations"] = cur.fetchone()[0]
        if not has_table and not counts["entities"] and not counts["relations"]:
            return fail(STATUS_NOT_FOUND,
                        f"unknown project {name!r}: it has neither a memory "
                        f"table nor graph nodes")
        if dry_run:
            conn.rollback()
            return ok(STATUS_OK, project=name, counts=counts, dry_run=True,
                      message=(f"Would drop project {name} "
                               f"({counts['memories']} memories, "
                               f"{counts['entities']} entities, "
                               f"{counts['relations']} relations)"))
        # The graph first: the AGE label and the mirror rows belong together.
        cur.execute("SET search_path TO ag_catalog, public")
        cur.execute(
            f"SELECT * FROM cypher('memory_graph', $$ "
            f"MATCH (n:{label}) DETACH DELETE n $$) AS (r agtype)",
        )
        cur.execute(
            "DELETE FROM relations WHERE source_id IN "
            "(SELECT id FROM entities WHERE project = %s) OR target_id IN "
            "(SELECT id FROM entities WHERE project = %s)",
            (scope, scope),
        )
        cur.execute("DELETE FROM entities WHERE project = %s", (scope,))
        # Back to the public schema before the DDL.
        cur.execute("SET search_path TO public")
        cur.execute(f"DROP TABLE IF EXISTS {table}")
        cur.execute("DELETE FROM projects WHERE suffix = %s", (suffix,))
        # A global fact superseded by a fact of this project now points at an
        # id that no longer exists: clear that link so the old fact becomes
        # readable history instead of staying invisible forever.
        clear_dangling_supersede(cur)
        conn.commit()
    except Exception as e:
        conn.rollback()
        return fail(STATUS_ERROR, str(e))
    finally:
        conn.close()
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
                          "relation counts.")] = True,
) -> dict[str, Any]:
    """List every project scope (and the shared global one).

    A project exists as soon as a write created its table (or `relate` created
    graph nodes for it). The shared scope is reported as `global`. This is the
    discovery tool for `recall`, `list_memories` and `init_context`: a caller
    that does not know the project names would otherwise have to guess them
    with `init_context(project='*')`.

    `project` on every row is the value to pass to the other tools (`global`
    for the shared scope). `has_table=False` marks a graph-only scope.
    """
    logger.info("list_projects include_empty=%s", include_empty)
    conn = get_conn()
    try:
        cur = conn.cursor()
        names = _project_names(cur)
        table_scopes = {
            table: (names.get(table[len("mem_"):], table[len("mem_"):]))
            for table in _project_tables(cur)
        }
        cur.execute("SELECT DISTINCT project FROM entities")
        graph_scopes = {row[0] for row in cur.fetchall() if row[0]}
        # Relations have no project column of their own: their scope is the
        # scope of the entities they connect.
        cur.execute(
            "SELECT DISTINCT e.project FROM relations r "
            "JOIN entities e ON e.id = r.source_id")
        graph_scopes |= {row[0] for row in cur.fetchall() if row[0]}

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
            if has_table:
                if include_counts:
                    cur.execute(
                        f"SELECT COUNT(*), "
                        f"COUNT(*) FILTER (WHERE superseded_by IS NULL "
                        f"AND deleted_at IS NULL), "
                        f"COUNT(*) FILTER (WHERE superseded_by IS NOT NULL), "
                        f"COUNT(*) FILTER (WHERE deleted_at IS NOT NULL) "
                        f"FROM {table}")
                    total, active, superseded, forgotten = cur.fetchone()
                    counts.update({
                        "memories": active, "total_memories": total,
                        "superseded_memories": superseded,
                        "forgotten_memories": forgotten,
                    })
                else:
                    # `include_empty=False` still needs to know whether the
                    # scope holds an active memory, so this one count runs
                    # even when the full report is off.
                    cur.execute(
                        f"SELECT COUNT(*) FILTER (WHERE superseded_by IS NULL "
                        f"AND deleted_at IS NULL) FROM {table}")
                    counts["memories"] = cur.fetchone()[0]
                if include_counts:
                    cur.execute(
                        "SELECT COUNT(*) FROM entities WHERE project = %s",
                        (scope,))
                    counts["entities"] = cur.fetchone()[0]
                    cur.execute(
                        "SELECT COUNT(*) FROM relations WHERE source_id IN "
                        "(SELECT id FROM entities WHERE project = %s) OR "
                        "target_id IN (SELECT id FROM entities WHERE project = %s)",
                        (scope, scope))
                    counts["relations"] = cur.fetchone()[0]
            has_graph = scope in graph_scopes
            if include_empty or counts["memories"] > 0 or has_graph:
                rows.append({
                    "project": scope,
                    "scope": "global" if scope == "global" else "project",
                    "has_table": has_table,
                    "has_graph": has_graph,
                    **counts,
                })
        conn.commit()
    except Exception as e:
        conn.rollback()
        return fail(STATUS_ERROR, str(e))
    finally:
        conn.close()
    return ok(STATUS_OK, count=len(rows), results=rows, projects=rows)
