"""Project scopes: list them and remove them completely.

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
from mcp_server.scopes import (
    _all_mem_tables,
    _mem_table,
    _norm_project,
    _project_names,
    _project_suffix,
    _scope_label,
    _scope_key,
    _table_exists,
)

logger = logging.getLogger(__name__)


def list_projects(include_empty: Annotated[
        bool,
        Field(description="True (default) also lists projects without active "
                          "memories; False only lists projects that still "
                          "have memories.")] = True) -> dict[str, Any]:
    """List the known project scopes with their sizes.

    Without `init_context(project="*")` an agent has no way to see which
    projects exist. Every row carries the project name, the number of active
    and total memories, the number of graph entities and relations, and
    whether the scope is the shared global scope.
    """
    logger.info("list_projects include_empty=%s", include_empty)
    conn = get_conn()
    try:
        cur = conn.cursor()
        names = _project_names(cur)
        rows: list[dict[str, Any]] = []

        def _scope_stats(tables: list[str]) -> tuple[int, int]:
            total = 0
            active = 0
            for table in tables:
                if not _table_exists(cur, table):
                    continue
                cur.execute(
                    f"SELECT COUNT(*), COUNT(*) FILTER (WHERE superseded_by IS NULL "
                    f"AND deleted_at IS NULL) FROM {table}"
                )
                t, a = cur.fetchone()
                total += t
                active += a
            return total, active

        for table in _all_mem_tables(cur):
            if table == "memories":
                scope = "global"
            else:
                scope = names.get(table[len("mem_"):], table[len("mem_"):])
            total, active = _scope_stats([table])
            if not include_empty and not active:
                continue
            cur.execute(
                "SELECT COUNT(*) FROM entities WHERE project = %s", (scope,)
            )
            entities = cur.fetchone()[0]
            cur.execute(
                "SELECT COUNT(*) FROM relations WHERE source_id IN "
                "(SELECT id FROM entities WHERE project = %s) OR target_id IN "
                "(SELECT id FROM entities WHERE project = %s)",
                (scope, scope),
            )
            relations = cur.fetchone()[0]
            rows.append({
                "project": scope,
                "scope": "global" if scope == "global" else "project",
                "memories": active,
                "total_memories": total,
                "entities": entities,
                "relations": relations,
            })
        conn.commit()
    except Exception as e:
        conn.rollback()
        return fail(STATUS_ERROR, str(e))
    finally:
        conn.close()
    rows.sort(key=lambda r: (r["scope"] != "global", str(r["project"])))
    return ok(STATUS_OK, count=len(rows), results=rows)


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
    """
    logger.info("drop_project project=%s dry_run=%s", project, dry_run)
    name = _norm_project(project)
    if not name:
        return fail(STATUS_ERROR,
                    "project is required: 'global' is the shared scope and "
                    "cannot be dropped")
    table = _mem_table(name)
    suffix = _project_suffix(name)
    label = _scope_label(name)
    scope = _scope_key(name)
    conn = get_conn()
    counts = {"memories": 0, "entities": 0, "relations": 0}
    try:
        cur = conn.cursor()
        if not _table_exists(cur, table):
            return fail(STATUS_NOT_FOUND,
                        f"unknown project {name!r}: it has no memory table")
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
