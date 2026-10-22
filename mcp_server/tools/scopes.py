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
    _project_suffix,
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
