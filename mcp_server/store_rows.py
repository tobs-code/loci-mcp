"""Row-shaped reads of a memory scope.

`get_entity`, `list_entities` and `init_context` need more than a single
lookup: entity presence in a scope, graph nodes, memory-only names and the
relation-context sentences of an edge. Those queries live here so the tools
describe *what* they read instead of naming a table and a predicate.

The module takes a :class:`~mcp_server.store.MemoryStore` rather than its own
connection: a read still runs inside the caller's transaction.
"""

from __future__ import annotations

from typing import Any

from mcp_server.lifecycle import ACTIVE_ROW_SQL, visibility_sql
from mcp_server.scope_catalog import _project_names

_CONTEXT_FLAG = (
    "COALESCE(metadata->>'relation_context', 'false') <> 'true'"
)

def _scope_name(table: str, names: dict[str, str]) -> str:
    return ("global" if table == "memories"
            else names.get(table[len("mem_"):], table[len("mem_"):]))

def entity_present(store, table: str, entity: str) -> bool:
    """True when the table has any row of this entity."""
    return bool(store.scalar(
        f"SELECT 1 FROM {table} WHERE entity = %s LIMIT 1", (entity,)))

def graph_projects(store, entity: str) -> list[str]:
    """Graph scopes whose entity node carries this name."""
    store.cur.execute("SELECT project FROM entities WHERE name = %s",
                      (entity,))
    return [row[0] for row in store.cur.fetchall()]

def graph_nodes(store, condition: str, params: list[Any]) -> list[tuple]:
    """(name, project, label) of every graph node matching the condition."""
    store.cur.execute(
        f"SELECT name, project, type FROM entities WHERE {condition}",
        params,
    )
    return store.cur.fetchall()

def count_graph_where(store, where: str, params: list[Any]) -> int:
    return store.scalar(
        f"SELECT COUNT(*) FROM entities WHERE {where}", params) or 0

def graph_entities(store, where: str, params: list[Any]) -> list[tuple]:
    """(id, name, label, project) of every matching graph node."""
    store.cur.execute(
        f"SELECT id, name, type, project FROM entities "
        f"WHERE {where} ORDER BY name",
        params,
    )
    return store.cur.fetchall()

def memory_only_count(store, table: str, scope_name: str) -> int:
    """Entities of one table that have no graph node in their scope."""
    return store.scalar(
        f"SELECT COUNT(*) FROM (SELECT entity FROM {table} "
        f"WHERE entity IS NOT NULL AND {ACTIVE_ROW_SQL} "
        f"GROUP BY entity HAVING bool_or({_CONTEXT_FLAG})) e "
        f"WHERE NOT EXISTS (SELECT 1 FROM entities en "
        f"WHERE en.name = e.entity AND en.project = %s)",
        (scope_name,),
    ) or 0

def memory_only_names(store, table: str) -> list[str]:
    """Entity names of one table that are not synthetic edge sentences."""
    store.cur.execute(
        f"SELECT entity FROM {table} "
        f"WHERE entity IS NOT NULL AND {ACTIVE_ROW_SQL} "
        f"GROUP BY entity HAVING bool_or({_CONTEXT_FLAG})"
    )
    return [row[0] for row in store.cur.fetchall()]


def relation_context_rows(store, target: str, tables: list[str],
                          include_superseded: bool,
                          include_expired: bool = False) -> list[dict]:
    """`relate(context=...)` sentences whose edge touches this entity."""
    rows: list[dict] = []
    names = _project_names(store.cur)
    for table in tables:
        conditions = [
            "entity IS NOT NULL",
            "metadata->>'relation_context' = 'true'",
            "(metadata->>'from' = %s OR metadata->>'to' = %s)",
            visibility_sql(include_superseded=include_superseded,
                           include_expired=include_expired),
        ]
        for item in store.all(
                f"SELECT * FROM {table} WHERE {' AND '.join(conditions)}",
                (target, target)):
            item.pop("embedding", None)
            item["_project"] = _scope_name(table, names)
            rows.append(item)
    return rows

def entity_row_count(store, target: str, tables: list[str],
                     include_superseded: bool,
                     include_expired: bool = False,
                     extra_conditions: list[str] | None = None,
                     extra_params: list | None = None) -> tuple[int, int]:
    """Rows `get_entity` would list, plus the copies the dedupe hides.

    `extra_conditions`/`extra_params` carry the optional column filters
    (validity, provenance, tags) so `total` counts exactly the rows the page
    may show.
    """
    from mcp_server.browse import dedupe_rows

    if not tables:
        return 0, 0
    where = ("entity = %s AND " + visibility_sql(
        include_superseded=include_superseded,
        include_expired=include_expired))
    conditions = list(extra_conditions or [])
    params: list = [target] + list(extra_params or [])
    if conditions:
        where += " AND " + " AND ".join(conditions)
    rows: list[dict] = []
    for table in tables:
        for item in store.all(
                f"SELECT id, content_hash, superseded_by, deleted_at, "
                f"expires_at "
                f"FROM {table} WHERE {where}", tuple(params)):
            item["_table"] = table
            rows.append(item)
    unique, hidden = dedupe_rows(rows)
    return len(unique), hidden


def entity_row_counts_by_scope(store, target: str, tables: list[str],
                               scopes: list[str],
                               include_superseded: bool,
                               include_expired: bool = False,
                               include_relation_context: bool = False,
                               extra_conditions: list[str] | None = None,
                               extra_params: list | None = None
                               ) -> dict[str, int]:
    """Exact row count of one entity per scope.

    The page only fetches `offset + limit + 1` rows, so deriving this from
    the returned list made the count depend on `limit`. A real COUNT(*) is
    independent of paging.
    """
    counts: dict[str, int] = {}
    conditions = list(extra_conditions or [])
    params: list = [target] + list(extra_params or [])
    where = ("entity = %s AND " + visibility_sql(
        include_superseded=include_superseded,
        include_expired=include_expired))
    if conditions:
        where += " AND " + " AND ".join(conditions)
    for table, scope_name in zip(tables, scopes):
        count = store.scalar(
            f"SELECT COUNT(*) FROM {table} WHERE {where}", tuple(params)) or 0
        if include_relation_context:
            context_where = (
                "entity IS NOT NULL AND "
                "metadata->>'relation_context' = 'true' AND "
                "(metadata->>'from' = %s OR metadata->>'to' = %s) AND "
            ) + visibility_sql(include_superseded=include_superseded,
                               include_expired=include_expired)
            count += store.scalar(
                f"SELECT COUNT(*) FROM {table} WHERE {context_where}",
                (target, target)) or 0
        counts[scope_name] = count
    return counts
