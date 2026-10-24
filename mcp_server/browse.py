"""The one owner of read visibility, cross-scope dedupe and paging.

Every browse-shaped read (`recall`, `list_memories`, `get_entity`,
`list_entities`, `init_context`) used to re-implement parts of the same
pipeline: which scopes to open, which hidden rows to include, how identical
copies collapse, how a page and its counters are produced.

This module owns those decisions:

- ``browse``: the SQL-driven browse path over a memory table (used by
  ``list_memories``).
- ``paginate``: the pager for rows another layer already ranked or assembled
  (``recall``, ``get_entity``, ``list_entities``, ``init_context``).
- ``dedupe_rows``: the single entry point for the cross-scope copy rule.
- ``project_rows``: the projection and ``forgotten``/``expired`` markers of a
  browse page.

Tools keep their ranking, their filters and their envelope. They take the
store/cursor owner from the caller, so a browse never opens a second
connection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mcp_server.config import SENTINEL_FORGOTTEN
from mcp_server.duplicates import dedupe_across_scopes
from mcp_server.lifecycle import is_expired, visibility_sql
from mcp_server.projection import PUBLIC_FIELDS
from mcp_server.projection import project as project_row
from mcp_server.scope import MODE_BROWSE, resolve_scope
from mcp_server.scope_catalog import _norm_entity, _project_names
from mcp_server.search import _extra_conditions

ORDER_COLUMNS = {
    "updated_at": "updated_at",
    "created_at": "created_at",
    "access_count": "access_count",
    "importance": "importance",
}


@dataclass
class BrowseRequest:
    """What a browse reads, independent of how a tool words its answer."""

    project: str | None = None
    include_global: bool = True
    include_deleted: bool = False
    include_expired: bool = False
    include_superseded: bool = False
    include_duplicates: bool = False
    limit: int = 20
    offset: int = 0
    order_by: str = "updated_at"
    entity: str | None = None
    mem_type: str | None = None
    contains: str | None = None
    updated_after: str | None = None
    metadata_filter: dict | None = None
    include_relation_context: bool = True
    fields: list[str] | None = None
    mode: str = MODE_BROWSE


@dataclass
class BrowsePage:
    """A page plus every counter a browse-shaped answer reports."""

    results: list[dict]
    total: int
    has_more: bool
    next_offset: int | None
    scopes_searched: list[str]
    total_by_scope: dict[str, int]
    listed_by_scope: dict[str, int]
    duplicates_hidden: int
    warnings: list[str]
    offset: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Paging:
    """A window over an already ordered row list."""

    window: list[dict]
    total: int
    has_more: bool
    next_offset: int | None


def escape_like(value: str) -> str:
    """Escape the wildcards of a LIKE/ILIKE pattern."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def order_column(order_by: str) -> str | None:
    """The SQL column for a browse order, or None for an unknown order."""
    return ORDER_COLUMNS.get(order_by)


def scope_tables(cur, project: str | None, include_global: bool,
                 mode: str = MODE_BROWSE):
    """Resolve a read scope once and return (selection, tables, scopes)."""
    selection = resolve_scope(project, include_global)
    tables, scopes = tables_and_scopes(cur, selection, mode)
    return selection, tables, scopes


def tables_and_scopes(cur, selection, mode: str = MODE_BROWSE):
    """The tables and readable scope names of an already resolved selection."""
    tables = selection.tables(cur, mode)
    return tables, selection.scopes(cur, tables)


def dedupe_rows(rows: list[dict],
                include_duplicates: bool = False) -> tuple[list[dict], int]:
    """Collapse active copies of identical content across scopes."""
    return dedupe_across_scopes(rows, include_duplicates)


def merge_unique_rows(rows: list[dict],
                      extra: list[dict]) -> tuple[list[dict], int]:
    """Append rows unless their id or content hash is already present.

    Used when one answer merges two sources (an entity's memories plus the
    context sentences of its edges): the same sentence must not appear twice
    under a different role.
    """
    known_ids = {row.get("id") for row in rows}
    known_hashes = {row.get("content_hash") for row in rows
                    if row.get("content_hash")}
    merged = list(rows)
    added = 0
    for row in extra:
        if row.get("id") in known_ids:
            continue
        row_hash = row.get("content_hash")
        if row_hash and row_hash in known_hashes:
            continue
        known_ids.add(row.get("id"))
        if row_hash:
            known_hashes.add(row_hash)
        merged.append(row)
        added += 1
    return merged, added


def paginate(rows: list[dict], limit: int, offset: int,
             total: int | None = None) -> Paging:
    """Cut one page and report whether another one exists."""
    if total is None:
        total = len(rows)
    has_more = total > offset + limit
    return Paging(
        window=rows[offset:offset + limit],
        total=total,
        has_more=has_more,
        next_offset=(offset + limit) if has_more else None,
    )


def project_rows(rows: list[dict], fields: list[str] | None,
                 include_deleted: bool = False,
                 mark_forgotten: bool = True,
                 projection_fields: list[str] | None = None,
                 all_markers: bool = False,
                 stable_markers: bool = False) -> tuple[list[dict], int]:
    """Project rows and mark hidden history; return (items, expired count).

    `fields` is what the caller requested and decides which marker keys
    appear; `projection_fields` overrides the columns actually returned (a
    slim recall still marks an expired row that the caller did not name).
    `all_markers` marks an expired row even when the projection was narrowed
    and no fields were requested. `include_deleted` adds `deleted_at`, so a
    caller can tell a forgotten row from an active one. `stable_markers`
    writes a requested ``forgotten``/``expired`` key even when it is false, so
    a browse answer has the same shape on every row; without it the key only
    appears on hidden rows (recall and get_entity). `mark_forgotten=False`
    keeps the forgotten marker out of an answer whose rows can never be
    soft-deleted (an entity's linked memories).
    """
    if projection_fields is None:
        projection_fields = fields
    if include_deleted:
        if projection_fields is None:
            projection_fields = list(PUBLIC_FIELDS) + ["deleted_at"]
        elif "deleted_at" not in projection_fields:
            projection_fields = projection_fields + ["deleted_at"]
    items: list[dict] = []
    expired_count = 0
    for row in rows:
        item = project_row(row, projection_fields)
        if mark_forgotten and row.get("deleted_at"):
            item["forgotten"] = True
            if item.get("superseded_by") == SENTINEL_FORGOTTEN:
                item["superseded_by"] = None
        elif (stable_markers and mark_forgotten and fields
              and "forgotten" in fields):
            item["forgotten"] = False
        if is_expired(row.get("expires_at")):
            expired_count += 1
            if all_markers or projection_fields is None or (
                    fields and "expired" in fields):
                item["expired"] = True
        elif stable_markers and fields and "expired" in fields:
            item["expired"] = False
        items.append(item)
    return items, expired_count


def visible_where(request: BrowseRequest) -> tuple[str, list[Any], str]:
    """The WHERE clause of browse: visibility flags plus the browse filters.

    Returns ``(where, params, entity)``. The entity is normalized here, so a
    blank name fails with the same argument error as before.
    """
    entity = _norm_entity(request.entity)
    conditions: list[str] = [
        visibility_sql(request.include_deleted, request.include_superseded,
                       request.include_expired)]
    if not request.include_relation_context:
        # The relation-context sentences live under a synthetic entity and
        # repeat what the edge already says.
        conditions.append(
            "COALESCE(metadata->>'relation_context', 'false') <> 'true'")
    params: list[Any] = []
    if entity:
        conditions.append("entity = %s")
        params.append(entity)
    if request.mem_type:
        conditions.append("mem_type = %s")
        params.append(request.mem_type)
    if request.contains:
        conditions.append("content ILIKE %s")
        params.append(f"%{escape_like(request.contains)}%")
    extra_conditions, extra_params = _extra_conditions(
        request.updated_after, request.metadata_filter)
    conditions.extend(extra_conditions)
    params.extend(extra_params)
    return " AND ".join(conditions), params, entity


def _index_rows(cur, tables: list[str], where: str, params: list[Any],
                column: str) -> list[dict]:
    """Light rows of every match: dedupe and paging need the whole set."""
    index_rows: list[dict] = []
    project_names = _project_names(cur)
    for table in tables:
        scope_name = ("global" if table == "memories"
                      else project_names.get(table[4:], table[4:]))
        cur.execute(
            f"SELECT id, content_hash, superseded_by, deleted_at, "
            f"expires_at, {column} AS _order FROM {table} WHERE {where}",
            params,
        )
        for (row_id, c_hash, superseded_by,
             deleted_at, expires_at, order_value) in cur.fetchall():
            index_rows.append({
                "id": row_id,
                "content_hash": c_hash,
                "superseded_by": superseded_by,
                "deleted_at": deleted_at,
                "expires_at": expires_at,
                "_order": order_value,
                "_project": scope_name,
                "_table": table,
            })
    return index_rows


def _sort_index(index_rows: list[dict]) -> None:
    """Newest first, with a stable id tie-break for paging."""
    index_rows.sort(key=lambda r: (r.get("_order") is not None,
                                   r.get("_order"),
                                   str(r.get("id") or "")), reverse=True)


def _count_by_scope(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        scope_name = row.get("_project") or "global"
        counts[scope_name] = counts.get(scope_name, 0) + 1
    return counts


def _load_page(cur, window_index: list[dict]) -> list[dict]:
    """Full rows of the page only, so a browse does not ship every vector."""
    if not window_index:
        return []
    scope_by_key = {(r["_table"], r["id"]): r["_project"]
                    for r in window_index}
    by_table: dict[str, list[str]] = {}
    for row in window_index:
        by_table.setdefault(row["_table"], []).append(row["id"])
    page: dict[tuple[str, str], dict] = {}
    for table, ids in by_table.items():
        cur.execute(f"SELECT * FROM {table} WHERE id = ANY(%s)", (ids,))
        columns = [d[0] for d in cur.description]
        for row in cur.fetchall():
            item = dict(zip(columns, row))
            item["_project"] = scope_by_key[(table, item["id"])]
            page[(table, item["id"])] = item
    return [page[(r["_table"], r["id"])]
            for r in window_index if (r["_table"], r["id"]) in page]


def _filter_names(request: BrowseRequest, entity: str | None) -> list[str]:
    """Names of the browse filters that actually narrowed the result."""
    return [
        name for name, value in (
            ("entity", entity), ("mem_type", request.mem_type),
            ("contains", request.contains),
            ("updated_after", request.updated_after),
            ("metadata_filter", request.metadata_filter),
        ) if value
    ]


def browse(db, request: BrowseRequest) -> BrowsePage:
    """Read one page of stored memories, with counters and warnings.

    The store/cursor owner is the caller's: this function never opens a
    connection and never commits, so the whole page is one unit of work.
    """
    warnings: list[str] = []
    cur = db.cur
    selection, tables, scopes_searched = scope_tables(
        cur, request.project, request.include_global, request.mode)
    unknown = selection.unknown_warning(cur)
    if unknown:
        warnings.append(unknown)
    if (request.mode == MODE_BROWSE and not selection.project
            and not selection.all_projects and len(scopes_searched) > 1):
        # Every other read tool defaults to the global scope. A silent
        # all-scope browse made the same call behave differently, so the
        # wider search is named (no results are hidden, only disclosed).
        warnings.append(
            "no project given: browsed every scope ("
            + ", ".join(scopes_searched)
            + "); pass project=<name> for one scope or project='*' to "
              "make this explicit")
    where, params, entity = visible_where(request)
    column = ORDER_COLUMNS.get(request.order_by)
    if column is None:
        raise ValueError(
            "order_by must be one of updated_at, created_at, access_count, "
            "importance")

    index_rows = _index_rows(cur, tables, where, params, column)
    unique, duplicates_hidden = dedupe_rows(
        index_rows, request.include_duplicates)
    total_by_scope = _count_by_scope(index_rows)
    listed_by_scope = _count_by_scope(unique)
    if duplicates_hidden:
        warnings.append(
            f"{duplicates_hidden} active memory/memories were hidden because "
            f"identical content exists in another scope; only one copy is "
            f"listed. Pass include_duplicates=true to list every copy")
    _sort_index(unique)
    paging = paginate(unique, request.limit, request.offset)
    window = _load_page(cur, paging.window)
    if request.offset > 0 and paging.total > 0 and not window:
        warnings.append(
            f"no results on this page: offset={request.offset} is beyond the "
            f"{paging.total} matching memories; lower the offset")
    results, expired_count = project_rows(
        window, request.fields, request.include_deleted, stable_markers=True)
    if expired_count:
        warnings.append(
            f"{expired_count} result(s) have an expired TTL: they are not "
            f"active memories anymore")
    if paging.total == 0:
        filters = _filter_names(request, entity)
        if filters:
            warnings.append(
                f"no memories matched: the filter(s) "
                f"{', '.join(filters)} narrowed the result to zero; widen or "
                f"remove them")
    return BrowsePage(
        results=results,
        total=paging.total,
        has_more=paging.has_more,
        next_offset=paging.next_offset,
        scopes_searched=scopes_searched,
        total_by_scope=total_by_scope,
        listed_by_scope=listed_by_scope,
        duplicates_hidden=duplicates_hidden,
        warnings=warnings,
        offset=request.offset,
    )
