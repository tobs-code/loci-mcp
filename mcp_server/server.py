import logging
import os

from mcp_server.app import app
from mcp_server.backup import export_memories, import_memories
from mcp_server.config import (
    MCP_ALLOW_INSECURE_HTTP,
    MCP_AUTH_TOKEN,
    SENTINEL_FORGOTTEN,
)
from mcp_server.db import ensure_schema, get_conn
from mcp_server.http_auth import BearerAuthMiddleware, resolve_http_auth
from mcp_server.startup_status import record_warning
from mcp_server.graph import (
    find_connections,
    find_paths,
    list_relations,
    relate,
    relate_batch,
    unrelate,
)
from mcp_server.graph_store import GraphStore
from mcp_server.schema_migrator import (
    ensure_fts_language,
    migrate_fts_indexes,
    migrate_idempotency_keys,
)
from mcp_server.tools.maintenance import (
    contradictions_report,
    prune_memories,
    server_health,
    stats_resource,
    verify_integrity,
)
from mcp_server.tools.memory import (
    forget,
    merge_memories,
    remember,
    remember_batch,
    update_batch,
    update_memory,
)
from mcp_server.tools.read import (
    compile_context,
    get_entities,
    get_entity,
    init_context,
    list_entities,
    list_memories,
    recall,
    recall_batch,
)
from mcp_server.tools.scopes import drop_project, list_projects

logger = logging.getLogger(__name__)

_CORE_TOOL_NAMES = frozenset({
    "remember",
    "remember_batch",
    "update_memory",
    "forget",
    "recall",
    "compile_context",
    "list_memories",
    "get_entity",
    "list_entities",
    "list_projects",
    "init_context",
    "relate",
    "unrelate",
    "find_connections",
    "find_paths",
    "list_relations",
})

_ALL_TOOLS = (
    remember,
    remember_batch,
    update_memory,
    update_batch,
    merge_memories,
    forget,
    recall,
    recall_batch,
    compile_context,
    list_memories,
    get_entity,
    get_entities,
    list_entities,
    list_projects,
    init_context,
    relate,
    relate_batch,
    unrelate,
    find_connections,
    find_paths,
    list_relations,
    contradictions_report,
    verify_integrity,
    prune_memories,
    server_health,
    export_memories,
    import_memories,
    drop_project,
)

# `MCP_TOOLSET=core` exposes only the daily read/write/graph surface (16
# tools); `full` (default) exposes all 28. Maintenance and batch tools are
# full-only, which keeps the per-turn schema small for interactive agents.
_toolset = os.environ.get("MCP_TOOLSET", "full").strip().lower() or "full"
if _toolset == "core":
    _TOOLS = tuple(t for t in _ALL_TOOLS if t.__name__ in _CORE_TOOL_NAMES)
elif _toolset == "full":
    _TOOLS = _ALL_TOOLS
else:
    logger.warning("Unknown MCP_TOOLSET=%s; exposing all tools", _toolset)
    _toolset = "full"
    _TOOLS = _ALL_TOOLS

for _tool in _TOOLS:
    app.tool()(_tool)

app.resource("memory://stats", mime_type="application/json",
             description="Memory server stats: counts, health.")(stats_resource)


def migrate_memory_tables() -> None:
    """Migrate the forgotten sentinel and `deleted_at` on every memory table.

    Only project tables in the public schema are ours. An AGE label table with
    the same name can live in another schema (ag_catalog); updating it as if it
    were a memory table fails at startup and hides the real scope.
    """
    project_tables_sql = (
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'public' AND table_name ~ '^mem_[A-Za-z0-9_]+$'"
    )
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("UPDATE memories SET superseded_by = %s WHERE superseded_by = 'forgotten'",
                    (SENTINEL_FORGOTTEN,))
        cur.execute(project_tables_sql)
        for t in cur.fetchall():
            cur.execute(f"UPDATE {t[0]} SET superseded_by = %s WHERE superseded_by = 'forgotten'",
                        (SENTINEL_FORGOTTEN,))
        # Add deleted_at column if missing (migration)
        cur.execute("ALTER TABLE memories ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ")
        cur.execute(project_tables_sql)
        for t in cur.fetchall():
            cur.execute(f"ALTER TABLE {t[0]} ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ")
        # Migrate existing __forgotten__ sentinel to deleted_at
        cur.execute("UPDATE memories SET deleted_at = updated_at WHERE superseded_by = %s AND deleted_at IS NULL",
                    (SENTINEL_FORGOTTEN,))
        cur.execute(project_tables_sql)
        for t in cur.fetchall():
            cur.execute(f"UPDATE {t[0]} SET deleted_at = updated_at WHERE superseded_by = %s AND deleted_at IS NULL",
                        (SENTINEL_FORGOTTEN,))
        conn.commit()
        logger.info("deleted_at column migrated")
    finally:
        conn.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    host = os.environ.get("MCP_HOST", "0.0.0.0")
    port = int(os.environ.get("MCP_PORT", "8766"))

    # Fail before touching the database: an exposed, unauthenticated HTTP
    # transport hands every tool (including drop_project/prune_memories) to
    # anyone who can reach the port.
    auth_token = resolve_http_auth(
        transport, host,
        token=MCP_AUTH_TOKEN,
        allow_insecure=MCP_ALLOW_INSECURE_HTTP,
    )
    if auth_token:
        logger.info("HTTP bearer authentication enabled")
    elif transport in {"sse", "streamable-http"} and MCP_ALLOW_INSECURE_HTTP:
        logger.warning(
            "MCP_ALLOW_INSECURE_HTTP=true: the HTTP transport is running "
            "without authentication - only use this in a trusted sandbox")

    # A dictionary that does not exist would break every FTS query later, so
    # resolve it before the schema/index work builds SQL with it.
    try:
        ensure_fts_language()
    except Exception as e:
        message = f"full-text language check skipped: {e}"
        logger.error(message)
        record_warning(message)

    try:
        ensure_schema()
        logger.info("Schema OK")
    except Exception as e:
        message = f"schema initialization failed: {e}"
        logger.error(message)
        record_warning(message)

    # Full-text indexes must use the configured dictionary; rebuild stale ones.
    try:
        rebuilt = migrate_fts_indexes()
        if rebuilt:
            logger.info("Rebuilt %s full-text index(es) for the current dictionary",
                        rebuilt)
    except Exception as e:
        message = f"full-text index migration skipped: {e}"
        logger.error(message)
        record_warning(message)

    try:
        migrate_memory_tables()
    except Exception as e:
        message = f"memory table migration skipped: {e}"
        logger.error(message)
        record_warning(message)

    # Older project tables only get the idempotency column when a write
    # touches that scope; replay has to work before that first write.
    try:
        changed = migrate_idempotency_keys()
        if changed:
            logger.info("Added idempotency_key to %s memory table(s)", changed)
    except Exception as e:
        message = f"idempotency-key migration skipped: {e}"
        logger.error(message)
        record_warning(message)

    # Deduplicate AGE edges (keep only the first edge per directed pair+type)
    try:
        conn = get_conn()
        cur = conn.cursor()
        graph = GraphStore(cur=cur)
        dup_rows = graph.duplicate_groups()
        for src, tgt, rel_type, cnt in dup_rows:
            for _ in range(int(str(cnt)) - 1):
                graph.delete_duplicate(
                    str(src).strip('"'), str(tgt).strip('"'),
                    str(rel_type).strip('"'))
        conn.commit()
        conn.close()
        if dup_rows:
            logger.info("Deduplicated %s duplicate AGE edge groups", len(dup_rows))
    except Exception as e:
        message = f"AGE edge dedup skipped: {e}"
        logger.error(message)
        record_warning(message)

    if transport == "sse":
        import uvicorn
        asgi_app = _with_auth(app.sse_app(), auth_token)
        logger.info("Starting SSE MCP server on %s:%s", host, port)
        uvicorn.run(asgi_app, host=host, port=port, log_level="info")
    elif transport == "streamable-http":
        import uvicorn
        asgi_app = _with_auth(app.streamable_http_app(), auth_token)
        logger.info("Starting Streamable HTTP MCP server on %s:%s", host, port)
        uvicorn.run(asgi_app, host=host, port=port, log_level="info")
    else:
        logger.info("Starting MCP server (transport=stdio)")
        app.run(transport="stdio")


def _with_auth(asgi_app, token: str | None):
    """Wrap an ASGI app with bearer auth when a token is configured."""
    return BearerAuthMiddleware(asgi_app, token) if token else asgi_app
