import json
import logging
import os

from mcp_server.app import app
from mcp_server.backup import export_memories, import_memories
from mcp_server.config import SENTINEL_FORGOTTEN
from mcp_server.db import ensure_schema, get_conn
from mcp_server.graph import (
    find_connections,
    find_paths,
    relate,
    relate_batch,
    unrelate,
)
from mcp_server.scopes import _migrate_fts_indexes
from mcp_server.tools.maintenance import (
    contradictions_report,
    server_health,
    stats_resource,
    verify_integrity,
)
from mcp_server.tools.memory import (
    forget,
    remember,
    remember_batch,
    update_batch,
    update_memory,
)
from mcp_server.tools.read import (
    get_entity,
    init_context,
    list_entities,
    list_memories,
    recall,
)
from mcp_server.tools.scopes import drop_project

logger = logging.getLogger(__name__)

_CORE_TOOL_NAMES = frozenset({
    "remember",
    "remember_batch",
    "update_memory",
    "forget",
    "recall",
    "list_memories",
    "get_entity",
    "list_entities",
    "init_context",
    "relate",
    "unrelate",
    "find_connections",
    "find_paths",
})

_ALL_TOOLS = (
    remember,
    remember_batch,
    update_memory,
    update_batch,
    forget,
    recall,
    list_memories,
    get_entity,
    list_entities,
    init_context,
    relate,
    relate_batch,
    unrelate,
    find_connections,
    find_paths,
    contradictions_report,
    verify_integrity,
    server_health,
    export_memories,
    import_memories,
    drop_project,
)

# `MCP_TOOLSET=core` exposes only the daily read/write/graph surface (13
# tools); `full` (default) exposes all 21. Maintenance and batch tools are
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


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    host = os.environ.get("MCP_HOST", "0.0.0.0")
    port = int(os.environ.get("MCP_PORT", "8766"))

    try:
        ensure_schema()
        logger.info("Schema OK")
    except Exception as e:
        logger.warning("Schema may already exist: %s", e)

    # Full-text indexes must use the configured dictionary; rebuild stale ones.
    try:
        rebuilt = _migrate_fts_indexes()
        if rebuilt:
            logger.info("Rebuilt %s full-text index(es) for the current dictionary",
                        rebuilt)
    except Exception as e:
        logger.warning("Full-text index migration skipped: %s", e)

    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("UPDATE memories SET superseded_by = %s WHERE superseded_by = 'forgotten'",
                    (SENTINEL_FORGOTTEN,))
        cur.execute("SELECT table_name FROM information_schema.tables WHERE table_name LIKE 'mem_%'")
        for t in cur.fetchall():
            cur.execute(f"UPDATE {t[0]} SET superseded_by = %s WHERE superseded_by = 'forgotten'",
                        (SENTINEL_FORGOTTEN,))
        # Add deleted_at column if missing (migration)
        cur.execute("ALTER TABLE memories ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ")
        cur.execute("SELECT table_name FROM information_schema.tables WHERE table_name LIKE 'mem_%'")
        for t in cur.fetchall():
            cur.execute(f"ALTER TABLE {t[0]} ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ")
        # Migrate existing __forgotten__ sentinel to deleted_at
        cur.execute("UPDATE memories SET deleted_at = updated_at WHERE superseded_by = %s AND deleted_at IS NULL",
                    (SENTINEL_FORGOTTEN,))
        cur.execute("SELECT table_name FROM information_schema.tables WHERE table_name LIKE 'mem_%'")
        for t in cur.fetchall():
            cur.execute(f"UPDATE {t[0]} SET deleted_at = updated_at WHERE superseded_by = %s AND deleted_at IS NULL",
                        (SENTINEL_FORGOTTEN,))
        conn.commit()
        conn.close()
        logger.info("deleted_at column migrated")
    except Exception as e:
        logger.warning("Schema migration skipped: %s", e)

    # Deduplicate AGE edges (keep only the first edge per directed pair+type)
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SET search_path TO ag_catalog, public")
        cur.execute("""
            SELECT * FROM cypher('memory_graph', $$
                MATCH (a)-[r1]->(b)
                WITH a, b, type(r1) AS t, collect(r1) AS rels
                WHERE size(rels) > 1
                RETURN a.name, b.name, t, size(rels)
            $$) AS (src agtype, tgt agtype, rel_type agtype, cnt agtype)
        """)
        dup_rows = cur.fetchall()
        for src, tgt, rel_type, cnt in dup_rows:
            for _ in range(int(str(cnt)) - 1):
                cur.execute(
                    f"SELECT * FROM cypher('memory_graph', $$ "
                    f"MATCH (a {{name: {json.dumps(str(src).strip('\"'))}}})"
                    f"-[r:{str(rel_type).strip('\"')}]->"
                    f"(b {{name: {json.dumps(str(tgt).strip('\"'))}}}) "
                    f"DELETE r RETURN 1 LIMIT 1 $$) AS (x agtype)"
                )
        conn.commit()
        conn.close()
        if dup_rows:
            logger.info("Deduplicated %s duplicate AGE edge groups", len(dup_rows))
    except Exception as e:
        logger.warning("AGE edge dedup skipped: %s", e)

    if transport == "sse":
        import uvicorn
        asgi_app = app.sse_app()
        logger.info("Starting SSE MCP server on %s:%s", host, port)
        uvicorn.run(asgi_app, host=host, port=port, log_level="info")
    elif transport == "streamable-http":
        import uvicorn
        asgi_app = app.streamable_http_app()
        logger.info("Starting Streamable HTTP MCP server on %s:%s", host, port)
        uvicorn.run(asgi_app, host=host, port=port, log_level="info")
    else:
        logger.info("Starting MCP server (transport=stdio)")
        app.run(transport="stdio")
