from mcp.server.fastmcp import FastMCP

app = FastMCP(
    name="loci",
    instructions="""Persistent memory tools for agents.

TOOLS BY TASK
- Store/change/remove: `remember`, `remember_batch`, `update_memory`, `forget`.
- Search/browse/bootstrap: `recall`, `list_memories`, `get_entity`,
  `list_entities`, `init_context`.
- Graph: `relate`, `relate_batch`, `unrelate`, `find_connections`, `find_paths`.
- Maintenance/backup: `server_health`, `contradictions_report`,
  `export_memories`, `import_memories`.

COMMON FLOWS
- Start/search: `init_context(project="api")` → `recall(query="deploy steps", project="api")`.
- Store/link: `remember(content="Service uses Postgres", entity="API", project="api")` →
  `relate(from_entity="API", to_entity="Postgres", relation_type="USES", project="api")`.
- Explore: `find_connections(entity="API", project="api")` or
  `find_paths(from_entity="API", to_entity="Postgres", project="api")`.
- Backup/restore: pass `export_memories(format="json").data` to
  `import_memories(data=..., format="json")`.

Read-scope defaults differ by tool; check each tool's description and pass
`include_global=false` when a project-only read is required. Collection tools
return items in `results`; export and import payloads use `data`.
""",
    streamable_http_path="/",
    json_response=True,
)
