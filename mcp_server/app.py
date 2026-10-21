from mcp.server.fastmcp import FastMCP

app = FastMCP(
    name="loci",
    instructions="""Persistent memory server powered by PostgreSQL + pgvector + Apache AGE.

Use `remember` to store facts, decisions, and observations about entities.
Use `recall` to search semantically, by keyword, or via graph traversal.
Use `relate` to connect entities in the knowledge graph.

Project-scoped reads include global memories by default. Pass
`include_global=false` when strict project isolation is required.
Collection tools use `results` as their canonical list field. Export payloads
are returned in `data`.

=== WHEN TO REMEMBER ===
- User preferences and corrections
- Architectural decisions, conventions, commands
- Facts about people, projects, tools, concepts
- Recurring patterns worth retaining across sessions

=== WHEN TO RECALL ===
- At session start to load relevant context
- When you need information from a previous session
- When the user references something discussed before
- Use `include_decay=true` to prioritize recent / frequently accessed facts
""",
    streamable_http_path="/",
    json_response=True,
)
