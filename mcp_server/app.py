from mcp.server.fastmcp import FastMCP

app = FastMCP(
    name="loci",
    instructions="""Persistent memory tools for agents.

Use `init_context` at session start, `remember` to store durable facts, and
`recall` to search. Use graph tools to connect and traverse entities.
Read-scope defaults differ by tool; check each tool's description and pass
`include_global=false` when a project-only read is required. Collection tools
return items in `results`; export and import payloads use `data`.
""",
    streamable_http_path="/",
    json_response=True,
)
