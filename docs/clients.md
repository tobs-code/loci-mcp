# Client Setup

The Loci MCP server speaks the **Streamable HTTP transport** and is reachable
at `http://localhost:8766/`. Start it with `docker compose up -d`.

## Streamable HTTP (most clients)

### VS Code (with Cline / Roo Code / Continue)

Add the following to your MCP configuration (`cline_mcp_settings.json` or
`mcp.json`):

```json
{
  "mcpServers": {
    "loci": {
      "type": "streamable-http",
      "url": "http://localhost:8766"
    }
  }
}
```

### Cursor / Windsurf / Copilot

```json
{
  "mcpServers": {
    "loci": {
      "type": "streamable-http",
      "url": "http://localhost:8766"
    }
  }
}
```

### Zed

Open `settings.json` (`~/.config/zed/settings.json` or
`Cmd+Shift+P` → `zed: open settings file`) and add:

```json
{
  "context_servers": {
    "loci": {
      "url": "http://localhost:8766"
    }
  }
}
```

> **Note:** Port `8766` is already exposed by `docker-compose.yml`. From another
> machine, use the host IP instead of `localhost`.

## stdio (local development, e.g. Claude Code)

```bash
# Terminal 1: start Postgres + Inference
docker compose up -d postgres inference

# Terminal 2: start the MCP server in stdio mode
pip install -r mcp_server/requirements.txt
python -m mcp_server
```

Then in the Claude Code MCP config:

```json
{
  "mcpServers": {
    "loci": {
      "command": "python",
      "args": ["-m", "mcp_server"],
      "env": {
        "MEMORY_DATABASE_URL": "postgresql://memory:geheim@localhost:5432/memory",
        "MEMORY_INFERENCE_URL": "http://localhost:8001"
      }
    }
  }
}
```

## Transports

| Transport | Selected with | Notes |
|-----------|---------------|-------|
| `stdio` | `MCP_TRANSPORT=stdio` (default) | The client starts the server itself. |
| `streamable-http` | `MCP_TRANSPORT=streamable-http` | The server listens on `MCP_HOST:MCP_PORT` (default `0.0.0.0:8766`). |
| `sse` | `MCP_TRANSPORT=sse` | Legacy transport. |

## Checking the connection

```bash
curl http://localhost:8001/health   # inference
```

The MCP server itself answers `server_health` with the status of Postgres, the
inference API and the LLM provider. It also reports `version`,
`code_fingerprint` (a hash of the Python files that are actually running) and
`started_at`. Compare `code_fingerprint` with the workspace to detect a stale
container:

```bash
python -c "from mcp_server.version import code_fingerprint; print(code_fingerprint())"
```
