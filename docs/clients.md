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
      "url": "http://localhost:8766",
      "headers": {
        "Authorization": "Bearer <MCP_AUTH_TOKEN>"
      }
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
      "url": "http://localhost:8766",
      "headers": {
        "Authorization": "Bearer <MCP_AUTH_TOKEN>"
      }
    }
  }
}
```

### Zed

Open `settings.json` (`~/.config/zed/settings.json` or
`Cmd+Shift+P`  `zed: open settings file`) and add:

```json
{
  "context_servers": {
    "loci": {
      "url": "http://localhost:8766",
      "headers": {
        "Authorization": "Bearer <MCP_AUTH_TOKEN>"
      }
    }
  }
}
```

Replace `<MCP_AUTH_TOKEN>` with the value from your `.env`. A request without
the header is rejected with HTTP 401.

> **Note:** Port `8766` is published on `127.0.0.1` only. To reach the server
> from another machine, change the compose mapping to `"8766:8766"` and keep
> `MCP_AUTH_TOKEN` set.

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
        "MEMORY_DATABASE_URL": "postgresql://memory:<POSTGRES_PASSWORD>@localhost:5432/memory",
        "MEMORY_INFERENCE_URL": "http://localhost:8001"
      }
    }
  }
}
```

## Transports

| Transport | Selected with | Notes |
|-----------|---------------|-------|
| `stdio` | `MCP_TRANSPORT=stdio` (default) | The client starts the server itself; no token needed. |
| `streamable-http` | `MCP_TRANSPORT=streamable-http` | Listens on `MCP_HOST:MCP_PORT` (default `0.0.0.0:8766`); requires `MCP_AUTH_TOKEN`. |
| `sse` | `MCP_TRANSPORT=sse` | Legacy transport; requires `MCP_AUTH_TOKEN`. |

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
