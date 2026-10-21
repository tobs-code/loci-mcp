# Loci — PostgreSQL Vector + Graph MCP Server

PostgreSQL 17 with **pgvector** (vector similarity search), **Apache AGE** (Cypher graph queries) and a **Loci MCP Server** for AI agents.

## Quick Start

```bash
cp .env.example .env   # configure an LLM provider (Groq by default)
docker compose up -d
```

> Contradiction detection needs an LLM key. Groq is the default; set
> `LLM_PROVIDER` to `openai`, `anthropic`, `ollama`, `openai-compatible` or
> `auto` to use another provider, or `none` to disable NLI. Embeddings always
> run locally on CPU.

After that, these services are running:
| Service | Port | Description |
|---------|------|-------------|
| **PostgreSQL** | `5432` | User/DB: `memory`, password: `geheim` |
| **Inference API** | `8001` | Embeddings (Jina v5 nano) + NLI (configurable LLM) |
| **Loci MCP** | `8766` | MCP server (Streamable HTTP) for AI agents |

## MCP Server — Client Connection

The Loci MCP server uses the **Streamable HTTP transport** and is reachable at `http://localhost:8766/`.

### Zed

Open `settings.json` (`~/.config/zed/settings.json` or `Cmd+Shift+P` → `zed: open settings file`) and add:

```json
{
  "context_servers": {
    "loci": {
      "url": "http://localhost:8766"
    }
  }
}
```

### VS Code (with Cline / Roo Code / Continue)

Add the following to your MCP configuration (`cline_mcp_settings.json` or `mcp.json`):

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

> **Note:** Port `8766` is already exposed by `docker-compose.yml`. From another machine, use the host IP instead of `localhost`.

### Claude Code (stdio mode, local development)

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

---

## MCP Tools (18 Tools + 1 Resource)

### Response Envelope

Every tool answers with the same shape, so an agent never has to guess:

```json
{"ok": true, "status": "stored", "id": "6dd84bf5092a4850",
 "error": null, "warnings": []}
```

`ok` is `false` on failure, `status` is one of `stored`, `created`, `duplicate`,
`updated`, `deleted`, `soft_forgotten`, `not_found`, `partial`, `error`, `ok`,
`id` is the affected memory id (`null` for collections), `error` carries the
message, `warnings` collects non-fatal notes. Collection tools add `count` and
`results`; failed calls never raise.

### Write
| Tool | Description |
|------|-------------|
| `remember` | Store a fact/decision/learning (auto-embedding, `dedupe` mode, `supersede`/`supersede_content`/`supersede_last`, optional contradiction check, `scope` global/project) |
| `remember_batch` | Store multiple facts (savepoints; per fact: `project`, `scope`, `supersede`, `fuzzy_threshold`, `dedupe`; optional `check_contradictions`) |
| `update_memory` | Update content (re-embedding), importance or metadata |
| `forget` | Soft delete (`deleted_at` + sentinel) or hard delete; `entity` also removes the graph node and that scope's memories; `counts` splits memories/entities/relations |

### Read / Search
| Tool | Description |
|------|-------------|
| `recall` | Hybrid vector + FTS (RRF fusion, project + global); `hybrid`, `min_similarity`, `include_decay`/`min_decay_score`, `include_superseded`/`only_superseded`, `include_global`, `graph_boost`, `fields` |
| `list_memories` | Browse memories without a query (newest first); `entity`, `mem_type`, `project`, `include_global`, `fields` |
| `get_entity` | Entity details (including graph connections) + all linked memories; `include_global`, `include_superseded`, `fields` |
| `list_entities` | Browse entities (`entity_type`, `project`, `include_global`) |
| `init_context` | Session bootstrap: recent memories (slim, near-duplicates removed), counters, projects, graph entity count, health; `detail="slim"\|"full"`, `include_global` |

### Graph (Apache AGE)
| Tool | Description |
|------|-------------|
| `relate` | Directed relation via Cypher (auto-creates entities); `project` selects the project graph |
| `relate_batch` | Many relations in one transaction (MERGE, idempotent, per-edge `weight`/`context`/`project`) |
| `unrelate` | Delete a single directed relation |
| `find_connections` | Graph traversal (outgoing/incoming/both), results carry `direction` |
| `find_paths` | Shortest paths between two existing entities (`project`-scoped); missing entities return `not_found` |

### Maintenance
| Tool | Description |
|------|-------------|
| `contradictions_report` | Detect conflicting facts for **one entity** (LLM NLI + vector pre-filter); `entity` is required |
| `export_memories` | Backup as JSON/JSONL (`format`, `project`, `include_embeddings`, `include_superseded`, `include_graph`); payload in `data` |
| `import_memories` | Restore from JSON/JSONL (`data` accepts object or string, `project` override, `format`, `skip_duplicates`, `fuzzy_threshold`); IDs are preserved |
| `server_health` | Check Postgres + Inference connectivity |

### Resource
| Resource | Description |
|----------|-------------|
| `memory://stats` | Memory/entity/relation counts + health |

---

## Retrieval

- **Read scope defaults:** `recall` and `get_entity` use global scope when
  `project` is omitted; with a project they include that project's and global
  memories by default. `list_memories` and `list_entities` list all scopes when
  `project` is omitted, and project + global by default when it is set. Pass
  `include_global=false` for project-only reads. `init_context` uses global
  scope when `project` is omitted, project + global when set, and all scopes
  with `project="*"`; `include_global=false` excludes global memories.
- `recall` searches the project table (if `project` is set) **and** the
  global `memories` table; global facts remain visible inside a project.
  Project hits receive a small scope tie-breaker. Set `include_global=false`
  for a project-only search.
- FTS is OR-based (`_fts_query`), weighted with `FTS_WEIGHT = 0.6` so a single
  common word cannot outrank the best semantic hit.
- `_similarity` = `1 - cosine distance` (0..1), the same scale as `min_similarity`.
  `recall` defaults to `min_similarity=0.3`; pass `null` to opt out.
- Embeddings are prompt-aware: queries use `mode="query"`, stored content uses
  `mode="document"`. After changing the model or prompts, re-embed:
  `docker compose exec loci-mcp python -m mcp_server.reembed`.
- `remember(scope="global")` writes to `memories` even inside a project.
- `graph_boost=true` (only in combination with `entity`) pulls in memories from
  directly connected graph entities and weights them with `_graph_boost`
  (30% of the edge weight); RRF remains the basis of the score.
- `include_superseded=true` shows superseded history, but never deleted rows
  (`deleted_at` is set) — soft delete stays invisible.

## Memory Types

Allowed values: `fact`, `decision`, `learning`, `preference`, `pattern`,
`incident`, `runbook`, `context`, `observation`, `convention`, `command`,
`note`. Other values are rejected (`MEMORY_TYPES` in `config.py`).

## Configuration

| Env Var | Default | Description |
|---------|---------|-------------|
| `MEMORY_DATABASE_URL` | `postgresql://memory:geheim@localhost:5432/memory` | Postgres DSN |
| `MEMORY_INFERENCE_URL` | `http://localhost:8001` | Inference API |
| `MCP_TRANSPORT` | `stdio` | `stdio`, `sse` or `streamable-http` |
| `MCP_HOST` | `0.0.0.0` | Bind address (`sse`/`streamable-http` only) |
| `MCP_PORT` | `8766` | Port (`sse`/`streamable-http` only) |
| `LLM_PROVIDER` | `groq` | `groq`, `openai`, `anthropic`, `ollama`, `openai-compatible`, `auto` or `none` |
| `LLM_MODEL` | provider default | NLI model (empty = provider default) |
| `LLM_API_KEY` | – | Generic key; overrides the provider-specific key |
| `LLM_BASE_URL` | provider default | Base URL for OpenAI-compatible or Ollama endpoints |
| `GROQ_API_KEY` | – | Groq key (backwards compatible) |
| `GROQ_MODEL` | `openai/gpt-oss-20b` (compose) | Groq model (backwards compatible) |
| `OPENAI_API_KEY` | – | Used when `LLM_PROVIDER=openai` |
| `ANTHROPIC_API_KEY` | – | Used when `LLM_PROVIDER=anthropic` |
| `OLLAMA_BASE_URL` | `http://host.docker.internal:11434/v1` | Ollama endpoint |

### LLM Providers

`/nli` uses one provider at a time. The default is Groq; the legacy
`GROQ_API_KEY`/`GROQ_MODEL` variables keep working.

| Provider | Default model | Default endpoint |
|----------|---------------|------------------|
| `groq` | `openai/gpt-oss-20b` | `https://api.groq.com/openai/v1` |
| `openai` | `gpt-4o-mini` | `https://api.openai.com/v1` |
| `anthropic` | `claude-3-5-haiku-latest` | `https://api.anthropic.com/v1` |
| `ollama` | `llama3.2` | `http://host.docker.internal:11434/v1` |
| `openai-compatible` | – (set `LLM_MODEL`) | – (set `LLM_BASE_URL`) |

Examples:

```dotenv
# OpenAI
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...

# Anthropic
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...

# Ollama on the host
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://host.docker.internal:11434/v1

# Any OpenAI-compatible endpoint (LM Studio, vLLM, OpenRouter, ...)
LLM_PROVIDER=openai-compatible
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_API_KEY=...
LLM_MODEL=...
```

`LLM_PROVIDER=auto` picks the first configured key (Groq, OpenAI, Anthropic),
then `LLM_BASE_URL`, then `OLLAMA_BASE_URL`. `LLM_PROVIDER=none` disables
contradiction detection; `/nli` then returns `0.0` for every pair.

## Scopes & Project Names

- Every fact belongs to exactly one scope: the global `memories` table or a
  project table `mem_<suffix>`.
- Project names may contain any characters. Plain lowercase identifiers up to
  40 characters are used verbatim as the suffix; all other names are
  lowercased, truncated and get a short SHA-1 of the original name appended
  (`Project Alpha-2` → `project_alpha_2_e6a52393`; `Team` and `team` get
  different suffixes). This keeps table and index names below the 63-character
  limit and prevents case variants from colliding.
- `global` is reserved: `project="global"` means "no project" and never creates
  a project with that name.
- The `projects` table maps suffix → original name, so results show the name
  that was passed in.
- The graph has one AGE label per scope (`Entity` for global,
  `Project_<slug>` for a project) plus a `project` property; `entities` has
  `UNIQUE(name, project)`. Identical entity names therefore do not collide
  across projects.
- `project` is accepted by reads and graph tools. Graph tools use the global
  graph when it is omitted; `export_memories` exports all scopes when omitted,
  and `import_memories` restores each record to its exported scope. For
  `init_context`, use `project="*"` to include all scopes.

## Backup & Restore

- `export_memories` (JSON, version `0.2`) contains IDs, `superseded_by`,
  `deleted_at`, timestamps, `access_count`, the project of each record and the
  complete graph (entities with scope, relations with weight/scope). JSONL is
  flat (one record per line, no graph). `include_embeddings=true` includes the
  vectors, `include_superseded=true` the history; `include_graph=false` omits
  the graph.
- `import_memories` restores IDs and timestamps, creates project tables on
  demand and upserts entities/relations via MERGE (idempotent). Without
  `project`, each record lands in its export scope; with `project`, all records
  are written to that project. Re-importing the same file reports existing IDs
  as `skipped`; `skip_duplicates=false` allows duplicates.

## Cognitive Decay

- Memories age with a half-life of 90 days
- `importance` (0.0–1.0) and `access_count` (max. 20) slow down decay
- `include_decay=true` in `recall` weights by recency + frequency
- Decay contributes at most 30% to the score (`DECAY_WEIGHT = 0.3`), 70% stays relevance
- `min_decay_score` (0..1) filters on normalized decay (`_decay_norm`)

## Contradiction Detection

With `remember(check_contradictions=true)`, the 15 semantically nearest active
facts of the same entity (`mem_type='fact'`) are pre-filtered per pgvector —
separately for the project and global tables — and then checked for a real
contradiction via the configured LLM (`/nli`, Groq by default). From a score
of ≥ 0.7, the old fact is automatically superseded.

`contradictions_report(entity=...)` works only on **active** facts of an entity:
for each fact, the 20 nearest vector neighbors are fetched and only the
deduplicated pairs are checked via NLI (`confidence > 0.7`).

## Schema

| Table | Purpose |
|-------|---------|
| `memories` | Global text content with vector embedding (768d) + FTS + decay |
| `mem_<suffix>` | Project-specific memories (created dynamically on demand) |
| `projects` | Registry suffix → original project name |
| `entities` | Graph nodes (people, concepts, places...) with `project` scope |
| `relations` | Graph edges between entities (with `project` scope) |

The AGE graph `memory_graph` is created automatically via `create_graph()`.

## Vector Search (pgvector)

```sql
SELECT id, content, metadata
FROM memories
ORDER BY embedding <=> '[0.1, 0.2, ...]'
LIMIT 5;
```

## Graph Queries (Apache AGE)

```sql
SELECT * FROM cypher('memory_graph', $$
    MATCH (p:Person)-[:KNOWS]->(friend)
    RETURN p.name, friend.name
$$) AS (name agtype, friend_name agtype);
```

## Docker Configuration

- **Postgres**: `postgres:17-bookworm` + pgvector v0.8.1 + Apache AGE 1.7.0
- **`shm_size: 1g`**: AGE needs more shared memory
- **Inference**: FastAPI + sentence-transformers (CPU)
- **Loci MCP**: Python FastMCP server with uvicorn (Streamable HTTP transport)

After changing the embedding model or prompts, rebuild both images and then
re-embed:

```bash
docker compose build inference loci-mcp && docker compose up -d
docker compose exec loci-mcp python -m mcp_server.reembed
```

On startup, the MCP server runs idempotent migrations
(`deleted_at` backfill, sentinel migration, dedup of duplicate AGE edges).

## Inference API

```python
import requests

# Embedding (Jina v5 nano, 239M) → 768d normalized vector
# mode: "query" (search query) or "document" (stored content, default)
r = requests.post("http://localhost:8001/embed", json={
    "texts": ["This is a test", "Another text"],
    "mode": "document"
})
vectors = r.json()["embeddings"]

# NLI / contradiction detection (configured LLM provider)
r = requests.post("http://localhost:8001/nli", json={
    "query": "Max lives in Berlin",
    "documents": ["Max lives in Hamburg"]
})
contradiction_scores = r.json()["contradiction_scores"]
```

## Testing

Tests run against an isolated `loci_test` database (created and dropped
automatically) and a local fake inference server, so no LLM key is needed.

```bash
pip install -r requirements-dev.txt
python -m pytest
```

## Files

| File | Purpose |
|------|---------|
| `Dockerfile` | Custom Postgres image with pgvector + AGE |
| `docker-compose.yml` | Container orchestration |
| `init.sql` | Schema + extensions + AGE graph |
| `mcp_server/` | Python MCP server (FastMCP + pgvector + AGE) |
| `mcp_server/server.py` | Public facade: registers the tools and starts the transports |
| `mcp_server/app.py` | FastMCP instance and server instructions |
| `mcp_server/config.py` | Memory types, scopes and scoring constants |
| `mcp_server/envelope.py` | Uniform response envelope shared by all tools |
| `mcp_server/projection.py` | Row projection (`fields`) and near-duplicate detection |
| `mcp_server/db.py` | Schema, migrations, connection handling |
| `mcp_server/scopes.py` | Project slugs, scope labels and per-project tables |
| `mcp_server/scoring.py` | Decay, RRF helpers and ID/content hashing |
| `mcp_server/storage.py` | Store, supersede, dedup, contradiction and access helpers |
| `mcp_server/search.py` | Vector + FTS RRF search and graph boost |
| `mcp_server/graph.py` | Apache AGE relations and traversal tools |
| `mcp_server/backup.py` | Export/import of memories and the graph |
| `mcp_server/tools/` | Write, read and maintenance MCP tools |
| `mcp_server/inference_client.py` | HTTP client for `/embed`, `/nli`, `/health` |
| `mcp_server/reembed.py` | Re-embed all memories after a model/prompt change |
| `tests/` | Integration tests against an isolated `loci_test` database |
| `inference/Dockerfile` | Inference service with sentence-transformers |
| `inference/app.py` | FastAPI server (`/embed`, `/nli`, `/health`) |
| `inference/llm.py` | Provider abstraction for Groq, OpenAI, Anthropic, Ollama and OpenAI-compatible APIs |
| `AGENTS.md` | Full tool documentation for AI agents |
