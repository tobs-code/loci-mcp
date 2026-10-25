# Loci — Memory Server for AI Agents

An MCP server that gives an agent a **persistent memory**: vector search over
PostgreSQL (pgvector), a knowledge graph (Apache AGE) and 28 tools to write,
read, connect and forget.

- **Hybrid retrieval** — vector similarity *and* full-text search, so both
  paraphrases and exact identifiers are found.
- **Scoped** — a global scope for what is true everywhere, a project scope per
  codebase.
- **Self-cleaning** — contradictions are detected and the outdated fact is
  superseded automatically.
- **One answer shape** — every tool returns the same envelope, so an agent
  never has to guess.

## Quick Start

```bash
cp .env.example .env
# Required: set POSTGRES_PASSWORD and MCP_AUTH_TOKEN (see the comments in
# .env.example; generate each with: python -c "import secrets; print(secrets.token_urlsafe(32))")
docker compose up -d
```

That starts three services:

| Service | Port | Description |
|---------|------|-------------|
| **PostgreSQL** | `127.0.0.1:5432` | User/DB from `.env` (`memory`/`memory` by default), password from `POSTGRES_PASSWORD` |
| **Inference API** | `127.0.0.1:8001` | Embeddings (local; GPU when available, CPU fallback) + NLI (configurable LLM) |
| **Loci MCP** | `127.0.0.1:8766` | The MCP server (Streamable HTTP, bearer-token protected) |

> **Authentication.** The HTTP transport exposes every tool, including
> `drop_project` and `prune_memories`. Every request must send
> `Authorization: Bearer <MCP_AUTH_TOKEN>`; the server refuses to start on a
> non-loopback bind without a token. Ports are published on loopback only by
> default. To reach the server from another machine, change the mapping to
> `"8766:8766"` **and** keep the token set.
>
> Contradiction detection needs a configured LLM provider and is **off by
> default** (`LLM_PROVIDER=none`). Set `LLM_PROVIDER` to `groq`, `openai`,
> `anthropic`, `ollama`, `openai-compatible` or `auto` to enable it. Embeddings
> always run locally: the image uses an NVIDIA GPU when Docker exposes one and
> falls back to CPU otherwise. See
> [GPU acceleration](docs/configuration.md#gpu).
>
> **Data egress.** `groq`, `openai` and `anthropic` are hosted providers: when
> a contradiction check runs, the compared memory content is sent to them.
> Use `ollama`/`openai-compatible` with a local endpoint (or `LLM_PROVIDER=none`)
> to keep memory content on this machine. `server_health` reports the active
> provider and whether it is external (`data_egress`).
>
> ```bash
> # Requires a driver and the NVIDIA Container Toolkit
> docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
> ```

## Connect your agent

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

Works with Cursor, Windsurf, Copilot, VS Code (Cline / Roo Code / Continue)
and Zed. Setup for each client, plus the stdio variant for Claude Code, is in
**[docs/clients.md](docs/clients.md)**.

## The tools

**Write**

| Tool | What it does |
|------|--------------|
| `remember` | Store a fact, decision or learning (auto-embedded, deduplicated) |
| `remember_batch` | Store many facts at once |
| `update_memory` | Change content, importance, metadata, entity, type or TTL; `touch=true` refreshes recency, `restore=true` undoes a soft delete |
| `update_batch` | Update many memories in one call |
| `merge_memories` | Fold 2-20 active memories of one scope into one new fact, superseding every source |
| `forget` | Soft delete or hard delete, by id, entity, metadata filter or content match |

**Read**

| Tool | What it does |
|------|--------------|
| `recall` | Hybrid vector + full-text search with ranking, decay and graph boost |
| `recall_batch` | Several independent searches in one call (multi-intent turns) |
| `compile_context` | One token-bounded context block assembled from ranked memories |
| `list_memories` | Browse memories without a query (filter, sort, page); `memory_id` fetches one, including hidden rows |
| `get_entity` | Everything known about one entity |
| `get_entities` | Batch lookup of 1-25 entities in one call |
| `list_entities` | Browse the entities of a scope |
| `list_projects` | Discover every scope (project name, tables, graph, counts) |
| `init_context` | Session bootstrap: recent memories plus `project_stats` for every scope |

**Graph**

| Tool | What it does |
|------|--------------|
| `relate` / `relate_batch` | Create relations between entities |
| `unrelate` | Remove a relation, optionally keeping its context sentence |
| `find_connections` | Traverse the graph from an entity (paged) |
| `find_paths` | Shortest path between two entities |
| `list_relations` | Browse edges without a start entity (paged) |

**Maintenance**

| Tool | What it does |
|------|--------------|
| `contradictions_report` | Find conflicting memories, per entity or scope-wide; `resolve="newest"`, `resolve_min_confidence`, `dry_run` |
| `export_memories` / `import_memories` | Backup and restore, including the graph |
| `verify_integrity` | Find and repair dangling supersede links |
| `prune_memories` | Permanently remove expired/forgotten rows (active memories never touched; `dry_run` by default) |
| `drop_project` | Remove a project scope completely |
| `server_health` | Check Postgres and the inference API |

`MCP_TOOLSET=core` exposes only the 16 daily read/write/graph tools; the
default `full` exposes all 28.

There is also one resource, `memory://stats`, with the memory, entity and
relation counts plus the health of the services.

Every parameter is documented in the tool's own JSON schema. The full
reference is in **[docs/tools.md](docs/tools.md)**; for agents there is
**[AGENTS.md](AGENTS.md)**.

**Memory types.** `remember` takes a `mem_type` that decides how a fact is
used later: `fact` (default), `decision`, `learning`, `preference`, `pattern`,
`incident`, `runbook`, `context`, `observation`, `convention`, `command` or
`note`. Other values are rejected (`MEMORY_TYPES` in `config.py`).

## How it behaves

**One envelope.** Every tool answers with
`{"ok", "status", "id", "error", "warnings"}` plus `count`/`results` for
collections. A failed call never raises — it returns `ok: false` with the
reason, even for an invalid argument.

**Two scopes.** Facts live either in the global scope (true everywhere) or in
a project scope. A project search sees both, so global facts stay visible
inside a project. See [docs/scopes.md](docs/scopes.md).

**Recency without forgetting.** Memories fade with a 90-day half-life, but
recency may only take 30% of the ranking score — semantics always win.
`min_decay_score` filters for "only fresh and important".

**Contradictions clean themselves up.** With `check_contradictions=true` a new
fact is compared against the nearest active memories of its scope; a
contradiction supersedes the old one automatically. Entity tagging is not
required: with an entity, its memories get the candidate seats first, without
one all memories compete (the answer reports `contradictions.cross_entity`).

**The bootstrap is not a search.** `init_context` returns the *most recently
changed* memories, so a fact the agent just stored is always there.

How a query becomes a ranked list — and what every score field means — is
documented in **[docs/retrieval.md](docs/retrieval.md)**.

A three-line example:

```text
init_context(project="api")
recall(query="deployment rollback procedure", project="api", include_decay=true)
remember(content="The API runs on Postgres", entity="API", project="api")
```

## Configuration

| Env Var | Default | Description |
|---------|---------|-------------|
| `MEMORY_DATABASE_URL` | `postgresql://memory@localhost:5432/memory` | Postgres DSN (supply the password here) |
| `POSTGRES_PASSWORD` | - (required by compose) | Password for the bundled Postgres; also used to build the DSN |
| `POSTGRES_USER` / `POSTGRES_DB` | `memory` | Bundled Postgres user and database |
| `MEMORY_INFERENCE_URL` | `http://localhost:8001` | Inference API |
| `MCP_TRANSPORT` | `stdio` | `stdio`, `sse` or `streamable-http` |
| `MCP_HOST` | `0.0.0.0` | Bind address (`sse`/`streamable-http` only) |
| `MCP_PORT` | `8766` | Port (`sse`/`streamable-http` only) |
| `MCP_AUTH_TOKEN` | - (required for HTTP) | Bearer token every HTTP request must send |
| `MCP_ALLOW_INSECURE_HTTP` | `false` | Allow an unauthenticated non-loopback HTTP bind (sandbox only) |
| `INFERENCE_DEVICE` | `auto` | `auto` (CUDA when available), `cpu` or `cuda` |
| `LLM_PROVIDER` | `none` | `groq`, `openai`, `anthropic`, `ollama`, `openai-compatible`, `auto` or `none`; `none` disables NLI (egress is opt-in) |
| `LLM_MODEL` | provider default | NLI model |
| `LLM_REASONING_EFFORT` | `low` for Groq/gpt-oss | Reasoning budget (`low`/`medium`/`high`; `none` disables) |
| `LLM_API_KEY` | – | Generic key; overrides the provider-specific key |
| `LLM_BASE_URL` | provider default | Base URL for OpenAI-compatible or Ollama endpoints |
| `GROQ_API_KEY` / `GROQ_MODEL` | – | Groq key and model (backwards compatible) |
| `OPENAI_API_KEY` | – | Used when `LLM_PROVIDER=openai` |
| `ANTHROPIC_API_KEY` | – | Used when `LLM_PROVIDER=anthropic` |
| `OLLAMA_BASE_URL` | `http://host.docker.internal:11434/v1` | Ollama endpoint |

All providers, the scoring constants and the inference API are documented in
**[docs/configuration.md](docs/configuration.md)**.

## Documentation

| Page | Content |
|------|---------|
| [docs/tools.md](docs/tools.md) | Every tool, every parameter, the envelope, typical workflows |
| [docs/retrieval.md](docs/retrieval.md) | How a query becomes a ranked list; scores, decay, contradictions |
| [docs/scopes.md](docs/scopes.md) | Global vs. project scopes, project names, project lifecycle |
| [docs/backup.md](docs/backup.md) | Export and import, including the graph |
| [docs/configuration.md](docs/configuration.md) | Env vars, LLM providers, Docker, inference API, constants |
| [docs/clients.md](docs/clients.md) | Editor and agent client setups |
| [docs/architecture.md](docs/architecture.md) | Schema, direct SQL, module map, request flow |
| [AGENTS.md](AGENTS.md) | The dense machine-facing contract for agents |

## Development

```bash
pip install -r requirements-dev.txt
python -m pytest          # integration tests against an isolated test database
```

The tests create and drop a `loci_test` database and run a local fake
inference server, so no LLM key is needed. The module map is in
[docs/architecture.md](docs/architecture.md).

## License

Apache 2.0 — see [LICENSE](LICENSE).
