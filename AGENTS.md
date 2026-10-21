# Loci MCP Server — Agent Instructions

This project provides a persistent memory MCP server for AI agents,
backed by PostgreSQL + pgvector (vector search) + Apache AGE (graph/Cypher).

## Architecture

```
┌──────────────┐  MCP stdio/HTTP   ┌──────────────────────┐
│  AI Agent     │ ◄──────────────► │  mcp_server/        │
│ (Claude Code, │                  │  Python FastMCP      │
│  Cursor, ...) │                  └──────────┬───────────┘
└──────────────┘                             │
                     ┌───────────────────────┼───────────┐
                     │                       │           │
                     ▼                       ▼           ▼
              ┌──────────────┐      ┌──────────────┐
              │  PostgreSQL   │      │  Inference   │
              │  + pgvector   │      │  API (:8001) │
              │  + Apache AGE │      │  Embeddings  │
              └──────────────┘      │  + NLI (LLM) │
                                    └──────────────┘
```

## Response Envelope

**Every tool answers with the same envelope** — an agent never has to guess the
shape:

```json
{"ok": true, "status": "stored", "id": "6dd84bf5092a4850",
 "error": null, "warnings": []}
```

| Key | Meaning |
|-----|---------|
| `ok` | `false` when the call failed, `true` otherwise |
| `status` | `stored`, `duplicate`, `updated`, `deleted`, `soft_forgotten`, `not_found`, `partial`, `error`, `ok` |
| `id` | Id of the affected memory (`null` for collections and for failures) |
| `error` | Error message, `null` on success |
| `warnings` | Non-fatal notes (skipped checks, superseded facts) |

Tools that return a collection add `count` and `results`. Tools with extra
payload add their own keys (e.g. `counts` for `forget`, `superseded` for
`remember`, `recent_memories` for `init_context`).

Failed calls never raise: they return `ok: false` with `error` set.

## MCP Tools (18 total)

### Write
| Tool | Description |
|------|-------------|
| `remember` | Store a fact/decision/learning with auto-embedding, dedup mode (`dedupe`), supersede variants (`supersede`, `supersede_content`, `supersede_last`), optional contradiction check |
| `remember_batch` | Store multiple facts at once (savepoints); `skip_duplicates`, `dedupe`, per-fact `project`/`scope`/`supersede`/`fuzzy_threshold`, optional `check_contradictions` |
| `update_memory` | Update content, importance, or metadata (re-embeds on content change) |
| `forget` | Soft delete (`deleted_at` + sentinel) or hard delete; `entity` also removes the graph node and that scope's memories; `counts` reports memories/entities/relations separately |

Valid `mem_type` values: `fact`, `decision`, `learning`, `preference`,
`pattern`, `incident`, `runbook`, `context`, `observation`, `convention`,
`command`, `note`.

### Read / Search
| Tool | Description |
|------|-------------|
| `recall` | Hybrid vector + FTS search (RRF fusion, project + global tables); `hybrid`, `min_similarity` to drop weak matches; optional cognitive decay weighting; `include_superseded` / `only_superseded`; `graph_boost` (with `entity`) to include connected entities; `fields` to narrow columns; logs `access_count` |
| `list_memories` | Browse memories without a query (newest first); filters: `entity`, `mem_type`, `project`, `fields` |
| `get_entity` | Get entity details + all linked memories + graph connections; `include_superseded`; `fields` |
| `list_entities` | Browse all entities (global and project tables) |
| `init_context` | Bootstrap: recent memories across all tables + projects + entity count + health; `detail="slim"` (default, deduped) or `"full"` |

### Graph (Apache AGE)
| Tool | Description |
|------|-------------|
| `relate` | Create directed relation between entities (Cypher); auto-creates entities; `project` selects the project graph |
| `relate_batch` | Create/update many relations in one transaction; per-edge `weight`/`context`/`project`; idempotent (MERGE) |
| `unrelate` | Delete a single directed relation |
| `find_connections` | Traverse graph from entity (outgoing/incoming/both); results carry `direction` |
| `find_paths` | Find shortest paths between two entities (`project`-scoped) |

### Maintenance & Advanced
| Tool | Description |
|------|-------------|
| `contradictions_report` | Detect conflicting facts per entity (LLM NLI + vector neighbor pre-filter) |
| `export_memories` | Backup as JSON/JSONL: ids, `superseded_by`, timestamps, project and graph are preserved; payload in `data` |
| `import_memories` | Restore JSON/JSONL (`data` accepts object or string); explicit ids make re-import idempotent; graph is restored from JSON exports |
| `server_health` | Check Postgres + Inference connectivity |

## MCP Resources
| Resource | Description |
|----------|-------------|
| `memory://stats` | Active/total memory counts, entity/relation counts, health |

## Retrieval

- `recall` searches the project table (if `project` is set) **and** the global
  `memories` table. Project rows win when the same content exists in both
  (dedup via `content_hash`); global facts stay visible from inside a project.
- FTS is OR-based (`_fts_query`): a multi-term query matches documents that
  contain any term. FTS rank contributions are weighted at `FTS_WEIGHT = 0.6`
  so a single common word cannot outrank the best semantic hit.
- RRF scores are normalized to `_rrf_norm` (0..1, best hit = 1.0);
  `_rrf_score` keeps the raw fusion value.
- `_similarity` is `1 - cosine distance`, i.e. 0..1 on the same scale as
  `min_similarity`. Exact matches are typically 0.75-0.9, unrelated facts
  below ~0.3.
- `hybrid=false` bypasses RRF and uses a single retriever (vector when an
  embedding exists, otherwise FTS) — `hybrid=true` (default) is recommended.
- `graph_boost=true` (requires `entity`) traverses one hop in the scope graph,
  loads up to 5 memories per connected entity and adds
  `_graph_boost = 0.3 * edge weight` on top of the blended RRF/decay score.
- Embeddings are prompt-aware: queries are sent with `mode="query"`, stored
  content with `mode="document"` (Jina retrieval prefixes). Mixing the two
  makes scores incomparable, so re-embed after changing the model or prompts:
  `docker compose exec loci-mcp python -m mcp_server.reembed`.
- `remember(scope=...)`: `"project"` (default when `project` is set) writes to
  the project table, `"global"` writes to `memories` even inside a project.
  Without `project` everything is global. `remember_batch` accepts the same
  top-level `scope` (per-fact `scope` wins); invalid facts are reported in
  their result entry without aborting the rest of the batch.
- `update_memory`/`forget` locate a `memory_id` in every table when `project`
  is omitted. `forget(entity=...)` removes the entity's memories from all
  scopes and deletes its graph node; with `project` it touches only that
  scope's memories and node. Unknown ids/entities return `NOT FOUND`.
- `include_superseded=true` exposes superseded history but never rows that were
  forgotten (`deleted_at` is set), so soft deletion stays invisible.
  `only_superseded=true` in `recall` returns the invalidated facts only.
- Results are projected: by default a row shows the public fields plus the
  score annotations, never `content_hash`, `deleted_at` or the embedding.
  `fields=["id", "content"]` returns exactly the named columns.

## Cognitive Decay

- Memories age with a half-life of 90 days
- `importance` (0.0–1.0) and `access_count` slow decay (capped at 20 accesses)
- Set `include_decay=true` in `recall` to weight results by recency + frequency
- Decay contributes at most 30% of the final score (`DECAY_WEIGHT = 0.3`);
  the remaining 70% is relevance, so recency cannot drown out semantics
- `min_decay_score` (0..1) filters on normalized decay (`_decay_norm`);
  `_decay_score` keeps the raw value

## Contradiction Detection

Two paths, both using **vector pre-filtering** to avoid massive API calls:

### Path A: `remember` with `check_contradictions=true`

1. The new fact's embedding is computed (already available from step 1)
2. **Vector pre-filter**: pgvector `<=>` finds the 15 nearest neighbors (index scan, no full-table scan)
3. **NLI per candidate**: `/nli` (configured LLM) checks each candidate for a real contradiction
4. At score ≥ 0.7 the old fact is automatically invalidated via `superseded_by = new_id`

→ **Change**: Previously `/rerank` (cross-encoder) was used, which produced false positives (semantic similarity ≠ contradiction). Endpoint and model were removed.

### Path B: `contradictions_report` (Standalone-Tool)

1. All **active** facts for the entity are loaded (max 200; already superseded pairs are done and not re-checked)
2. **Vector self-join**: for each fact, the 20 nearest neighbors are fetched via `ORDER BY embedding <=>`
3. Only the deduplicated candidate pairs are checked via `/nli` (configured LLM)
4. Pairs with confidence > 0.7 count as a contradiction

→ **Savings**: O(n²) → O(n × k) LLM calls (k=20).

### Inference Endpoints

| Endpoint | Model | Usage |
|----------|-------|-------|
| `/embed` | `jinaai/jina-embeddings-v5-text-nano-retrieval` (local, CPU) | Vectors for `query` and `document` |
| `/nli` | Configurable (Groq by default; OpenAI, Anthropic, Ollama, OpenAI-compatible) | `_check_contradictions` + `contradictions_report` |

Provider selection lives in `inference/llm.py` and is controlled by
`LLM_PROVIDER` (`groq`, `openai`, `anthropic`, `ollama`, `openai-compatible`,
`auto`, `none`). `LLM_MODEL`, `LLM_API_KEY` and `LLM_BASE_URL` override the
provider defaults; `GROQ_API_KEY`/`GROQ_MODEL` remain backwards compatible.
For OpenAI-style providers `/nli` first tries `json_schema` + `strict: true`,
then `json_object`, then a plain request. Anthropic uses `/messages`; no
`response_format` is sent.

**Note**: The quality of contradiction detection depends heavily on the embedding model (`jina-embeddings-v5`), but that is not a problem — the model provides very good similarity values for semantically similar texts. Distracting prefixes such as `Batch-X: ` in the content should be avoided because they destroy embedding similarity. If necessary, increase `neighbors_per_fact` in `contradictions_report`.

### Auto-Supersede

If `_check_contradictions` detects a contradiction, the old fact is automatically marked with `superseded_by = new_mem_id` (in both tables: project-specific + global `memories`). The old fact is then invisible to all queries (everything filters `superseded_by IS NULL`).

With `include_superseded=true` in `recall` or `get_entity`, superseded facts can be retrieved again (recognizable by the set `superseded_by` field).

## Scope Isolation

- Every fact belongs to exactly one scope: the global `memories` table or a
  per-project table `mem_<suffix>`.
- Project names may contain any characters. Plain lower-case identifiers up to
  40 chars are used verbatim as suffix (backwards compatible). All other names
  are lower-cased, truncated and get a short SHA-1 of the *original* name
  appended (`Project Alpha-2` → `project_alpha_2_e6a52393`, `Team` and `team`
  get different suffixes). This keeps names below PostgreSQL's 63-char
  identifier limit, so long names and case variants cannot share a table.
- `global` is reserved: passing `project="global"` means "no project" (global
  scope), it never creates a project of that name.
- The `projects` registry maps suffix → original project name, so results
  show the name the caller passed. Existing plain-identifier tables keep
  working unchanged.
- The graph holds one AGE label per scope (`Entity` for global,
  `Project_<suffix>` for a project) plus a `project` property; the SQL mirror
  `entities` has `UNIQUE(name, project)`. Two projects can use the same entity
  name without collisions, and global facts stay visible inside a project.
- `get_entity`, `list_entities`, `list_memories`, `relate`, `relate_batch`,
  `unrelate`, `find_connections`, `find_paths`, `forget`, `export_memories` and
  `import_memories` accept `project`; omitting it means global scope (for
  `export`/`init_context`: all scopes). `forget(entity=..., project=...)`
  deletes only that scope's node and memories; without `project` it removes
  every scope.

## Backup & Restore

- `export_memories` (JSON, version `0.2`) keeps ids, `superseded_by`,
  `deleted_at`, timestamps, `access_count`, the project of every record and —
  for JSON — the full graph (entities with scope plus relations with weight and
  scope). JSONL is flat: one record per line, no graph. `include_embeddings=true`
  exports the vectors, `include_superseded=true` the history,
  `include_graph=false` omits the graph. The payload is returned in `data`.
- `import_memories` restores ids and timestamps, recreates per-project tables
  on demand and upserts entities/relations with MERGE (idempotent). Without
  `project` every record keeps its export scope; with `project` all records go
  into that project. Re-importing the same file reports those ids as `skipped`;
  `skip_duplicates=false` inserts them again instead.

## Configuration

| Env Var | Default | Description |
|---------|---------|-------------|
| `MEMORY_DATABASE_URL` | `postgresql://memory:geheim@localhost:5432/memory` | PostgreSQL connection string |
| `MEMORY_INFERENCE_URL` | `http://localhost:8001` | Inference API base URL |
| `MCP_TRANSPORT` | `stdio` | `stdio`, `sse` or `streamable-http` |
| `MCP_HOST` | `0.0.0.0` | Bind address (`sse`/`streamable-http` only) |
| `MCP_PORT` | `8766` | Port (`sse`/`streamable-http` only) |
| `LLM_PROVIDER` | `groq` | `groq`, `openai`, `anthropic`, `ollama`, `openai-compatible`, `auto` or `none` |
| `LLM_MODEL` | provider default | NLI model (empty = provider default) |
| `LLM_API_KEY` | – | Generic key; overrides the provider-specific key |
| `LLM_BASE_URL` | provider default | Base URL for OpenAI-compatible/Ollama endpoints |
| `GROQ_API_KEY` | – | Groq key (backwards compatible) |
| `GROQ_MODEL` | `openai/gpt-oss-20b` (compose) | Groq model (backwards compatible) |
| `OPENAI_API_KEY` | – | Used when `LLM_PROVIDER=openai` |
| `ANTHROPIC_API_KEY` | – | Used when `LLM_PROVIDER=anthropic` |
| `OLLAMA_BASE_URL` | `http://host.docker.internal:11434/v1` | Ollama endpoint |

Without a usable LLM configuration `/nli` returns `0.0` for every pair;
`LLM_PROVIDER=none` disables it explicitly.

## Tool Parameter Reference

**Important**: Use these exact parameter names when calling tools (verified Oct 2026):

| Tool | Key Parameters | Notes |
|------|---------------|-------|
| `get_entity` | `entity` (`name` kept as alias) | Entity name to fetch details for |
| `relate` | `from_entity`, `to_entity` (not `from`, `to`) | Source and target entity names |
| `relate_batch` | `edges: [{from_entity, to_entity, relation_type, weight?, context?, project?}]` | One transaction, MERGE-based (idempotent) |
| `unrelate` | `from_entity`, `to_entity`, `relation_type`, `project` | Deletes one directed edge |
| `recall` | `min_similarity` (0..1), `min_decay_score` (0..1), `hybrid`, `include_decay`, `include_superseded`, `only_superseded`, `graph_boost`, `fields` | Drops semantically weak / stale hits; `graph_boost` requires `entity` |
| `list_memories` | `entity`, `mem_type`, `project`, `limit`, `fields` | Browse without a query |
| `remember` | `scope: "project" \| "global"`, `dedupe: "exact" \| "semantic" \| "off"`, `fuzzy_threshold` (0..1), `supersede`, `supersede_content`, `supersede_last` | `global` stores in `memories` even when `project` is set |
| `remember_batch` | `facts: [{content, project?, scope?, supersede?, fuzzy_threshold?, dedupe?, ...}]` | Savepoints; invalid facts are reported individually |
| `find_connections` | `entity` (`entity_name` kept as alias) | Entity to traverse from |
| `contradictions_report` | `entity` | Required; checks one entity at a time |
| `forget` | `memory_id`, `entity` (`entity_name` kept as alias), `soft`, `project` | `counts` splits memories/entities/relations |
| `export_memories` | `format` (`json`/`jsonl`), `project`, `include_embeddings`, `include_superseded`, `include_graph` | Payload is returned in `data`; graph is only included in JSON exports |
| `import_memories` | `data` (object or string), `project`, `format`, `skip_duplicates`, `fuzzy_threshold` | Without `project` records keep their export scope |

Dedupe modes: `exact` (default, content hash), `semantic` (also vector
similarity, threshold `fuzzy_threshold`, default 0.9), `off` (store everything).

## When to use which tool

- **Session start** → `init_context`
- **Store user fact** → `remember` with `mem_type="preference"`, attach `entity`
- **Search past context** → `recall` with `hybrid=true`, `include_decay=true`
- **Browse without semantics** → `list_memories` / `list_entities`
- **Connect concepts** → `relate` (auto-creates AGE nodes + edges); many at once → `relate_batch`
- **Undo a relation** → `unrelate`
- **Explore relationships** → `find_connections` or `find_paths`
- **Check for stale/wrong facts** → `contradictions_report`
- **Remove stale data** → `forget` with `soft=true`
- **Backup** → `export_memories`
- **Restore** → `import_memories`
- **Debug** → `server_health`, `memory://stats`

## Running

```bash
# Local dev (separate terminals):
docker compose up -d postgres inference
pip install -r mcp_server/requirements.txt
python -m mcp_server

# Full stack (all services; configure an LLM provider in .env):
cp .env.example .env   # Groq by default; see LLM_PROVIDER for alternatives
docker compose up -d

# After changing embedding model or prompts (rebuilds both images):
docker compose build inference loci-mcp && docker compose up -d
docker compose exec loci-mcp python -m mcp_server.reembed

# Client config (e.g. Cursor/Claude):
# Add to mcpServers:
"loci": {
  "command": "python",
  "args": ["-m", "mcp_server"],
  "env": {
    "MEMORY_DATABASE_URL": "postgresql://memory:geheim@localhost:5432/memory",
    "MEMORY_INFERENCE_URL": "http://localhost:8001"
  }
}
```


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

