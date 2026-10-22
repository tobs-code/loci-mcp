# Loci MCP Server — Agent Instructions

This project provides a persistent memory MCP server for AI agents,
backed by PostgreSQL + pgvector (vector search) + Apache AGE (graph/Cypher).

> This file is the machine-facing contract. The human documentation lives in
> [`docs/`](docs/): [tools](docs/tools.md), [retrieval](docs/retrieval.md),
> [scopes](docs/scopes.md), [backup](docs/backup.md),
> [configuration](docs/configuration.md), [clients](docs/clients.md),
> [architecture](docs/architecture.md).

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
| `status` | `stored`, `created`, `duplicate`, `updated`, `deleted`, `soft_forgotten`, `not_found`, `partial`, `error`, `ok` |
| `id` | Id of the affected memory (`null` for collections and for failures) |
| `error` | Error message, `null` on success |
| `warnings` | Non-fatal notes (skipped checks, superseded facts) |

Tools that return a collection add `count` and `results`. Tools with extra
payload add their own keys (e.g. `counts` for `forget`, `superseded` for
`remember`, `recent_memories` for `init_context`). Export payloads are in
`data`; collection results are in `results`.

Failed calls never raise: they return `ok: false` with `error` set. That also
holds for an invalid call (missing argument, wrong type): the argument check
answers with the same envelope (`error` names the argument) instead of a raw
validation error.

## MCP Tools (21 total)

### Write
| Tool | Description |
|------|-------------|
| `remember` | Store a fact/decision/learning with auto-embedding, dedup mode (`dedupe`), supersede variants (`supersede`, `supersede_content`, `supersede_last`), optional contradiction check; the answer names the `scope`/`project` it wrote to |
| `remember_batch` | Store multiple facts at once (savepoints); `skip_duplicates`, `dedupe`, per-fact `project`/`scope`/`supersede`/`fuzzy_threshold`, optional `check_contradictions`; every entry names its `scope`/`project` |
| `update_memory` | Update content, importance, or metadata (re-embeds on content change) |
| `forget` | Soft delete (`deleted_at` + sentinel) or hard delete; `entity` also removes the graph node and that scope's memories; `entity="*"` purges a whole project; `dry_run` previews exactly what the real call removes; `counts` reports memories/entities/relations separately |
| `restore_memory` | Bring a soft-forgotten memory back (clears `deleted_at` and the forgotten sentinel); only soft-forgotten rows can be restored |

Valid `mem_type` values: `fact`, `decision`, `learning`, `preference`,
`pattern`, `incident`, `runbook`, `context`, `observation`, `convention`,
`command`, `note`.

### Read / Search
| Tool | Description |
|------|-------------|
| `recall` | Hybrid vector + FTS search (RRF fusion, project + global tables); `hybrid`, `min_similarity` to drop weak vector matches; optional cognitive decay weighting; `include_superseded` / `only_superseded`; `graph_boost` (with `graph_entity`) to include connected entities; `fields` to narrow columns; `offset` pages through ranked hits; logs `access_count`; reports `candidates` / `dropped_by_similarity` / `dropped_by_coverage` / `scopes_searched` |
| `list_memories` | Browse memories without a query (newest first); filters: `entity`, `mem_type`, `project`, `include_global`, `fields`, `contains`, `memory_id` (direct id lookup), `include_deleted` (soft-forgotten rows), `include_relation_context` (hide the edge sentences), `updated_after`, `metadata_filter`, `order_by`; `offset` pages through a scope |
| `get_entity` | Get entity details + all linked memories + graph connections; `include_global`, `include_superseded`, `fields`, `limit`/`offset`; the answer reports `total`/`has_more` and warns when it was truncated |
| `list_entities` | Browse entities; filters: `entity_type`, `project`, `include_global`; `offset` pages through the list; synthetic relation-context entities (`<from>_<relation>_<to>`) are hidden; `entity_type` selects one kind of entity (a graph node or a memory-only one) and `count` always equals `total` |
| `init_context` | Bootstrap: recent memories + entity count + health; global by default, project + global when `project` is set, all scopes with `project="*"` (`projects` then includes `"global"`); `detail="slim"` (default, deduped) or `"full"` |
| `list_projects` | List the known project scopes with their sizes (memories, entities, relations) |

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
| `contradictions_report` | Detect conflicting facts per entity (LLM NLI + vector neighbor pre-filter); `resolve="newest"` keeps the newer fact and supersedes the older |
| `export_memories` | Backup as JSON/JSONL: ids, `superseded_by`, timestamps, project and graph are preserved; payload in `data`; warns when the default export leaves superseded/forgotten rows out |
| `import_memories` | Restore JSON/JSONL (`data` accepts object or string); explicit ids make re-import idempotent; graph is restored from JSON exports |
| `drop_project` | Remove a project scope completely (memory table, registry entry, graph nodes); `dry_run` previews the counts |
| `server_health` | Check Postgres + Inference connectivity |

## MCP Resources
| Resource | Description |
|----------|-------------|
| `memory://stats` | Active/total memory counts, entity/relation counts, health |

## Retrieval

- `recall` searches the project table (if `project` is set) **and** the global
  `memories` table. Project rows win when the same content exists in both
  (dedup via `content_hash`); global facts stay visible from inside a project.
- Read scope defaults vary by tool: `recall` and `get_entity` use global scope
  when `project` is omitted; with a project they include that project's and
  global memories by default. `list_memories` and `list_entities` list all
  scopes when `project` is omitted, and project + global by default when it is
  set. `project="global"` means the shared scope in every tool: it is not
  "all scopes". Pass `include_global=false` for project-only reads.
  `init_context` uses global scope when `project` is omitted, project + global
  when it is set, and all scopes with `project="*"`; `include_global=false`
  excludes globals.
- FTS is OR-based (`_fts_query`): a multi-term query matches documents that
  contain any term. The normalized text score is multiplied by **query
  coverage** (matched query lexemes / all query lexemes after stemming and
  stopword removal), so a document that matched one common word of a three-word
  query cannot look like a perfect full-text match. FTS rank contributions are
  weighted at `FTS_WEIGHT = 0.6`, so a single common word cannot outrank the
  best semantic hit.
- A hit that **only** the full-text leg found has no similarity to vouch for
  it. It is dropped when it covers less than `FTS_MIN_COVERAGE` (0.2) of the
  query, so a one-word match of a six-word query no longer comes back as the
  only result. Hits the vector retriever found are never dropped by this.
  `recall` reports the number as `dropped_by_coverage`.
- RRF scores are normalized to `_rrf_norm` (0..1, best hit = 1.0);
  `_rrf_score` keeps the raw fusion value. RRF ranks by position, so it
  barely separates hits (`_rrf_norm` 1.00 / 0.98 / 0.97 for similarities
  0.71 / 0.47 / 0.42) — use `_relevance_score` to compare hits.
- `_similarity` is `1 - cosine distance`, i.e. 0..1 on the same scale as
  `min_similarity`. Measured with the current embedding model: a
  natural-language question that matches scores **0.62-0.95**, a bare keyword
  that occurs verbatim in the memory only **0.30-0.61** (a single token
  dilutes the query vector), unrelated facts below ~0.3.
  `recall` defaults to `min_similarity=0.35`; pass `null` to opt out.
- `min_similarity` filters the **vector** retriever only. A full-text hit is
  kept even when the embedding model scores it badly — that is the case the
  full-text leg exists for (identifiers, file names, foreign words).
  `recall` reports `candidates` (the hits that made it into the ranking),
  `dropped_by_similarity` (the number of vector hits removed, not the number
  kept), `dropped_by_coverage` and `scopes_searched` (the scopes that were
  really searched). An empty result names **every** filter that removed
  something (`min_similarity`, `min_decay_score`, `only_superseded`, the
  coverage floor) — not just the first one — so an empty result is never
  silent and never misdiagnosed.
- `_relevance_score` is normalized to 0..1 and combines both signals: the
  cosine similarity of the vector hit and the rank of the full-text hit
  (`_text_score`). A memory that both retrievers found scores higher than
  either signal alone. It is `null` for graph-only hits, which were not
  retrieved by relevance.
- `_scope_boost` (0.05) is added to `_relevance_score` for a project row, so
  a project memory outranks an equally relevant global one.
- `_final_score` is the ranking score in **0..1**: relevance blended with
  decay plus the graph boost, clamped. It is only present with
  `include_decay=true` or `graph_boost=true`; without them the ranking is
  `_relevance_score`. `_distance` is the raw cosine distance to the query.
- Graph hits carry `_graph_boost` (0.3 × edge weight), `_graph_source` (the
  connected entity) and `_graph_weight` (the edge weight). They count as
  relevance 0 and are lifted by the boost only.
- `hybrid=false` bypasses RRF and uses a single retriever (vector when an
  embedding exists, otherwise FTS) — `hybrid=true` (default) is recommended.
- `graph_boost=true` traverses one hop in the scope graph, loads up to 5
  memories per connected entity and adds
  `_graph_boost = 0.3 * edge weight` on top of the blended relevance/decay
  score. If only graph hits come back, a warning says so. Edge weights are
  limited to 0.0-1.0, so a single edge cannot dominate the ranking.
  The expansion point is `graph_entity`; `entity` is the fallback. With
  `graph_entity` the query search stays unfiltered ("query hits + graph
  neighbourhood"), with only `entity` the search is also restricted to that
  entity — which drops unrelated query hits.
- `recall`/`list_memories` accept `updated_after` (ISO date) and
  `metadata_filter` (key/value pairs the metadata must contain);
  `list_memories` additionally accepts `order_by`
  (`updated_at`, `created_at`, `access_count`, `importance`), `contains`
  (case-insensitive literal substring — the right tool for "which memory
  contains exactly X", where the semantic ranking of `recall` is not needed),
  `memory_id` (direct id lookup in the selected scope(s)),
  `include_deleted` (lists soft-forgotten rows, which then carry `deleted_at`
  and the machine-readable `forgotten: true` flag; the `__forgotten__` sentinel
  is hidden from `superseded_by` in those rows) and
  `include_relation_context` (false hides the relation-context sentences of
  the graph edges, which repeat what the edge already says).
  `recall` accepts `offset` (`has_more`/`next_offset` in the answer).
- `init_context` returns the **most recently changed** memories of the scope
  (never a decay-ranked mix): a freshly stored fact always shows up, even if
  it was never read before. `limit` controls how many (default 10).
  `detail="full"` returns every column except the internal ones and the
  search score annotations (`_decay_score`, `_final_score`, ...): the
  bootstrap is not a search result, it is sorted by the last update.
  `projects` names every scope the answer covers, including the shared one
  (`"global"`); `active_memories` counts the searchable rows, `total_memories`
  every row (so `total = active + superseded + forgotten`);
  `graph_entities_count` counts the graph nodes of the covered scopes, not of
  the whole database.
- A project that does not exist is never silently ignored: `recall`,
  `list_memories`, `list_entities` and `init_context` fall back to the global
  scope and say so in `warnings`.
- The full-text retriever uses the English dictionary (`FTS_LANGUAGE` in
  `config.py`). Index and query side always use the same dictionary; a
  changed dictionary is migrated at startup.
- Embeddings are prompt-aware: queries are sent with `mode="query"`, stored
  content with `mode="document"` (Jina retrieval prefixes). Mixing the two
  makes scores incomparable, so re-embed after changing the model or prompts:
  `docker compose exec loci-mcp python -m mcp_server.reembed`.
- `remember(scope=...)`: `"project"` (default when `project` is set) writes to
  the project table, `"global"` writes to `memories` even inside a project.
  Without `project` everything is global. `remember_batch` accepts the same
  top-level `scope` (per-fact `scope` wins); invalid facts are reported in
  their result entry without aborting the rest of the batch. Both tools report
  the `scope`/`project` they really wrote to (per fact for a batch), so a
  later read or delete addresses the right scope.
- `remember(supersede=...)` and `supersede_content` are checked: an id that
  does not exist (or a content without an active match) produces a warning
  and replaces nothing. The answer never claims a replacement that did not
  happen.
- `update_memory`/`restore_memory`/`forget` locate a `memory_id` in every
  table when `project` is omitted. `forget(entity=...)` removes the entity's
  memories from all scopes and deletes its graph node; with `project` it
  touches only that scope's memories and node. Unknown ids/entities return
  `NOT FOUND`. `dry_run` counts exactly what the real call removes, including
  the graph nodes and relations of every scope when no `project` is given.
- A soft-forgotten memory can be recovered with `restore_memory`: it clears
  `deleted_at` and the forgotten sentinel. `list_memories(include_deleted=true)`
  shows the forgotten rows (with `deleted_at`) so an agent can find the id.
  Hard-deleted rows cannot be restored.
- `remember` warns when the new content matches a **superseded** memory: the
  fact is still stored as active (an intentional revert is legitimate), but the
  response names the old id and its replacement so the fork is visible.
- `include_superseded=true` exposes superseded history but never rows that were
  forgotten (`deleted_at` is set), so soft deletion stays invisible to normal
  reads. `only_superseded=true` in `recall` returns the invalidated facts only.
- `list_entities` hides synthetic relation-context entities
  (`<from>_<relation>_<to>`) that
  exist only to make an edge context searchable; an entity that also has a
  real memory stays listed. `entity_type` selects one kind of entity (a graph
  node or a memory-only one), and `count` always equals `total`.
- The relation-context sentence of an edge is stored under the synthetic
  entity `<from>_<relation>_<to>` (e.g. `Loci_depends_on_PostgreSQL`).
  `get_entity` returns it for both endpoints of the edge,
  and its metadata `weight` follows the edge when the edge is re-weighted.
  `list_memories(include_relation_context=false)` hides those sentences.
- Results are projected: by default a row shows the public fields plus the
  score annotations, never `content_hash`, `deleted_at` or the embedding.
  `fields=["id", "content"]` returns exactly the named columns; `deleted_at`
  can be requested explicitly and is included by `include_deleted=true`.

## Cognitive Decay

- Memories age with a half-life of 90 days
- `importance` (0.0–1.0) and `access_count` slow decay (capped at 20 accesses)
- Set `include_decay=true` in `recall` to weight results by recency + frequency
- Decay contributes at most 30% of the final score (`DECAY_WEIGHT = 0.3`);
  the remaining 70% is relevance, so recency cannot drown out semantics
- `min_decay_score` (0..1) filters on the normalized decay (`_decay_norm`):
  `importance × recency` plus at most 10% access-frequency bonus. A brand-new
  memory with `importance=1.0` scores ~0.9, a 90-day-old one ~0.5, so
  `min_decay_score=0.9` keeps fresh, important facts. The filter runs before
  the access counters of the current query are counted.
- `_decay_score` keeps the raw value (uncapped, can exceed 1.0)

## Contradiction Detection

Two paths, both using **vector pre-filtering** to avoid massive API calls:

### Path A: `remember` with `check_contradictions=true`

1. The new fact's embedding is computed (already available from step 1)
2. **Vector pre-filter**: pgvector `<=>` finds the 15 nearest neighbors (index scan, no full-table scan)
3. **NLI per candidate**: `/nli` (configured LLM) checks each candidate for a real contradiction
4. At score ≥ 0.7 the old fact is automatically invalidated via `superseded_by = new_id`

→ **Change**: Previously `/rerank` (cross-encoder) was used, which produced false positives (semantic similarity ≠ contradiction). Endpoint and model were removed.

→ **Change**: The candidates are no longer restricted to `mem_type = 'fact'`.
A decision that contradicts a pattern is just as harmful as a fact that
contradicts a fact, and the old filter hid most real conflicts (the
conflicting memory is often a decision, a pattern or a convention). The
candidate count (15) is unchanged, so the number of NLI calls stays the same.

### Path B: `contradictions_report` (Standalone-Tool)

1. All **active** memories of the entity are loaded (max 200; already superseded pairs are done and not re-checked)
2. **Vector self-join**: for each memory, the 20 nearest neighbors are fetched via `ORDER BY embedding <=>`
3. Only the deduplicated candidate pairs are checked via `/nli` (configured LLM) – one call per memory, not per pair
4. Pairs with confidence > 0.7 count as a contradiction
5. `resolve="newest"` (opt-in) keeps the most recently updated memory of each
   pair active and marks the older one as superseded; `resolution` on the pair
   names `superseded` and `kept`

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

**NLI prompt**: `/nli` asks the LLM to compare the two facts **field by field**;
a differing attribute (environment, host, port, version, date/time, state) or
an explicit negation is a contradiction (0.9-1.0), newly added or missing
details are not. Earlier prompts judged "deploys to staging" vs "deploys to
production" as neutral (both could be true) — the widened prompt detects those.

**NLI outages are surfaced, never silent**: `/nli` sets an `error` field when
scores are not a real judgment (provider down, not configured); the MCP client
raises, write tools keep the fact but report `contradiction check skipped:
NLI unavailable: ...` in `warnings` **and** `contradictions.skipped`;
`contradictions_report` marks the scan as incomplete in `warnings`
("scan incomplete: ... not a clean contradiction scan"). Zero contradiction
scores are therefore always trustworthy.

### Auto-Supersede

If `_check_contradictions` detects a contradiction, the old fact is automatically marked with `superseded_by = new_mem_id` (in both tables: project-specific + global `memories`). The old fact is then invisible to all queries (everything filters `superseded_by IS NULL`).

With `include_superseded=true` in `recall` or `get_entity`, superseded facts can be retrieved again (recognizable by the set `superseded_by` field).

Metadata of replaced memories is not lost: keys the new fact does not set are merged into the new memory (explicitly passed metadata always wins); the response lists the carried keys as a warning.

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
- `get_entity`, `list_entities`, `list_memories`, `recall`, `init_context`,
  `relate`, `relate_batch`, `unrelate`, `find_connections`, `find_paths`,
  `forget`, `restore_memory`, `export_memories` and `import_memories` accept
  `project`. Read
  defaults are described above. Graph tools use the global graph when
  `project` is omitted. `export_memories` exports all scopes when omitted;
  `import_memories` restores each record to its exported scope. For
  `init_context`, use `project="*"` to include all scopes. `forget(entity=...,
  project=...)` deletes only that scope's node and memories; without `project`
  it removes every scope.
- A **memory id is unique per scope, not globally**: an import copies the id
  into the target scope. `update_memory`, `restore_memory` and
  `forget(memory_id=...)` refuse an
  id that exists in several scopes and name the scopes in the error
  (`scopes`); `project` says which one to use. `forget(entity=...)` keeps
  removing the entity from every scope.
- `list_projects` shows every scope with its sizes; `drop_project` removes a
  scope completely (memory table, registry entry, graph nodes, mirror rows).
  A scope that only exists in the graph (`relate` without a stored fact) is
  listed like every other scope and is removed as well, so its nodes cannot
  stay behind as orphans.
  `forget(entity="*")` only empties a scope, the scope itself survives.

## Backup & Restore

- `export_memories` (JSON, version `0.2`) keeps ids, `superseded_by`,
  `deleted_at`, timestamps, `access_count`, the project of every record and —
  for JSON — the full graph (entities with scope plus relations with weight and
  scope). JSONL is flat: one record per line, no graph. `include_embeddings=true`
  exports the vectors, `include_superseded=true` the history (superseded and
  soft-forgotten rows), `include_graph=false` omits the graph. The payload is
  returned in `data`. The default export is active-only and **warns** how many
  superseded/forgotten rows it left out, so a backup is never silently
  incomplete.
- `import_memories` restores ids and timestamps, recreates per-project tables
  on demand and upserts entities/relations with MERGE (idempotent). Without
  `project` every record keeps its export scope; with `project` all records go
  into that project. Re-importing the same file reports those ids as `skipped`;
  `skip_duplicates=false` inserts them again instead.
  A record with a broken `metadata` value is reported individually (in
  `errors`) and skipped — the rest of the import still runs.
  `data` accepts the export object or its JSON string. The string form is the
  safer one: some MCP clients repack a JSON array inside a free-form object
  into `{"item": ...}` (the server tolerates that, but why rely on it).

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
contradiction tools report a warning and `nli_configured=false` instead of
presenting that as a clean contradiction scan. `LLM_PROVIDER=none` disables
it explicitly.

## Tool Parameter Reference

**Important**: Use these exact parameter names when calling tools (verified Oct 2026):

| Tool | Key Parameters | Notes |
|------|---------------|-------|
| `get_entity` | `entity` (`name` kept as alias), `limit`, `offset` | Entity name to fetch details for; the answer reports `total`/`has_more` and warns when it was truncated |
| `relate` | `from_entity`, `to_entity` (not `from`, `to`), `weight` (0..1) | Source and target entity names; `weight` is the edge strength |
| `relate_batch` | `edges: [{from_entity, to_entity, relation_type, weight?, context?, project?}]` | One transaction, MERGE-based (idempotent) |
| `unrelate` | `from_entity`, `to_entity`, `relation_type`, `project` | Deletes one directed edge |
| `recall` | `project` (`"*"` = every scope), `min_similarity` (0..1, default 0.35), `min_decay_score` (0..1), `hybrid`, `include_decay`, `include_superseded`, `only_superseded`, `graph_boost`, `graph_entity`, `fields`, `updated_after`, `metadata_filter`, `offset` | Drops weak **vector** hits / stale hits; full-text hits are never dropped by `min_similarity`; `graph_boost` expands from `graph_entity` (fallback `entity`); `offset` pages (`has_more`/`next_offset`); an unknown project falls back to the global scope and says so in `warnings`; the answer names every filter that emptied the result and reports `scopes_searched` |
| `list_memories` | `entity`, `mem_type`, `project`, `include_global`, `limit`, `offset`, `fields`, `updated_after`, `metadata_filter`, `order_by`, `contains`, `memory_id`, `include_deleted`, `include_relation_context` | Browse without a query; `contains` is a case-insensitive literal substring; `memory_id` is a direct lookup; `include_deleted` lists forgotten rows with `deleted_at`; `offset` pages, the response reports `total`/`has_more`/`next_offset`; without `project` every scope is listed, `project="global"` only the shared one |
| `list_entities` | `entity_type`, `project`, `include_global`, `limit`, `offset` | Without `project`, lists all scopes; memory-only rows carry `type="memory-only"`; synthetic `<from>_<relation>_<to>` entities are hidden; `entity_type` selects one kind of entity; `count` equals `total`; sorted by name so `offset` is stable |
| `init_context` | `project` (`"*"` = every scope), `include_global`, `detail`, `limit` | every memory reports its `project`; the list holds the most recently changed memories; `projects` names every covered scope (including `"global"`); `graph_entities_count` counts the covered scopes only |
| `list_projects` | `include_empty` | Lists every scope with memories/entities/relations counts; `total_memories` splits into `active`/`superseded`/`forgotten` |
| `drop_project` | `project`, `dry_run` | Removes the scope completely (table, registry, graph nodes, mirror rows); a graph-only scope is removed as well; `global` is refused (use `forget(entity="*", project="global")`) |
| `remember` | `scope: "project" \| "global"`, `dedupe: "exact" \| "semantic" \| "off"`, `fuzzy_threshold` (0..1), `supersede`, `supersede_content`, `supersede_last` | `global` stores in `memories` even when `project` is set; the answer names the scope it wrote to; an unknown `supersede` id replaces nothing and warns |
| `remember_batch` | `facts: [{content, project?, scope?, supersede?, fuzzy_threshold?, dedupe?, ...}]` | Savepoints; invalid facts (missing `content`, wrong shape) are reported individually, the rest is still stored; every entry names its scope |
| `update_memory` | `memory_id`, `content`, `importance` (0..1), `metadata`, `entity`, `mem_type`, `project` | Content changes are re-embedded; an id that exists in several scopes is refused (`scopes`) |
| `find_connections` | `entity` (`entity_name` kept as alias) | Entity to traverse from; a missing entity names the scope it searched |
| `contradictions_report` | `entity`, `resolve` (`"none"`/`"newest"`) | Required; checks one entity at a time; `newest` supersedes the older fact of each confirmed pair |
| `forget` | `memory_id`, `entity` (`entity_name` kept as alias), `soft`, `project`, `dry_run` | `counts` splits memories/entities/relations; `entity="*"` + `project` purges a whole scope; `dry_run=True` counts exactly what the real call removes |
| `restore_memory` | `memory_id`, `project` | Reverses a soft delete; only rows with `deleted_at` set can be restored |
| `export_memories` | `format` (`json`/`jsonl`), `project`, `include_embeddings`, `include_superseded`, `include_graph` | Payload is returned in `data`; graph is only included in JSON exports |
| `import_memories` | `data` (object or string), `project`, `format`, `skip_duplicates`, `fuzzy_threshold` | Without `project` records keep their export scope |

Every tool parameter carries a description in the JSON schema; the table
above only lists the ones that are easy to get wrong.

Dedupe modes: `exact` (default, content hash), `semantic` (also vector
similarity, threshold `fuzzy_threshold`, default 0.9), `off` (store everything).
A duplicate answer names the mode that matched (`duplicate_mode`) and the
measured similarity.

`supersede_last` replaces the newest memory of an entity only when the new
fact is about the same topic (cosine similarity ≥ 0.5). Otherwise the old
memory stays active, the new fact is stored additionally and the response
carries a warning - replacing an unrelated fact would destroy it silently.

## When to use which tool

- **Session start** → `init_context`
- **Store user fact** → `remember` with `mem_type="preference"`, attach `entity`
- **Search past context** → `recall` with `hybrid=true`, `include_decay=true`
- **Browse without semantics** → `list_memories` / `list_entities` (`offset` pages)
- **Connect concepts** → `relate` (auto-creates AGE nodes + edges); many at once → `relate_batch`
- **Undo a relation** → `unrelate`
- **Explore relationships** → `find_connections` or `find_paths`
- **Check for stale/wrong facts** → `contradictions_report`
- **Remove stale data** → `forget` with `soft=true`
- **Undo a soft delete** → `restore_memory` (find the id with `list_memories(include_deleted=true)`)
- **Remove a whole project** → `drop_project` (after `forget(entity="*")` emptied it)
- **Which projects exist** → `list_projects`
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
