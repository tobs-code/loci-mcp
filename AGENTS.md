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
┌──────────────┐  MCP stdio/HTTP     ┌────────────────────────┐
│  AI Agent    │ ◄────────────────► │  mcp_server/           │
│ (Claude Code,│                     │  Python FastMCP        │
│  Cursor, ...)│                     └──────────┬─────────────┘
└──────────────┘                                │
                           ┌────────────────────┴────────────────────┐
                           │                                         │
                           ▼                                         ▼
              ┌─────────────────────────┐               ┌─────────────────────────┐
              │  PostgreSQL             │               │  Inference API          │
              │  + pgvector             │               │  (:8001)                │
              │  + Apache AGE           │               │  Embeddings + NLI       │
              └─────────────────────────┘               └─────────────────────────┘
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
| `status` | `stored`, `created`, `duplicate`, `updated`, `deleted`, `soft_forgotten`, `restored`, `not_found`, `partial`, `error`, `ok` |
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

## MCP Tools (24 total)

### Write
| Tool | Description |
|------|-------------|
| `remember` | Store a fact/decision/learning with auto-embedding, dedup mode (`dedupe`), supersede variants (`supersede`, `supersede_content`, `supersede_last` with scope-wide fallback), near-duplicate policy (`on_near_duplicate`, alias `auto_supersede_near_duplicate`), metadata carry mode (`metadata_mode`), optional contradiction check, optional TTL (`expires_at`, ISO 8601); the answer names the `scope`/`project` it wrote to |
| `remember_batch` | Store multiple facts at once (savepoints); `skip_duplicates`, `dedupe`, per-fact `project`/`scope`/`supersede`/`fuzzy_threshold`/`expires_at`/`on_near_duplicate`, optional `check_contradictions`; every entry names its `scope`/`project` |
| `update_memory` | Update content, importance, metadata, entity, type or TTL (`expires_at`, `clear_expires_at`) (re-embeds on content change); a content change is refused as `duplicate` when it would duplicate another active memory (`dedupe`, `fuzzy_threshold`) |
| `update_batch` | Update many memories in one call (savepoints); per entry `memory_id`, `content`, `importance`, `metadata`, `entity`, `mem_type`, `expires_at`, `clear_expires_at`, `project`, `dedupe`, `fuzzy_threshold`; invalid/unknown/duplicate entries are reported individually (`partial`), the rest is still applied |
| `forget` | Soft delete (`deleted_at` + sentinel) or hard delete by `memory_id`, `memory_ids` or entity; `entity` also removes the graph node and that scope's memories; `entity="*"` empties a whole project (soft by default: recoverable rows, the scope itself survives - `drop_project` removes the scope completely); `dry_run` previews exactly what the real call removes; `counts` reports memories/entities/relations separately |
| `restore_memory` | Bring one soft-forgotten memory back, or several with `memory_ids` (per-id results, `partial` when some fail); clears `deleted_at` and the forgotten sentinel; only soft-forgotten rows can be restored |
| `touch_memory` | Bump `updated_at` without changing the content (refreshes the decay recency); optional `importance`; unknown ids answer `not_found` |

Valid `mem_type` values: `fact`, `decision`, `learning`, `preference`,
`pattern`, `incident`, `runbook`, `context`, `observation`, `convention`,
`command`, `note`.

### Read / Search
| Tool | Description |
|------|-------------|
| `recall` | Hybrid vector + FTS search (RRF fusion, project + global tables); `hybrid`, `min_similarity` to drop weak vector matches; optional cognitive decay weighting; `include_superseded` / `only_superseded`; `graph_boost` (with `graph_entity`) to include connected entities; `fields` to narrow columns; `offset` pages through ranked hits; logs `access_count`; reports `candidates` / `dropped_by_similarity` / `dropped_by_coverage` / `dropped_by_decay` / `dropped_by_superseded` / `scopes_searched` |
| `list_memories` | Browse memories without a query (newest first); filters: `entity`, `mem_type`, `project`, `include_global`, `fields`, `contains`, `memory_id` (direct id lookup), `include_deleted` (soft-forgotten rows), `include_expired` (rows past their TTL, marked `expired: true`), `include_relation_context` (hide the edge sentences), `updated_after`, `metadata_filter`, `order_by`; `offset` pages through a scope |
| `get_memory` | Fetch one memory by id, including hidden rows (`include_superseded`, `include_deleted`, `include_expired`); a hidden row explains which flag would reveal it; `project` disambiguates an id that exists in several scopes; `fields` narrows the row |
| `get_entity` | Get entity details + all linked memories + graph connections; `include_global`, `include_superseded`, `fields`, `limit`/`offset`; the answer reports `total`, `relation_context_count`/`has_more` and warns when it was truncated |
| `list_entities` | Browse entities; filters: `entity_type`, `project`, `include_global`; `offset` pages through the list; synthetic relation-context entities (`<from>_<relation>_<to>`) are hidden; `entity_type` selects one kind of entity (a graph node or a memory-only one); `count` is the page size and `total` the full match count |
| `init_context` | Bootstrap: recent memories + entity count + health; global by default, project + global when `project` is set, all scopes with `project="*"` (`projects` then includes `"global"`); `detail="slim"` (default, deduped) or `"full"` |
| `list_projects` | List the known project scopes with their sizes (memories, entities, relations) |

### Graph (Apache AGE)
| Tool | Description |
|------|-------------|
| `relate` | Create directed relation between entities (Cypher); auto-creates entities; `project` selects the project graph |
| `relate_batch` | Create/update many relations in one transaction; per-edge `weight`/`context`/`project`; idempotent (MERGE) |
| `unrelate` | Delete a single directed relation; `keep_context` keeps the edge sentence instead of soft-forgetting it |
| `find_connections` | Traverse graph from entity (outgoing/incoming/both); results carry `direction`; `limit`/`offset` page, the answer reports `total`/`has_more`/`next_offset` |
| `find_paths` | Find shortest paths between two entities (`project`-scoped); edges are traversed in **both** directions, so a path can run against an edge; `from_entity == to_entity` answers a 0-hop path and `max_hops` above 10 is clamped with a warning |

### Maintenance & Advanced
| Tool | Description |
|------|-------------|
| `contradictions_report` | Detect conflicting facts (LLM NLI + vector neighbor pre-filter); per entity or scope-wide without one; each pair names the entity of both facts and `cross_entity`; `resolve="newest"` supersedes the older fact of same-entity pairs only, cross-entity pairs stay active (warning) |
| `export_memories` | Backup as JSON/JSONL: ids, `superseded_by`, TTL (`expires_at`), timestamps, project and graph are preserved; `entity`, `mem_type`, `updated_after` and `metadata_filter` narrow the export; payload in `data`; warns about excluded history and included expired rows |
| `import_memories` | Restore JSON/JSONL (`data` accepts object or string); explicit ids make re-import idempotent; `dry_run` validates and counts without writing; graph is restored from JSON exports |
| `drop_project` | Remove a project scope completely (memory table, registry entry, graph nodes); `dry_run` previews the counts |
| `server_health` | Check Postgres + Inference connectivity; reports `version`, `code_fingerprint` (detects a stale container) and `started_at` |

## MCP Resources
| Resource | Description |
|----------|-------------|
| `memory://stats` | Active/total memory counts, entity/relation counts, health |

## Retrieval

- `recall` searches the project table (if `project` is set) **and** the global
  `memories` table. Project rows win when the same content exists in both
  (dedup via `content_hash`); global facts stay visible from inside a project.
  The dedup only drops an **active** copy in another scope: superseded and
  soft-forgotten rows are history and stay visible, and two active rows in the
  same scope (allowed by `dedupe="off"`) both stay. A superseded fact used to
  disappear from `include_superseded` because an active copy of its text
  existed in the other scope.
- Read scope defaults vary by tool: `recall` and `get_entity` use global scope
  when `project` is omitted; with a project they include that project's and
  global memories by default. `list_memories` and `list_entities` list all
  scopes when `project` is omitted or `project="*"`, and project + global by
  default when a project name is set. `get_entity` rejects `project="*"`: its
  answer describes exactly one entity. `project="global"` means the shared
  scope in every tool: it is not "all scopes". Pass `include_global=false` for
  project-only reads.
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
  it. It is dropped when it covers less than `FTS_MIN_COVERAGE` (0.3) of the
  query, so a one-word match of a four-word query no longer comes back as the
  only result: "per" is NOT a stopword for the english dictionary, so the
  memory "... $500 per month" matched the query "requests per minute limit"
  only through that one function word (coverage 1/4 = 0.25 < 0.3) and is now
  dropped. Hits the vector retriever found are never dropped by this.
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
  `recall` defaults to `min_similarity=0.40`; pass `null` to opt out.
- `min_similarity` filters the **vector** retriever only. A full-text hit is
  kept even when the embedding model scores it badly — that is the case the
  full-text leg exists for (identifiers, file names, foreign words).
  `recall` reports `candidates` (the hits that made it into the ranking),
  `dropped_by_similarity` (the number of vector hits removed, not the number
  kept), `dropped_by_coverage` and `scopes_searched` (the scopes that were
  really searched). `dropped_by_decay` and `dropped_by_superseded` are
  reported whenever the `min_decay_score` / `only_superseded` filter removed
  hits - also when other hits survived, so a non-empty result is never
  quietly short. An empty result names **every** filter that removed
  something (`min_similarity`, `min_decay_score`, `only_superseded`, the
  coverage floor) — not just the first one — so an empty result is never
  silent and never misdiagnosed.
- `_relevance_score` is normalized to 0..1 and combines both signals: the
  cosine similarity of the vector hit and the rank of the full-text hit
  (`_text_score`). A memory that both retrievers found scores higher than
  either signal alone. It is `null` for graph-only hits, which were not
  retrieved by relevance.
- `_scope_boost` is added to `_relevance_score` for a project row, so
  a project memory outranks an equally relevant global one. It is compressed
  with the relevance (`boost = 0.05 × (1 − relevance)`): a strong hit stays
  below 1.0 and comparable, a weak one gets the full 0.05.
- `_final_score` is the ranking score in **0..1**: relevance blended with
  decay plus the graph boost, clamped. Without `include_decay` the base is the
  plain relevance, so `graph_boost=true` cannot *lower* a query hit; the
  70/30 blend only applies when decay was actually requested. It is only
  present with `include_decay=true` or `graph_boost=true`; without them the
  ranking is `_relevance_score`. `_distance` is the raw cosine distance to
  the query.
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
  `metadata_filter` (key/value pairs the metadata must contain; scalar values
  are compared tolerantly, so `{"port": "8001"}` and `{"port": 8001}` both
  match a stored number, while objects/arrays keep the JSON containment
  check);
  `list_memories` additionally accepts `order_by`
  (`updated_at`, `created_at`, `access_count`, `importance`), `contains`
  (case-insensitive literal substring — the right tool for "which memory
  contains exactly X", where the semantic ranking of `recall` is not needed),
  `memory_id` (direct id lookup in the selected scope(s)),
  `include_deleted` (lists soft-forgotten rows, which then carry `deleted_at`
  and the machine-readable `forgotten: true` flag; the `__forgotten__` sentinel
  is hidden from `superseded_by` in those rows; even with `include_deleted`
  the superseded history stays hidden unless `include_superseded=true` is also
  passed — the two flags are orthogonal), `include_expired` (lists rows past
  their TTL, carrying `expired: true`; an expired row is never shown by
  `include_deleted` alone) and
  `include_relation_context` (false hides the relation-context sentences of
  the graph edges, which repeat what the edge already says).
  `recall` accepts `offset` (`has_more`/`next_offset` in the answer).
  `list_memories`, `list_entities` and `get_entity` warn when an `offset`
  beyond the result set returns an empty page instead of leaving it silent.
  `list_memories` computes `total` **after** the cross-scope dedup, so
  `total`, `count`, `has_more` and `next_offset` always describe the same
  row set (a deduped global copy no longer made `total` larger than the
  list).
- `get_memory` fetches exactly one id. Hidden rows are reported as hidden
  (`ok: false`) with the flag that reveals them (`include_superseded`,
  `include_deleted`, `include_expired`) instead of looking absent. An id that
  exists in several scopes is refused and names them; `project` selects one.
- `init_context` returns the **most recently changed** memories of the scope
  (never a decay-ranked mix): a freshly stored fact always shows up, even if
  it was never read before. `limit` controls how many (default 10).
  `detail="full"` returns every column except the internal ones and the
  search score annotations (`_decay_score`, `_final_score`, ...): the
  bootstrap is not a search result, it is sorted by the last update.
  `projects` names every scope the answer covers, including the shared one
  (`"global"`); `active_memories` counts the active rows across the covered
  tables (a cross-scope copy can be counted in both), `total_memories`
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
- `update_memory`/`touch_memory`/`restore_memory`/`forget` locate a
  `memory_id` in every table when `project` is omitted.
  `forget(entity=...)` removes the entity's
  memories from all scopes and deletes its graph node; with `project` it
  touches only that scope's memories and node. Unknown ids/entities return
  `NOT FOUND` - `update_memory` checks the id **before** the
  "nothing to update" validation, so an unknown id never reads like a valid
  one. `dry_run` counts exactly what the real call removes, including
  the graph nodes and relations of every scope when no `project` is given.
- A soft-forgotten memory can be recovered with `restore_memory`: it clears
  `deleted_at` and the forgotten sentinel — never a real `superseded_by` id,
  so a restored memory cannot resurrect next to the fact that replaced it;
  it stays invisible to active reads. `list_memories(include_deleted=true)`
  shows the forgotten rows (with `deleted_at`) so an agent can find the id.
  Hard-deleted rows cannot be restored.
- `remember` warns when the new content matches a **superseded** memory: the
  fact is still stored as active (an intentional revert is legitimate), but the
  response names the old id and its replacement so the fork is visible.
- `remember`/`remember_batch` also handle a **near-duplicate** of an active
  memory (cosine similarity at or above `NEAR_DUPLICATE_THRESHOLD`, 0.75) that
  dedupe did not reject. `on_near_duplicate` selects the action: `warn`
  (default; the fact is stored and the warning names the similar id and the
  measured similarity), `supersede` (the new memory replaces it and the old id
  is listed in `superseded`) or `reject` (`status: duplicate`, nothing is
  stored, `duplicate_mode: "near"`). The check runs under `dedupe="off"` when
  the policy is not `warn`. A one-word change (0.857 for the fake test
  embedding) lands in this window.
- `auto_supersede_near_duplicate=true` is the backwards-compatible alias for
  `on_near_duplicate="supersede"`. In `remember_batch` both the top-level
  policy and the per-fact flag work, also under `dedupe="off"`.
- `supersede_last` resolves the best match **of the entity**; without an
  entity it falls back to the best-matching active memory of the whole scope.
  The entity path replaces from similarity 0.5 on, the scope-wide fallback is
  stricter (0.75, i.e. the near-duplicate floor) because the target may
  belong to another entity. A skipped or scope-wide resolve is always warned.
- A cross-scope supersede is named, never silent: when a project write
  invalidates a fact of the shared `memories` table (the contradiction check,
  or an explicit `supersede` of a global id), the fact disappears from
  **every** scope. The answer carries `cross_scope_superseded` (the ids) and a
  `cross-scope supersede` warning, and a per-entry one in `remember_batch`.
- `include_superseded=true` exposes superseded history but never rows that were
  forgotten (`deleted_at` is set), so soft deletion stays invisible to normal
  reads. `only_superseded=true` in `recall` returns the invalidated facts only.
- `list_entities` hides synthetic relation-context entities
  (`<from>_<relation>_<to>`) that
  exist only to make an edge context searchable; an entity that also has a
  real memory stays listed. `entity_type` selects one kind of entity: "graph"
  (every graph node), "memory-only", or a raw AGE label for exactly one
  scope's nodes. The `type` column is user-facing: 'graph' (raw AGE label in
  `label`) or 'memory-only'; `count` is the page size and `total` the full
  match count.
- The relation-context sentence of an edge is stored under the synthetic
  entity `<from>_<relation>_<to>` (e.g. `Loci_depends_on_PostgreSQL`).
  `get_entity` returns it for both endpoints of the edge,
  and its metadata `weight` follows the edge when the edge is re-weighted.
  `list_memories(include_relation_context=false)` and
  `recall(include_relation_context=false)` hide those sentences.
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

# TTL (`expires_at`)

- `remember`, `remember_batch` (per fact) and `update_memory`/`update_batch`
  accept `expires_at` (ISO 8601, e.g. '2027-01-15' or
  '2026-10-25T12:00:00Z'); an invalid value is a named argument error, never a
  silent no-TTL write. `clear_expires_at=true` removes an existing TTL.
- An expired row is not an active memory anymore: it disappears from
  `recall`/`list_memories`/`get_entity`/`contradictions_report`, is no
  duplicate candidate for a re-store, and is skipped by the dedupe,
  supersede-pool and near-duplicate checks (pooled in `ACTIVE_ROW_SQL`).
- Expired rows are NOT deleted: the row keeps its data (`expires_at`
  visible in read outputs) until it is removed with `forget` or superseded.
  `list_memories(include_expired=true)` lists them with `expired: true`, and
  `get_memory(include_expired=true)` fetches one by id; `include_deleted=True`
  alone never shows them. `export_memories` still exports them (a backup never
  loses rows), and they do not resurface through reads on their own.

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

→ **Change**: The check no longer requires an entity. With an entity, that
entity's memories get the candidate seats first and the leftovers go to
cross-entity neighbours, so a differently tagged memory cannot hide a conflict.
Without an entity all memories of the scope compete; the response sets
`contradictions.cross_entity = true` and warns "no entity set: the
contradiction check compared the fact against the nearest memories across all
entities", so the broader comparison is never silent.

### Path B: `contradictions_report` (Standalone-Tool)

1. All **active** memories of the given entity are loaded (max 200; already
   superseded pairs are done and not re-checked). Without an entity the scan
   is scope-wide: all active memories compete, so pairs across different
   entities are reported too.
2. **Vector self-join**: for each memory, the 20 nearest neighbors are fetched via `ORDER BY embedding <=>`
3. Only the deduplicated candidate pairs are checked via `/nli` (configured LLM) – one call per memory, not per pair. The calls run in a bounded pool (8 workers) and are still reported in candidate order; a scope-wide scan used to hit the client timeout when the calls ran strictly one after another.
4. Pairs with confidence over 0.7 count as a contradiction. Every pair names
   the entity of both facts and a `cross_entity` flag.
5. `resolve="newest"` (opt-in) keeps the most recently updated memory of each
   **same-entity** pair active and marks the older one as superseded; both
   untagged facts count as the same subject. Cross-entity pairs are reported
   but **never auto-resolved** (recency cannot decide between two different
   subjects); they carry `cross_entity: true` and the answer warns
   `resolve skipped for N cross-entity pair(s) ...`; `resolution` names
   `superseded` and `kept` on the resolved pairs only.

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

**NLI prompt**: `/nli` asks the LLM to first decide whether both facts
describe the **same subject** at all; a differing attribute (environment,
host, port, version, date/time, state) or an explicit negation is a
contradiction (0.9-1.0) **only then**, newly added or missing details are
never contradictions. Two facts about different subjects stay neutral even
when their attribute values differ ("staging API on port 9443" vs "inference
service on port 8001" was 0.95 under an earlier prompt that attributed every
field difference to one subject). The `/nli` contract carries optional
`query_entity`/`document_entities` labels that both contradiction paths send;
mismatched labels add a hint to the user prompt to first verify that the texts
name the same concrete item — since the prompt widening of Oct 2026 the hint no
longer pushes toward neutral: when both texts plainly discuss the same named
item (same feature, plan, ticket, service), a differing date/number or a direct
negation is still a contradiction ("FeatureX cannot release before 2027" vs
"FeatureX ships December 2026" under labels `ReleasePlan`/`Roadmap` was 0.0
before and is 0.95 now). Facts about genuinely different subjects (staging API
vs inference service) stay neutral.
Earlier prompts judged "deploys to staging" vs "deploys to production" as
neutral (both could be true) — the widened prompt detects that once the
subject is shared. Near-identical wording alone does not make two facts the
same subject: two instances of the same pattern (two test facts, tickets,
incidents) stay neutral so a schema difference ("unknown id" vs "mismatched
content") is not inflated to a 0.95 contradiction.

**NLI outages are surfaced, never silent**: `/nli` sets an `error` field when
**no** score is a real judgment (provider down, not configured); the MCP client
raises. A provider that returns unparseable JSON is retried once, because that
is a transient format hiccup rather than an outage; the retry is not silent
(a warning is logged) and a second failure is reported like any other error. When only some documents fail it sets `partial_error` and `failed`:
the successful scores are real, the failed ones are 0.0 and
`contradictions_report` reports `scan partially incomplete: ...`. Write tools
keep the fact but report `contradiction check skipped:
NLI unavailable: ...` in `warnings` **and** `contradictions.skipped`;
`contradictions_report` marks the scan as incomplete in `warnings`
("scan incomplete: ... not a clean contradiction scan"). Zero contradiction
scores are therefore always trustworthy.

### Auto-Supersede

If `_check_contradictions` detects a contradiction, the old fact is automatically marked with `superseded_by = new_mem_id` (in both tables: project-specific + global `memories`). The old fact is then invisible to all queries (everything filters `superseded_by IS NULL`).

With `include_superseded=true` in `recall` or `get_entity`, superseded facts can be retrieved again (recognizable by the set `superseded_by` field).

Metadata of replaced memories is not lost by default: keys the new fact does not set are merged into the new memory (explicitly passed metadata always wins); the response lists the carried keys as a warning. Pass `metadata_mode="replace"` to drop the old keys instead of merging them.

## Scope Isolation

- Every fact belongs to exactly one scope: the global `memories` table or a
  per-project table `mem_<suffix>`.
- Project names may contain any characters. Plain lower-case identifiers up to
  40 chars are used verbatim as suffix (backwards compatible). All other names
  are lower-cased, truncated and get a short SHA-1 of the *original* name
  appended (`Project Alpha-2` → `project_alpha_2_484299f9`, `Team` and `team`
  get different suffixes). This keeps names below PostgreSQL's 63-char
  identifier limit, so long names and case variants cannot share a table.
- `global` is reserved: passing `project="global"` means "no project" (global
  scope), it never creates a project of that name.
- A blank or whitespace-only `project` is refused with a named argument error
  (`ok: false`) instead of silently writing to the global scope; data-driven
  call sites report the affected item individually (`remember_batch` the fact,
  `import_memories` the record) instead of failing the whole call.
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
  `project` is omitted. `export_memories` exports all scopes when omitted,
  while an explicit `project="global"` exports only the shared scope (memories
  and graph, like every other tool);
  `import_memories` restores each record to its exported scope. For
  `init_context`, use `project="*"` to include all scopes. `forget(entity=...,
  project=...)` deletes only that scope's node and memories; without `project`
  it removes every scope.
- A **memory id is unique per scope, not globally**: an import copies the id
  into the target scope. `update_memory`, `touch_memory`, `restore_memory` and
  `forget(memory_id=...)` refuse an id that exists in several scopes and name
  the scopes in the error (`scopes`); `project` says which one to use.
  `forget(memory_ids=[...])` is the bulk exception: without `project` it
  applies each id to every scope and reports each one in `bulk_results`.
  `forget(entity=...)` keeps removing the entity from every scope.
- `list_projects` shows every scope with a memory table, with its sizes;
  `drop_project` removes a scope completely (memory table, registry entry,
  graph nodes, mirror rows). A scope created by `relate` gets a memory table
  even without a stored fact. A scope that exists only as graph rows (for
  example from a graph-only import) is not listed by `list_projects`, but
  `drop_project` still removes it, so its nodes cannot stay behind as
  orphans. `forget(entity="*")` only empties a scope, the scope itself
  survives.

## Backup & Restore

- `export_memories` (JSON, version `0.3`) keeps ids, `superseded_by`, TTL
  (`expires_at`), `deleted_at`, timestamps, `access_count`, the project of
  every record and — for JSON — the full graph (entities with scope plus
  relations with weight and scope). JSONL is flat: one record per line, no
  graph. `include_embeddings=true` exports the vectors,
  `include_superseded=true` the history (superseded and soft-forgotten rows),
  `include_graph=false` omits the graph. `entity`, `mem_type`,
  `updated_after` and `metadata_filter` narrow the export. The payload is
  returned in `data`. The default export leaves superseded/forgotten history
  out and **warns** how many rows it excluded; expired rows are included with
  their `expires_at` and an explicit warning, so they cannot silently become
  active again. A forgotten row carries `deleted_at` and
  `superseded_by: null`; the internal `__forgotten__` sentinel is never
  exported.
- `import_memories` restores ids and timestamps, recreates per-project tables
  on demand and upserts entities/relations with MERGE (idempotent). Without
  `project` every record keeps its export scope; with `project` all records go
  into that project. `dry_run=true` validates and counts without writing.
  A record whose id already exists in the target scope is
  always `skipped` (that is what makes a re-import idempotent);
  `skip_duplicates=false` only disables the *content* dedupe for records
  **without** an id. When anything was skipped the answer warns and splits the
  count in `skipped_reasons` (`id_exists`, `duplicate_content`,
  `empty_content`, `invalid`), so an idempotent re-import is distinguishable
  from a backup that silently lost rows. `graph_entities` and
  `graph_relations` count the processed records; `graph_entities_created` and
  `graph_relations_created` count the ones that did not exist before (MERGE
  updates an existing node/edge instead of duplicating it).
  A record with a broken `metadata` value is reported individually (in
  `errors`) and skipped — the rest of the import still runs. `invalid` counts
  those records; they are a subset of `skipped`, so `imported + skipped`
  still equals the number of records.
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

Without a usable LLM configuration `/nli` returns placeholder `0.0` scores
together with an `error`; the MCP client raises instead of treating those
zeros as judgments. `contradictions_report` short-circuits when health reports
that no provider is configured and returns `nli_configured=false` plus a
warning. Write tools keep the fact and warn
(`contradiction check skipped: NLI unavailable: ...`) in `warnings` and
`contradictions.skipped`. `LLM_PROVIDER=none` disables NLI explicitly.

## Tool Parameter Reference

**Important**: Use these exact parameter names when calling tools (verified Oct 2026):

| Tool | Key Parameters | Notes |
|------|---------------|-------|
| `get_entity` | `entity` (`name` kept as alias), `limit`, `offset` | Entity name to fetch details for; the answer reports `total`/`has_more` and warns when it was truncated; `total` also counts the relation-context sentences of the entity's edges; `project="*"` is rejected (the answer describes exactly one entity) |
| `relate` | `from_entity`, `to_entity` (not `from`, `to`), `weight` (0..1) | Source and target entity names; `weight` is the edge strength |
| `relate_batch` | `edges: [{from_entity, to_entity, relation_type, weight?, context?, project?}]` | One transaction, MERGE-based (idempotent); unlike `remember_batch` it is all-or-nothing - one invalid edge aborts the whole call; an empty `edges` list is an error; self-loop edges are stored with a warning |
| `unrelate` | `from_entity`, `to_entity`, `relation_type`, `project`, `keep_context` | Deletes one directed edge; `keep_context=true` keeps the edge sentence instead of soft-forgetting it |
| `recall` | `project` (`"*"` = every scope), `min_similarity` (0..1, default 0.40), `min_decay_score` (0..1), `limit`, `hybrid`, `include_decay`, `include_superseded`, `only_superseded`, `graph_boost`, `graph_entity`, `fields`, `updated_after`, `metadata_filter`, `include_relation_context`, `offset` | Drops weak **vector** hits / stale hits; full-text hits are never dropped by `min_similarity`; `graph_boost` expands from `graph_entity` (fallback `entity`); `include_relation_context=false` hides the edge sentences from the ranking; `limit` caps the ranked result (default 10) and `offset` pages (`has_more`/`next_offset`); an unknown project falls back to the global scope and says so in `warnings`; the answer names every filter that emptied the result and reports `scopes_searched` |
| `list_memories` | `entity`, `mem_type`, `project`, `include_global`, `limit`, `offset`, `fields`, `updated_after`, `metadata_filter`, `order_by`, `contains`, `memory_id`, `include_deleted`, `include_expired`, `include_superseded`, `include_relation_context` | Browse without a query; `contains` is a case-insensitive literal substring; `memory_id` is a direct lookup; `include_deleted` lists forgotten rows with `deleted_at` (superseded rows only with `include_superseded` - the two flags are orthogonal); `include_expired` lists TTL-expired rows with `expired: true`; `offset` pages, the response reports `total`/`has_more`/`next_offset` (all after dedup); without `project` or with `project="*"` every scope is listed, `project="global"` only the shared one |
| `get_memory` | `memory_id`, `project`, `include_superseded`, `include_deleted`, `include_expired`, `fields` | Fetch exactly one id; a hidden row answers `ok: false` and names the flag that reveals it; an id in several scopes is refused with `scopes` |
| `list_entities` | `entity_type`, `project`, `include_global`, `limit`, `offset` | Without `project` or with `project="*"`, lists all scopes; `type` is 'graph' (raw AGE label in `label`) or 'memory-only'; synthetic `<from>_<relation>_<to>` entities are hidden; `entity_type` selects one kind ('graph', 'memory-only', or a raw AGE label); `count` is the page size and `total` the full match count; sorted by name so `offset` is stable |
| `init_context` | `project` (`"*"` = every scope), `include_global`, `detail`, `limit` | every memory reports its `project`; the list holds the most recently changed memories; `projects` names every covered scope (including `"global"`); `graph_entities_count` counts the covered scopes only |
| `list_projects` | `include_empty` | Lists every scope with a memory table, with memories/entities/relations counts; `total_memories` splits into `active`/`superseded`/`forgotten` |
| `drop_project` | `project`, `dry_run` | Removes the scope completely (table, registry, graph nodes, mirror rows); a graph-only scope is removed as well; `global` is refused (use `forget(entity="*", project="global")`) |
| `remember` | `scope: "project" \| "global"`, `dedupe: "exact" \| "semantic" \| "off"`, `fuzzy_threshold` (0..1), `supersede`, `supersede_content`, `supersede_last`, `on_near_duplicate` (`warn`/`supersede`/`reject`), `auto_supersede_near_duplicate`, `metadata_mode` (`merge`/`replace`), `expires_at` (ISO 8601) | `global` stores in `memories` even when `project` is set; the answer names the scope it wrote to; an unknown `supersede` id replaces nothing and warns; `supersede_last` works without an entity (scope-wide best match, stricter topic floor 0.75) |
| `remember_batch` | `facts: [{content, project?, scope?, supersede?, fuzzy_threshold?, dedupe?, on_near_duplicate?, auto_supersede_near_duplicate?, expires_at?, ...}]` | Savepoints; invalid facts (missing `content`, wrong shape) are reported individually, the rest is still stored; every entry names its scope |
| `update_memory` | `memory_id`, `content`, `importance` (0..1), `metadata`, `entity`, `mem_type`, `expires_at`, `clear_expires_at`, `project`, `dedupe` (`exact`/`semantic`/`off`), `fuzzy_threshold` (0..1) | Content changes are re-embedded; a content change that would duplicate another active memory answers `duplicate` with `duplicate_of` and applies nothing (`dedupe="off"` bypasses the check); an id that exists in several scopes is refused (`scopes`); an unknown id returns `not_found` before the "nothing to update" validation |
| `update_batch` | `updates: [{memory_id, content?, importance?, metadata?, entity?, mem_type?, expires_at?, clear_expires_at?, project?, dedupe?, fuzzy_threshold?}]` | Savepoints: an invalid/unknown/ambiguous/duplicate entry is reported in its result entry (`partial`), the rest is still applied; `ok` is false only when nothing could be updated |
| `find_connections` | `entity` (`entity_name` kept as alias), `direction`, `limit`, `offset` | Entity to traverse from; a missing entity names the scope it searched; the answer reports `total`/`has_more`/`next_offset` |
| `contradictions_report` | `entity`, `resolve` (`"none"`/`"newest"`) | Entity optional: without it the scan checks all active memories of the scope (pairs across entities included); `newest` supersedes the older fact of **same-entity** pairs only - cross-entity pairs stay active with `cross_entity: true` and a `resolve skipped` warning; a partially failed NLI reports `nli_failures` (`fact_id`, `neighbor_ids`, `reason`) so the unjudged pairs are named |
| `forget` | `memory_id`, `memory_ids`, `entity` (`entity_name` kept as alias), `soft`, `project`, `dry_run` | `memory_id` and `memory_ids` are mutually exclusive; the bulk form validates each id and reports it in `bulk_results` (without `project` it applies to every scope); `counts` splits memories/entities/relations; `entity="*"` + `project` empties a whole scope (soft by default, hard with `soft=false`; the scope itself survives either way - `drop_project` removes it completely); `dry_run=True` counts exactly what the real call removes |
| `restore_memory` | `memory_id`, `memory_ids`, `project` | Reverses a soft delete; only rows with `deleted_at` set can be restored; `memory_ids` restores several per-id (`partial` when some fail) |
| `touch_memory` | `memory_id`, `importance` (0..1), `project` | Bumps `updated_at` (decay recency) without touching the content; an unknown id answers `not_found` |
| `export_memories` | `format` (`json`/`jsonl`), `project`, `include_embeddings`, `include_superseded`, `include_graph`, `entity`, `mem_type`, `updated_after`, `metadata_filter` | Payload is returned in `data`; graph is only included in JSON exports; filters narrow the records |
| `import_memories` | `data` (object or string), `project`, `format`, `skip_duplicates`, `fuzzy_threshold`, `dry_run` | Without `project` records keep their export scope; `dry_run` validates and counts without writing; an id that already exists in the target scope is always skipped (idempotent re-import), a non-empty `skipped` comes with `skipped_reasons` and a warning |

Every tool parameter carries a description in the JSON schema; the table
above only lists the ones that are easy to get wrong.

Dedupe modes: `exact` (default, content hash), `semantic` (also vector
similarity, threshold `fuzzy_threshold`, default 0.85), `off` (store everything).
A duplicate answer names the mode that matched (`duplicate_mode`) and the
measured similarity. A near-duplicate that dedupe did not reject (similarity
>= 0.75) is stored and reported in `warnings`.

`supersede_last` replaces the memory of an entity that matches the new fact
best (most similar, not the newest) only when the new fact is about the same
topic (cosine similarity ≥ 0.5). Without an entity the best-matching memory of
the whole scope is the target, guarded by the stricter 0.75 near-duplicate
floor (the target may belong to another entity). Otherwise no memory is
replaced, the new fact is stored additionally and the response carries a
warning - replacing an unrelated fact would destroy it silently.

## When to use which tool

- **Session start** → `init_context`
- **Store user fact** → `remember` with `mem_type="preference"`, attach `entity`
- **Search past context** → `recall` with `hybrid=true`, `include_decay=true`
- **Change many facts at once** → `update_batch` (per-entry savepoints, `partial` on bad ids)
- **Browse without semantics** → `list_memories` / `list_entities` (`offset` pages)
- **Connect concepts** → `relate` (auto-creates AGE nodes + edges); many at once → `relate_batch`
- **Undo a relation** → `unrelate`
- **Explore relationships** → `find_connections` or `find_paths`
- **Check for stale/wrong facts** → `contradictions_report`
- **Remove stale data** → `forget` with `soft=true`
- **Refresh an old fact's recency** → `touch_memory` (no content change)
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
