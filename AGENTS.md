# Loci MCP Server — Agent Instructions

Persistent memory MCP server for AI agents, backed by PostgreSQL + pgvector
(vector search) + Apache AGE (graph/Cypher).

This file is the machine-facing contract. Human documentation:
[`docs/tools.md`](docs/tools.md), [`docs/retrieval.md`](docs/retrieval.md),
[`docs/scopes.md`](docs/scopes.md), [`docs/backup.md`](docs/backup.md),
[`docs/configuration.md`](docs/configuration.md),
[`docs/clients.md`](docs/clients.md), [`docs/architecture.md`](docs/architecture.md).

## Response Envelope

Every tool answers with the same envelope:

```json
{"ok": true, "status": "stored", "id": "6dd84bf5092a4850",
 "error": null, "warnings": []}
```

| Key | Meaning |
|-----|---------|
| `ok` | `false` when the call failed, `true` otherwise |
| `status` | `stored`, `created`, `duplicate`, `updated`, `deleted`, `soft_forgotten`, `restored`, `not_found`, `partial`, `error`, `ok` |
| `id` | Affected memory id (`null` for collections and failures) |
| `error` | Error message, `null` on success |
| `warnings` | Non-fatal notes (skipped checks, superseded facts, empty results) |

Collections add `count` and `results`; tool-specific payloads add their own
keys (`counts` for `forget`, `superseded` for `remember`, `projects` /
`project_stats` for `init_context`, ...). Export payloads are in `data`.

Failed calls never raise. Invalid arguments (missing, wrong type, invalid
TTL/project) return the same envelope with `ok: false` and an `error` naming
the argument.

## MCP Tools (21 total)

### Write

| Tool | Behaviour and key parameters |
|------|------------------------------|
| `remember` | Store one memory with embedding. `project`, `scope` (`project`/`global`), `entity`, `mem_type`, `importance`, `metadata`, `expires_at`; `dedupe` (`exact`/`semantic`/`off`), `fuzzy_threshold`; `supersede` (id), `supersede_content` (exact text), `supersede_last`; `on_near_duplicate` (`warn`/`supersede`/`reject`), alias `auto_supersede_near_duplicate`; `metadata_mode` (`merge`/`replace`); `check_contradictions`. Reports the `scope`/`project` it wrote to. |
| `remember_batch` | Store many facts with per-entry savepoints. Top-level `project`, `scope`, `dedupe`, `fuzzy_threshold`, `skip_duplicates`, `on_near_duplicate`, `check_contradictions`; per fact `content`, `project`, `scope`, `entity`, `mem_type`, `importance`, `metadata`, `supersede`, `supersede_content`, `supersede_last`, `dedupe`, `fuzzy_threshold`, `on_near_duplicate`, `expires_at`. Invalid facts are reported individually; the rest are stored. `count` is the number of entries, `succeeded`/`failed` split the outcomes. A supersede carries metadata over like `remember`. |
| `update_memory` | Change `content` (re-embeds), `importance`, `metadata`, `entity`, `mem_type`, TTL (`expires_at`/`clear_expires_at`); `project`, `dedupe`, `fuzzy_threshold`. `touch=true` only bumps `updated_at` (decay refresh; combinable with `importance`). `restore=true` undoes a soft delete, clears `deleted_at` and the sentinel, never a real `superseded_by`, and is mutually exclusive with other fields. A content change that would duplicate an active memory answers `duplicate` and applies nothing. Unknown ids answer `not_found` before the "nothing to update" check. |
| `update_batch` | Many `updates` with per-entry savepoints (`memory_id` plus the same fields as `update_memory`, including `project` and `restore`). Invalid, unknown, ambiguous or duplicate entries are reported individually (`partial`); the rest are applied. `count` is the number of entries, `succeeded`/`failed` split the outcomes. |
| `forget` | Remove by `memory_id`, `memory_ids` or `entity`; `project`, `soft` (default `true`), `dry_run`. Soft delete sets `deleted_at` and a forgotten sentinel. `entity` also removes that scope's graph node and memories; without `project` it touches every scope. `entity="*"` empties the given project only. An entity with only soft-forgotten rows answers `ok` with a `message` instead of `not_found`. Hard deletes clear dangling supersede links. `memory_id` and `memory_ids` are mutually exclusive; `counts` splits memories/entities/relations. |

Valid `mem_type`: `fact`, `decision`, `learning`, `preference`, `pattern`,
`incident`, `runbook`, `context`, `observation`, `convention`, `command`,
`note`.

### Read / Search

| Tool | Behaviour and key parameters |
|------|------------------------------|
| `recall` | Hybrid vector + FTS search with RRF. `query`, `project`, `include_global`, `limit`, `offset`, `hybrid`; `min_similarity`, `min_decay_score`, `include_decay`; `include_superseded`, `only_superseded`, `include_deleted`, `include_expired`; `graph_boost`, `graph_entity`, `entity`; `updated_after`, `metadata_filter`; `include_relation_context`; `fields`. Logs `access_count`. Reports `candidates`, `dropped_by_*`, `scopes_searched`, `has_more`/`next_offset`. Hidden rows carry `forgotten`/`expired` markers. |
| `list_memories` | Browse without a query, newest first. `entity`, `mem_type`, `project`, `include_global`, `limit`, `offset`, `fields`, `updated_after`, `metadata_filter`, `order_by` (`updated_at`/`created_at`/`access_count`/`importance`), `contains` (case-insensitive literal), `memory_id`; `include_deleted`, `include_expired`, `include_superseded`, `include_relation_context`. `total`/`has_more`/`next_offset` describe the deduped row set. With `memory_id` it is the direct lookup: the answer carries the single `memory` (plus a one-item `results`), a hidden row answers `ok: false` and names the flag that reveals it, an id in several scopes is refused with `scopes`, `"*"` searches every scope, and browse filters are reported as ignored. |
| `get_entity` | Entity details, linked memories and graph connections. `entity` (alias `name`), `project`, `include_global`, `include_superseded`, `include_expired`, `fields`, `limit`, `offset`. Reports `total`, `relation_context_count`, `has_more`; `project="*"` is rejected. |
| `list_entities` | Browse entities. `entity_type` (`graph`, `memory-only`, or a raw AGE label), `project`, `include_global`, `limit`, `offset`. Synthetic relation-context entities are hidden. `count` is the page size, `total` the full match count. Sorted by name. |
| `init_context` | Bootstrap: most recently changed memories + counts + health. `project` (`"*"` = all scopes), `include_global`, `detail` (`slim`/`full`), `limit` (default 10), `include_empty_projects`. Reports `projects`, `project_stats` (per-scope memories/total/superseded/forgotten/entities/relations), `active_memories`, `total_memories`, `graph_entities_count`. |

### Graph (Apache AGE)

| Tool | Behaviour and key parameters |
|------|------------------------------|
| `relate` | Create a directed edge: `from_entity`, `to_entity`, `relation_type` (letters/digits/`_`), `weight` (0..1), `context`, `project`. Entities are auto-created; `context` is stored as a relation memory. |
| `relate_batch` | Many `edges: [{from_entity, to_entity, relation_type, weight?, context?, project?}]` in one transaction. MERGE-based and idempotent. All-or-nothing: one invalid edge aborts the call; an empty list is an error; self-loops are stored with a warning. |
| `unrelate` | Delete one directed edge (`from_entity`, `to_entity`, `relation_type`, `project`). The edge's context memory is soft-forgotten unless `keep_context=true`. |
| `find_connections` | Traverse from `entity` (alias `entity_name`), `direction` (`outgoing`/`incoming`/`both`), `relation_type`, `project`, `limit`, `offset`. Each result carries `direction`; the answer reports `total`/`has_more`/`next_offset`. |
| `find_paths` | Shortest paths between `from_entity` and `to_entity`, `project`, `max_hops` (max 10). Edges are traversed in both directions. Equal endpoints answer a 0-hop path; `max_hops` above 10 is clamped with a warning. |

### Maintenance

| Tool | Behaviour and key parameters |
|------|------------------------------|
| `contradictions_report` | NLI-check conflicting active memories. `entity` optional (without it the scan is scope-wide), `project`, `resolve` (`none`/`newest`), `limit`. Pairs carry both entities and `cross_entity`. `newest` resolves same-entity pairs only; cross-entity pairs stay active with a warning. Partial NLI failures are named in `nli_failures`. |
| `export_memories` | Backup in `data`. `format` (`json`/`jsonl`), `project`, `include_embeddings`, `include_superseded`, `include_graph`, `entity`, `mem_type`, `updated_after`, `metadata_filter`. JSON includes the graph; JSONL is flat and warns that `include_graph` was ignored. `entity` narrows the graph too; the other memory filters do not and warn. Warns about excluded history and included expired rows. |
| `import_memories` | Restore `data` (object or JSON string). `project`, `format`, `skip_duplicates`, `fuzzy_threshold`, `dry_run`. Keeps ids and timestamps; re-import is idempotent. Skips are split into `skipped_reasons`. Broken metadata is reported per record and skipped. `dry_run` adds `would_create`; dangling supersede links left by a partial backup are cleared with a warning. |
| `verify_integrity` | Find rows hidden by a broken `superseded_by` (target missing or self), report supersede chains and expired active rows. `project`, `fix` (clear broken links), `limit`. `fix=true` never deletes content or restores a valid supersede. |
| `drop_project` | Remove a scope completely: memory table, registry entry, graph nodes, mirror rows. `project`, `dry_run`. `global` is refused. A graph-only scope is removed as well. Dangling supersede links of other scopes are cleared. |
| `server_health` | Check Postgres, inference and NLI configuration. Reports `version`, `code_fingerprint` (detects a stale container), `started_at`, `toolset`. |

`MCP_TOOLSET=core` exposes only the daily read/write/graph surface (13 tools:
`remember`, `remember_batch`, `update_memory`, `forget`, `recall`,
`list_memories`, `get_entity`, `list_entities`, `init_context`, `relate`,
`unrelate`, `find_connections`, `find_paths`). The default `full` exposes all
21, including the batch, backup and repair tools.

Resource: `memory://stats` returns active/total memory counts, entity/relation
counts and health.

## Scopes

- Every fact belongs to exactly one scope: the global `memories` table or one
  project table. `project="global"` always means the shared scope, never "all
  scopes". A blank project is refused.
- Write default: `scope="project"` when `project` is set, otherwise global.
  `scope="global"` writes to the shared table even inside a project. Both write
  tools report the scope they used.
- Read defaults:
  - `recall`, `get_entity`, `init_context`: global when `project` is omitted;
    project + global when set. `init_context` accepts `"*"` for all scopes.
  - `list_memories`, `list_entities`: all scopes when `project` is omitted or
    `"*"`; project + global when a project is set; `"global"` for only the
    shared scope.
  - `include_global=false` restricts project reads to project rows.
  - An unknown project falls back to global and says so in `warnings`
    (`recall`, `list_memories`, `list_entities`, `init_context`).
- A memory id is unique per scope, not globally. `update_memory` (including
  `touch=true`/`restore=true`) and `forget(memory_id=...)` refuse an id that
  exists in several scopes and name the scopes; pass `project` to select one.
  `forget(memory_ids=[...])` is the bulk exception: without `project` it
  applies each id to every scope and reports each in `bulk_results`.
- `forget(entity=...)` removes the entity and its memories from every scope
  unless `project` is given. `forget(entity="*", project=...)` only empties the
  scope; `drop_project` removes the scope itself.
- `export_memories` exports every scope when `project` is omitted;
  `project="global"` exports only the shared scope. `import_memories` restores
  each record to its exported scope unless `project` overrides it.
- Same entity names can exist in different scopes without collision. Global
  facts stay visible from inside a project.

## Writes, Dedupe and Supersede

- Dedupe modes: `exact` (default, content hash), `semantic` (also vector
  similarity at `fuzzy_threshold`, default 0.85), `off`. A duplicate answer
  names `duplicate_mode` and the measured similarity.
- Near-duplicate policy applies to an active memory at or above
  `NEAR_DUPLICATE_THRESHOLD` (0.75) that dedupe did not reject:
  `warn` (default: store, warn with id and similarity), `supersede` (replace
  it, list the old id in `superseded`) or `reject` (store nothing,
  `status: duplicate`, `duplicate_mode: "near"`).
- `auto_supersede_near_duplicate=true` is an alias for
  `on_near_duplicate="supersede"`. It works in `remember_batch` per fact and at
  the top level, also under `dedupe="off"`.
- `supersede`/`supersede_content` replace only an existing active memory; an
  unknown id or unmatched content warns and replaces nothing.
- `supersede_last` replaces the best-matching memory, not the newest. With an
  entity it uses that entity's memories from similarity 0.5; without one it
  searches the whole scope at the stricter 0.75 floor. If the topic does not
  match, the new fact is stored additionally and the answer warns.
- Replaced metadata is carried over by default (`metadata_mode="merge"`);
  explicitly passed keys win. `metadata_mode="replace"` drops the old keys.
- A cross-scope supersede is named: `cross_scope_superseded` plus a warning.
  The invalidated global fact disappears from every scope.
- Storing content that matches a superseded memory is allowed (an intentional
  revert) but warns with the old id and its replacement.
- `remember` with a project can invalidate a global fact via
  `check_contradictions` or an explicit global `supersede`; this is reported
  per call and per entry in a batch.

## TTL (`expires_at`)

- `remember`, `remember_batch` (per fact), `update_memory` and `update_batch`
  accept `expires_at` (ISO 8601). An invalid value is an argument error, never
  a silent no-TTL write. `clear_expires_at=true` removes the TTL.
- Expired rows are inactive: hidden from `recall`, `list_memories`,
  `get_entity` and `contradictions_report`, and excluded from dedupe,
  supersede and near-duplicate checks.
- Rows are not deleted on expiry. `list_memories(include_expired=true)` shows
  them with `expired: true`; `list_memories(memory_id=...,
  include_expired=true)` fetches one.
  `include_deleted` alone never shows expired rows. Exports keep them with
  `expires_at` and warn; they stay inactive after import.

## Retrieval

- `recall` searches the project table (when set) and the global table. An
  active copy of the same content in another scope is deduped; history stays
  visible with the matching flags.
- FTS is OR-based; the text score is multiplied by query coverage, and a
  full-text-only hit below `FTS_MIN_COVERAGE` (0.3) is dropped. `min_similarity`
  filters **vector** hits only; full-text hits survive a bad embedding score.
  `recall` reports `dropped_by_similarity`, `dropped_by_coverage`,
  `dropped_by_decay` and `dropped_by_superseded`; an empty result names every
  filter that removed something.
- `_similarity` is cosine similarity (0..1). Natural-language matches usually
  score 0.6-0.95, verbatim keywords 0.3-0.6, unrelated facts below ~0.3. The
  default `min_similarity` is 0.40; `null` disables it.
- Ranking fields:
  - `_rrf_norm` (0..1) and raw `_rrf_score`.
  - `_relevance_score` (0..1): vector similarity + FTS rank, plus
    `_scope_boost` for project rows (max 0.05). `null` for graph-only hits.
  - `_final_score` (0..1): relevance blended with decay (70/30) plus graph
    boost. Only present with `include_decay=true` or `graph_boost=true`.
  - `_distance` is raw cosine distance.
  - Graph hits carry `_graph_boost` (0.3 × weight), `_graph_source` and
    `_graph_weight`.
- `graph_boost=true` traverses one hop, loads up to 5 memories per connected
  entity and adds the boost. `graph_entity` is the expansion point and leaves
  the query unfiltered; `entity` alone also restricts the search to that
  entity. If only graph hits return, a warning says so.
- `hybrid=false` uses one retriever (vector if available, otherwise FTS).
  `hybrid=true` is the default and recommended.
- `updated_after` (ISO date) and `metadata_filter` (scalar values compare
  tolerantly, so `"8001"` matches `8001`) narrow `recall` and
  `list_memories`.
- `list_memories` additionally supports `contains`, `memory_id`, `order_by`
  and paging. `offset` beyond the result set warns in `list_memories`,
  `list_entities` and `get_entity`.
- Hidden-row semantics:
  - Superseded history requires `include_superseded=true`;
    `only_superseded=true` returns only invalidated facts.
  - Soft-forgotten rows require `include_deleted=true` and carry `deleted_at`
    plus `forgotten: true`. The `__forgotten__` sentinel is hidden.
  - Expired rows require `include_expired=true` and carry `expired: true`.
  - `include_deleted` and `include_superseded` are orthogonal.
- Results are projected: internal columns (`content_hash`, embedding) are never
  returned. `deleted_at` appears with `include_deleted` or when explicitly
  requested via `fields`; `fields=["id", "content"]` returns exactly those.
- Relation-context sentences are stored under the synthetic entity
  `<from>_<relation>_<to>`. `get_entity` returns them for both endpoints;
  `include_relation_context=false` hides them from `recall`/`list_memories`.
- `init_context` returns the most recently changed memories (not a decay
  ranking). `detail="full"` returns all public columns without score
  annotations.
- The FTS dictionary is English (`FTS_LANGUAGE`). Index and query side always
  match; a changed dictionary migrates at startup.
- Embeddings are prompt-aware: queries use `mode="query"`, stored content uses
  `mode="document"`. After changing the model or prompts, re-embed:
  `docker compose exec loci-mcp python -m mcp_server.reembed`.

## Cognitive Decay

- Half-life: 90 days.
- `importance` (0..1) and `access_count` (capped at 20) slow decay.
- `include_decay=true` blends recency/frequency into the score; decay is at
  most 30% (`DECAY_WEIGHT = 0.3`), relevance stays at 70%.
- `min_decay_score` (0..1) filters on `_decay_norm`: importance × recency plus
  at most a 10% access bonus. A fresh, important memory scores ~0.9; a
  90-day-old one ~0.5. The filter runs before the current query's access
  counters are counted. `_decay_score` is the raw, uncapped value.

## Contradictions

- Two paths, both pre-filter with pgvector before calling `/nli`:
  - `remember(check_contradictions=true)` checks the 15 nearest active
    memories. With an entity, that entity's memories get candidate priority
    and the rest go to cross-entity neighbours; without one, all entities
    compete and the answer sets `contradictions.cross_entity = true` plus a
    warning.
  - `contradictions_report` loads up to 200 active memories (entity-scoped or
    scope-wide), fetches 20 neighbours per memory and NLI-checks the
    deduplicated pairs. Calls run in a small worker pool.
- Candidates are not restricted to `mem_type="fact"`.
- A score of 0.7 or higher is a contradiction. Auto-supersede marks the old
  memory with `superseded_by = new_id`; with `resolve="newest"` the older
  memory of each same-entity pair is superseded. Cross-entity pairs are never
  auto-resolved.
- NLI judges whether both texts describe the same subject first. A differing
  attribute (environment, host, port, version, date/time, state) or an explicit
  negation is a contradiction only then; added or missing details are not.
  Different subjects stay neutral. Near-identical wording alone does not make
  two pattern instances the same subject.
- NLI outages are surfaced: `/nli` sets `error` when no score is a real
  judgment; the client raises. Unparseable JSON is retried once (logged). A
  partial failure sets `partial_error`/`failed`; `contradictions_report`
  reports an incomplete scan and names `nli_failures`. Write tools keep the
  fact and warn (`contradiction check skipped: NLI unavailable: ...`) in
  `warnings` and `contradictions.skipped`. Zero scores are always trustworthy.
- `contradictions_report` short-circuits when no LLM provider is configured
  and returns `nli_configured=false` plus a warning.
- `/nli` uses the configured LLM provider (`LLM_PROVIDER`: `groq`, `openai`,
  `anthropic`, `ollama`, `openai-compatible`, `auto`, `none`). OpenAI-style
  providers try `json_schema` + `strict`, then `json_object`, then a plain
  request; Anthropic uses `/messages`.

## Backup & Restore

- `export_memories` (JSON version `0.3`) keeps ids, `superseded_by`, TTL,
  `deleted_at`, timestamps, `access_count`, project, and for JSON the graph
  (entities and weighted relations with scope). JSONL is one record per line
  with no graph.
- Default export excludes superseded/forgotten history and warns how many rows
  it left out. Expired rows are included with `expires_at` and an explicit
  warning. Forgotten rows carry `deleted_at` and `superseded_by: null`; the
  internal sentinel is never exported.
- `import_memories` recreates project tables on demand and MERGEs
  entities/relations, so it is idempotent. Records keep ids and timestamps.
  `dry_run=true` only validates and counts.
- A record whose id already exists in the target scope is always skipped;
  `skip_duplicates=false` only disables content dedupe for records without an
  id. A non-empty `skipped` count comes with `skipped_reasons`
  (`id_exists`, `duplicate_content`, `empty_content`, `invalid`). Broken
  metadata is reported in `errors` and skipped; the rest still imports.
- `graph_entities`/`graph_relations` count processed records;
  `*_created` counts records that did not exist before.
- Prefer passing `data` as the JSON string: some clients repack objects into
  `{"item": ...}` (the server tolerates it, but the string is safer).

## Configuration

| Env Var | Default | Description |
|---------|---------|-------------|
| `MEMORY_DATABASE_URL` | `postgresql://memory:geheim@localhost:5432/memory` | PostgreSQL connection string |
| `MEMORY_INFERENCE_URL` | `http://localhost:8001` | Inference API base URL |
| `MCP_TRANSPORT` | `stdio` | `stdio`, `sse` or `streamable-http` |
| `MCP_HOST` | `0.0.0.0` | Bind address (`sse`/`streamable-http`) |
| `MCP_PORT` | `8766` | Port (`sse`/`streamable-http`) |
| `LLM_PROVIDER` | `groq` | `groq`, `openai`, `anthropic`, `ollama`, `openai-compatible`, `auto`, `none` |
| `LLM_MODEL` | provider default | NLI model |
| `LLM_API_KEY` | – | Generic key, overrides provider-specific keys |
| `LLM_BASE_URL` | provider default | OpenAI-compatible/Ollama endpoint |
| `GROQ_API_KEY` / `GROQ_MODEL` | – / compose value | Backwards-compatible Groq settings |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | – | Provider keys |
| `OLLAMA_BASE_URL` | `http://host.docker.internal:11434/v1` | Ollama endpoint |

`LLM_PROVIDER=none` disables NLI explicitly.

## Running

```bash
# Local dev
docker compose up -d postgres inference
pip install -r mcp_server/requirements.txt
python -m mcp_server

# Full stack
cp .env.example .env
docker compose up -d

# After changing embedding model or prompts
docker compose build inference loci-mcp && docker compose up -d
docker compose exec loci-mcp python -m mcp_server.reembed
```

Client configuration examples live in [`docs/clients.md`](docs/clients.md).
