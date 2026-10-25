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
| `status` | `stored`, `created`, `merged`, `duplicate`, `updated`, `deleted`, `soft_forgotten`, `restored`, `not_found`, `conflict`, `partial`, `error`, `ok` |
| `id` | Affected memory id (`null` for every failed call, collections and not-found results; an ambiguous id that exists in several scopes answers `ok:false` with extra keys instead, e.g. `scopes`) |
| `error` | Error message, `null` on success |
| `warnings` | Non-fatal notes (skipped checks, superseded facts, empty results) |

Collections add `count` and `results`; tool-specific payloads add their own
keys (`counts` for `forget`, `superseded` for `remember`, `projects` /
`project_stats` for `init_context`, ...). Export payloads are in `data`.

Failed calls never raise. Invalid arguments (missing, wrong type, invalid
TTL/project) return the same envelope with `ok: false` and an `error` naming
the argument. The schema is the contract: a top-level argument the tool does
not accept (a forgotten or renamed parameter) is refused with
`unknown argument(s) ...; accepted arguments are: ...` instead of being
silently ignored.

## MCP Tools (28 total)

### Write

| Tool | Behaviour and key parameters |
|------|------------------------------|
| `remember` | Store one memory with embedding. `project`, `scope` (`project`/`global`), `entity`, `mem_type`, `importance`, `metadata` (JSON object; a JSON-encoded string of an object is accepted and parsed), `expires_at`; the validity window `valid_from`/`valid_to` (ISO, default now/open; `valid_to <= valid_from` is an argument error); provenance `source`/`agent_id` (trimmed, max 200, empty clears) and `confidence` (0..1); `tags` (normalized, max 32, each 1-64 chars); `dedupe` (`exact`/`semantic`/`off`), `fuzzy_threshold`; `supersede` (id), `supersede_content` (exact text), `supersede_last`; `pinned` (always-include flag, default `false`); `on_near_duplicate` (`warn`/`supersede`/`reject`), alias `auto_supersede_near_duplicate`; `metadata_mode` (`merge`/`replace`); `check_contradictions`; `contradiction_mode` (`supersede`/`warn`), `contradiction_min_confidence` (0.70), `max_contradiction_supersedes` (1); `idempotency_key` (1-128 chars; a retry with the same key in the same scope answers `duplicate` naming the stored row, `idempotent_replay: true`, `duplicate_mode: "idempotency_key"`); `links` (`[{to_entity, relation_type, weight?, context?, from_entity?}]` — creates graph edges in the same step, `from_entity` defaults to `entity`; unknown fields inside an edge object are refused with an argument error). Reports the `scope`/`project` it wrote to and `links` (`status: created/updated/duplicate` per edge). A write that had to create a previously unknown project scope warns (`created a new project scope ...`), so a typo in `project` is visible instead of silently starting a new scope. Malformed links are an argument error before the write; edge failures roll back to a savepoint and warn (`links failed: ...`) while the fact stays stored. |
| `remember_batch` | Store many facts with per-entry savepoints. Top-level `project`, `scope`, `dedupe`, `fuzzy_threshold`, `skip_duplicates`, `on_near_duplicate`, `check_contradictions`, `contradiction_mode`, `contradiction_min_confidence`, `max_contradiction_supersedes`; per fact `content`, `project`, `scope`, `entity`, `mem_type`, `importance`, `metadata` (JSON object or JSON-encoded string), `supersede`, `supersede_content`, `supersede_last`, `dedupe`, `fuzzy_threshold`, `on_near_duplicate`, `contradiction_mode`, `contradiction_min_confidence`, `max_contradiction_supersedes`, `expires_at`, `valid_from`/`valid_to`, `source`/`agent_id`/`confidence`, `tags`, `pinned`, `idempotency_key` (per fact; a retry answers `duplicate` with `idempotent_replay: true`), `links` (the same edge objects as `remember`; created with the fact, reported in `entry["links"]`, edge failures hang off the entry's warnings). A malformed edge is a per-fact prepared failure, not a call-level abort; rejected/duplicate facts store no edges. An unknown per-fact key is refused in that entry (`unknown argument(s) ...; accepted arguments are: ...`) instead of being silently dropped; the rest of the batch is still stored. Invalid facts are reported individually; the rest are stored. `count` is
the number of entries, `succeeded`/`failed` split the outcomes. A duplicate
entry names the blocking row in a dedicated `duplicate_of` key (like
`remember`, so batch entries and single calls stay symmetric). A supersede
carries metadata over like `remember`. `skip_duplicates=false` stores the
copy and warns when identical content is already active. |
| `update_memory` | Change `content` (re-embeds), `importance`, `metadata` (JSON object or JSON-encoded string; `metadata_mode` `replace` default or `merge`; `clear_metadata_keys` removes stored keys), `entity`, `mem_type`, TTL (`expires_at`/`clear_expires_at`), the validity window (`valid_from`/`valid_to`/`clear_valid_to`), provenance (`source`/`agent_id`/`confidence`), tags (`tags` with `tags_mode` `replace` default or `merge`, `clear_tags`) and `pinned`; `project`, `dedupe`, `fuzzy_threshold`. `touch=true` only bumps `updated_at` (decay refresh; combinable with `importance`). `restore=true` undoes a soft delete, clears `deleted_at` and the sentinel, never a real `superseded_by`, and is mutually exclusive with other fields; a known id with nothing to restore answers `conflict`. A content change that would duplicate an active memory answers `duplicate`, applies nothing, and names the blocking row: `id`/`duplicate_of` are the blocking id, `memory_id` the edited one. Unknown ids answer `not_found` before the "nothing to update" check; an id that exists but is soft-forgotten, superseded or expired answers `conflict` with the reason. |
| `update_batch` | Many `updates` with per-entry savepoints (`memory_id` plus the same fields as `update_memory`, including `project`, `touch`, `restore`, `metadata_mode`, `clear_metadata_keys`, `valid_from`/`valid_to`/`clear_valid_to`, `source`/`agent_id`/`confidence`, `tags`/`tags_mode`/`clear_tags`, `pinned`). Invalid, unknown, ambiguous or duplicate entries are reported individually (`partial`); the rest are applied. `count` is the number of entries, `succeeded`/`failed` split the outcomes. |
| `forget` | Remove by `memory_id`, `memory_ids`, `entity`, `metadata_filter` (object, same comparison as `recall`) or `contains` (literal, case-insensitive); `project`, `soft` (default `true`), `dry_run`. Soft delete sets `deleted_at` and a forgotten sentinel. `entity` also removes that scope's graph node and memories; without `project` it touches every scope. `entity="*"` empties the given project only. An entity with only soft-forgotten rows answers `ok` with a `message` instead of `not_found`. Hard deletes clear dangling supersede links. `memory_id` and `memory_ids` are mutually exclusive; `counts` splits memories/entities/relations (`active_memories`/`already_forgotten` are reported for `entity="*"`, so a soft dry run and `drop_project` agree), the answer names the `project` it acted in, and an entity delete without `project` reports `scopes_affected` (with a warning when it touched several scopes). An explicit `project` — including the reserved `"global"`, which normalizes to `None` — bounds the id lookup to that one table: copies of the id in other scopes are neither removed nor counted. Each `bulk_results` entry carries `count` and the `scopes` it touched; with `dry_run=true` its status is `would_soft_forget`/`would_delete` instead of the real outcome. The two selectors remove a matched set instead of ids: they need an explicit `project` (`"global"` for the shared table), are mutually exclusive with ids, `entity` and each other, name the hits in `bulk_results` (`content_preview`, `reason`), and a selector that matches nothing answers `ok` with a "no memories matched" warning rather than `not_found`. |

| `merge_memories` | Fold 2-20 active memories of one scope into one new memory. `memory_ids`, `content` (required), `project`, `entity`, `importance`, `metadata` + `metadata_mode` (`merge` default/`replace`), `mem_type`, `dedupe`/`fuzzy_threshold`, `dry_run`. Sources must be active and live in one scope: without `project` the sources name it and a mixed set answers `error` with `scopes`; an inactive/unknown source answers `conflict` with `invalid` and writes nothing. The merged content is dedupe-checked (`duplicate` names the blocking row); the new fact is written first and every source is then marked `superseded_by` it in the same transaction. `metadata_mode='merge'` combines the sources' metadata (later ids win) without graph-edge keys; `entity`/`mem_type`/`importance` default to the sources' shared value (highest importance). Answers `status: merged` with `id`, `merged_ids`, `superseded`, `count`, `project`; `dry_run=true` answers `would_create`/`would_supersede`. |

Valid `mem_type`: `fact`, `decision`, `learning`, `preference`, `pattern`,
`incident`, `runbook`, `context`, `observation`, `convention`, `command`,
`note`.

### Read / Search

| Tool | Behaviour and key parameters |
|------|------------------------------|
| `recall` | Hybrid vector + FTS search with RRF. `query`, `project`, `include_global`, `limit`, `offset`, `hybrid`; `min_similarity`, `min_relevance` (floor on `_relevance_score`; graph-only hits are never dropped by it), `min_decay_score`, `include_decay`; `include_superseded`, `only_superseded`, `include_deleted`, `include_expired`; `graph_boost`, `graph_entity`, `entity`; `updated_after`, `as_of` (validity window), `tags` (all) / `tags_any` (at least one; mutually exclusive), `source`, `agent_id`, `min_confidence`, `metadata_filter`; `include_relation_context`; `include_pinned`; `fields`, `detail` (`full`/`slim`). Logs `access_count`. Reports `candidates`, `dropped_by_*` (including `dropped_by_cross_scope_duplicate`, `dropped_by_expired` and `dropped_by_validity` with `as_of`), `scopes_searched`, `detail`, `has_more`/`next_offset`, plus `confidence` (`high` >= 0.60, `medium` >= 0.45, `low`, `none`, `graph`) and `best_relevance`; a low-confidence answer warns. Hidden rows carry `forgotten`/`expired` markers. `include_pinned=true` backfills the page with the scope's pinned facts: they satisfy the structural filters but are never dropped by `min_similarity`/`min_relevance`/`min_decay_score`, carry `_pinned: true`, are listed after the ranked hits (displacing the lowest non-pinned hit when the page is full) and are counted in `pinned_included`. |
| `recall_batch` | Run 1-25 independent searches in one call (`queries`), with shared `project`, `include_global`, `limit`, `offset`, `hybrid`, `min_similarity`, `min_relevance`, decay/history/TTL flags, `graph_boost`/`entity`, `mem_type`, `as_of`/`tags`/`tags_any`/`source`/`agent_id`/`min_confidence`, `fields`, `detail`. Each `results` entry carries its own `query`, `count`, filter counters, `warnings` and `scopes_searched`; a failed query (including an empty one) is named in the top-level `warnings` and answered in its own entry instead of aborting the batch. |
| `compile_context` | Assemble one token-bounded context block from ranked memories. `query`, `project`, `include_global`, `max_tokens` (256-32000, default 4000), `limit` (default 50), `entity`, `metadata_filter`, `updated_after`, `as_of`/`tags`/`tags_any`/`source`/`agent_id`/`min_confidence`, `hybrid`, `include_decay` (default `true`), `graph_boost`/`graph_entity`; `include_pinned` (default `true`: pinned facts lead the block and are never dropped by a floor), `min_relevance`/`min_similarity`, `diversity` (MMR weight, default 0.3; `0.0` is a strict no-op) and `max_per_entity` (default None); runs `recall` internally (same scope/filter/dedupe rules) and answers `context`, `items` (each with `pinned`), `used_tokens`, `max_tokens`, `truncated` (`true` when a hit or pin did not fit), `pinned_included`, `dropped_by_relevance`/`dropped_by_similarity`/`dropped_by_diversity`/`dropped_by_entity_cap`, `diversity`, `candidates`, `scopes_searched`; an empty match answers `ok` with an empty context and a warning. With `diversity > 0` near-duplicates (identical content hash or cosine >= 0.95) collapse; a pin is never evicted by `max_per_entity` (it still counts toward it). |
| `list_memories` | Browse without a query, newest first. `entity`, `mem_type`, `project`, `include_global`, `limit`, `offset`, `fields`, `updated_after`, `as_of`/`tags`/`tags_any`/`source`/`agent_id`/`min_confidence`, `metadata_filter`, `order_by` (`updated_at`/`created_at`/`access_count`/`importance`/`pinned`), `contains` (case-insensitive literal), `memory_id`, `memory_ids` (bulk direct lookup, max 200; reports `missing` and a warning, mutually exclusive with `memory_id`; rows that exist but are hidden are listed separately in `hidden` with `reasons`/`reveal_flags`); `include_deleted`, `include_expired`, `include_superseded`, `include_relation_context`, `include_duplicates`. `total`/`has_more`/`next_offset` describe the deduped row set; identical content in another scope is hidden by default and counted in `duplicates_hidden` with a warning (`include_duplicates=true` shows every copy). Reports `scopes_searched` plus `total_by_scope` (before dedup) and `listed_by_scope` (after dedup). With `memory_id` it is the direct lookup: the answer carries the single `memory` (plus a one-item `results`), a hidden row answers `ok: false` and names the flag that reveals it, an id in several scopes is refused with `scopes`, `"*"` searches every scope, and browse filters are reported as ignored. An empty filtered browse warns and names the filters that narrowed it to zero. |
| `get_entity` | Entity details, linked memories and graph connections. `entity` (alias `name`), `project`, `include_global`, `include_relation_context`, `include_superseded`, `include_expired`, `as_of`/`tags`/`tags_any`/`source`/`agent_id`/`min_confidence`, `fields`, `limit`, `offset`. Reports `total`, `relation_context_count`, `has_more`, `scopes_searched`, `memories_by_scope`; `total` follows the same cross-scope dedupe as the returned page and `duplicates_hidden` counts the copies that dedupe hid; `include_relation_context=false` hides the edge sentences (the entity's own memories stay). An unknown project is named in `warnings` like `recall`. When the name exists in several scopes, `details.scopes_with_entity` names each occurrence and a warning names the scope `details` describes. On a miss the not-found answer carries `scopes_with_entity` (the other scopes holding the entity) and the error/warning say to repeat the call with `project=<name>`, so a cross-scope lookup needs no second discovery round trip. `project="*"` is rejected. |
| `get_entities` | Batch form of `get_entity` for 1-25 `entities` (trimmed, duplicates removed, blank names rejected) with `project`, `include_global`, `include_superseded`, `include_expired`, `include_relation_context`, `fields`, `limit_per_entity` (default 50), `offset_per_entity`, plus the same `as_of`/`tags`/`tags_any`/`source`/`agent_id`/`min_confidence` filters. Each `results` entry answers like a single `get_entity` call (same dedupe/paging); an unknown name is a `not_found` entry instead of an abort. Reports `count`, `succeeded`, `failed`; `status: partial` when some names failed, `ok: false`/`not_found` when none was found. `project="*"` is rejected. |
| `list_entities` | Browse entities. `entity_type` (`graph`, `memory-only`, or an existing raw AGE label), `project`, `include_global`, `limit`, `offset`. An unknown `entity_type` is an argument error that names the labels present in the selected scope. Synthetic relation-context entities are hidden. `count` is the page size, `total` the full match count. Sorted by name. |
| `list_projects` | Discover every scope without guessing names. `include_empty` (default `true`), `include_counts` (default `true`). Each row reports `project` (the value to pass to every other tool), `scope` (`global`/`project`), `has_table` and `has_graph`; with counts it also reports memories/total/superseded/forgotten/entities/relations, without counts those keys are omitted entirely. `include_empty=false` keeps scopes with active memories or graph data. `relate` also creates the scope's memory table, so `has_table=false` only appears for legacy graph-only scopes. |
| `init_context` | Bootstrap: most recently changed memories + counts + health. `project` (`"*"` = all scopes), `include_global`, `detail` (`slim`/`full`), `limit` (default 10), `include_empty_projects`. Reports `projects`, `project_stats` (per-scope memories/total/superseded/forgotten/entities/relations), `active_memories`, `total_memories`, `graph_entities_count`. Without an explicit project it summarizes only the global scope and warns which other scopes exist (pass `project="*"` or use `list_projects`). |

### Graph (Apache AGE)

| Tool | Behaviour and key parameters |
|------|------------------------------|
| `relate` | Create a directed edge: `from_entity`, `to_entity`, `relation_type` (letters/digits/`_`), `weight` (0..1), `context`, `project`. Omitting `weight` keeps the weight of an existing edge (1.0 for a new one), so adding a context sentence never resets it. Entities are auto-created; `context` is stored as a relation memory, and repeating the same edge with the same context reuses that memory (`context_reused: true`). |
| `relate_batch` | Many `edges: [{from_entity, to_entity, relation_type, weight?, context?, project?}]` in one transaction. MERGE-based and idempotent. All-or-nothing: one invalid edge aborts the call and the error names its index (`edge 1: ...`); an empty list is an error; self-loops are stored with a warning. |
| `unrelate` | Delete one directed edge (`from_entity`, `to_entity`, `relation_type`, `project`). The edge's context memory is soft-forgotten unless `keep_context=true`. A graph node that the deletion leaves without any edge or memory is removed too (`orphan_nodes_removed`), so `list_entities` does not keep showing an empty node. |
| `find_connections` | Traverse from `entity` (alias `entity_name`), `direction` (`outgoing`/`incoming`/`both`), `relation_type`, `project`, `limit`, `offset`. Each result carries `direction` and the edge's `context` sentence (when one was stored); the answer reports `total`/`has_more`/`next_offset`. |
| `list_relations` | Browse edges without a start entity: `entity` (either endpoint), `relation_type`, `project` (omitted or `"*"` = every scope, `"global"` = shared only), `limit`, `offset`. Sorted by weight, then endpoints. Reports `count`/`total`/`has_more`/`next_offset`; an explicitly named project that has neither a memory table nor graph data is named in `warnings` instead of looking like an empty scope. |
| `find_paths` | Shortest paths between `from_entity` and `to_entity`, `project`, `max_hops` (max 10). Edges are traversed in both directions; each relation carries `relation_type` and the edge's `context` sentence when one was stored. Equal endpoints answer a 0-hop path; `max_hops` above 10 is clamped with a warning. |

Entity names are trimmed; a blank name is an argument error, and a name
containing `$` or a backtick is refused because the Cypher layer reserves
those characters (such a node could be created but never looked up again).
Relation types allow letters, digits and `_` only.

### Maintenance

| Tool | Behaviour and key parameters |
|------|------------------------------|
| `contradictions_report` | NLI-check conflicting active memories. `entity` optional (an entity scan covers that entity's memories **plus untagged neighbours** of the scope, so a tagged/untagged conflict stays visible; cross-entity pairs are reported, never auto-resolved; without it the scan is scope-wide), `project`, `resolve` (`none`/`newest`), `resolve_min_confidence` (default 0.8), `dry_run`, `limit`, `budget_seconds` (wall-clock budget of the WHOLE scan - loading plus judging - default 40s). At most 200 candidate seeds are checked; overflow seeds are named in `nli_failures` (`kind: seed_limit`) with a warning and `limit` only caps the reported pairs. Pairs carry both entities and `cross_entity`. `newest` resolves same-entity pairs only and uses the creation time (a `touch` does not decide the winner); cross-entity pairs, pairs below the confidence floor and pairs without a shared content word (`_shares_subject`) stay active with a named warning, so a noisy NLI hit cannot invalidate an unrelated fact. `dry_run=true` reports `would_resolve` and per-pair `resolution.dry_run` without writing. When the budget runs out the scan stops early, sets `budget_exhausted` and lists every unjudged pair in `nli_failures` (`reason: skipped: NLI time budget exhausted`); `nli_failures_count`/`nli_failure_summary` summarize them without reading every entry (`nli_failure_kinds` splits them into `budget`/`timeout`/`parse`/`outage`/`error`, and each entry carries `kind`), and other NLI failures are named there too. |
| `export_memories` | Backup in `data`. `format` (`json`/`jsonl`), `project`, `include_embeddings`, `include_superseded`, `include_deleted` (default keeps the historical behaviour: `include_superseded=true` carries forgotten rows; `false` leaves soft-forgotten rows out), `include_expired` (default `true`; `false` leaves expired rows out), `include_graph`, `entity`, `mem_type`, `updated_after`, `as_of`/`tags`/`tags_any`/`source`/`agent_id`/`min_confidence`, `metadata_filter`. JSON carries the graph as a section; JSONL carries it as extra `record_type` (`entity`/`relation`) lines, so both formats restore the same state. `entity` narrows the graph too, keeps the other endpoints of the exported edges so the graph stays complete, and includes the relation-context sentences of its edges; relation `context` travels with the edge in both formats; the other memory filters do not narrow the graph and warn. Warns about excluded history and included expired rows. |
| `import_memories` | Restore `data` (object, JSON string, or a list/JSONL string of records). `project`, `format`, `skip_duplicates`, `fuzzy_threshold`, `dry_run`. JSONL `record_type` graph lines are restored through the same idempotent MERGE path (relation `context` included). Keeps ids and timestamps; re-import is idempotent. Skips are split into `skipped_reasons`. Broken metadata is reported per record and skipped. Each record runs in its own savepoint, so one invalid row (a bad embedding, a failed insert) is reported in `errors` and the rest still imports; a malformed JSONL line or broken graph record (unknown `record_type`, relation without endpoints, bad weight) is collected as `skipped_reasons.invalid` (`jsonl line N`) instead of aborting the file. `dry_run` adds `would_create`; dangling supersede links left by a partial backup are cleared with a warning, and self-referencing or internal (`__forgotten__`) `superseded_by` values of imported records are cleared the same way (`repaired_self_references`). A kept supersede link whose target lives in ANOTHER scope (a valid cross-scope supersede that was not re-imported) is reported as `cross_scope_supersede_links` plus a warning naming record and scope, so the rows do not silently stay history-hidden inside the restored scope — if that scope is never restored again, `drop_project` of the missing scope clears the link. A dry run pre-checks the collected links and warns when the real import would clear one (a target that exists nowhere). |
| `verify_integrity` | Find rows hidden by a broken `superseded_by` (target missing or self), report supersede chains and expired active rows. Broken links and chain rows are listed in `results` (`problem` names the kind, chains also carry `chain_to`); `chains` counts them even when `limit` truncates the list; a chain alone adds a warning instead of "no integrity problems found". `project`, `fix` (clear broken links), `limit`. `fix=true` never deletes content or restores a valid supersede, and the counts and `results` then describe the state AFTER the repair (`fixed` carries what was cleared). |
| `drop_project` | Remove a scope completely: memory table, registry entry, graph nodes, mirror rows. `project`, `dry_run`. `global` is refused. A graph-only scope is removed as well. Dangling supersede links of other scopes are cleared. Counts include `active_memories`/`already_forgotten` so a dry run can be compared with `forget(entity="*")`. |
| `prune_memories` | Permanently delete inactive rows only (active memories are never touched). `project` (default: every scope), `older_than_days` (default 30), `include_expired`/`include_forgotten` (default true), `include_superseded` (default false: history is kept), `dry_run` (default true). Removes rows whose `expires_at`/`deleted_at` is older than the cutoff, and clears supersede links that would dangle afterwards. `counts` carries `memories` and per-scope `by_scope`; the message says "Would remove" or "Removed". |
| `server_health` | Check Postgres, inference and NLI configuration. Reports `version`, `code_fingerprint` (detects a stale container), `started_at`, `toolset`, `llm_provider`/`llm_model`, `data_egress` (true when contradiction checks send memory content to a hosted LLM: `groq`/`openai`/`anthropic`), and `startup_warnings` (schema/FTS migrations that failed at startup instead of only logging). Adds a `data egress` warning for a hosted provider. |

`MCP_TOOLSET=core` exposes only the daily read/write/graph surface (16 tools:
`remember`, `remember_batch`, `update_memory`, `forget`, `recall`,
`compile_context`, `list_memories`, `get_entity`, `list_entities`,
`list_projects`, `init_context`, `relate`, `unrelate`, `find_connections`,
`find_paths`, `list_relations`). The default `full` exposes all 28,
including the batch, backup and repair tools.

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
    shared scope. An implicit all-scope `list_memories` browse warns when more
    than one scope was searched, so a merged list is never mistaken for a
    single scope.
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

- `importance` defaults to 0.5, so decay has effect from the first write; pass
  a higher value for facts that must stay prominent.
- Dedupe modes: `exact` (default, content hash), `semantic` (also vector
  similarity at `fuzzy_threshold`, default 0.85), `off`. A duplicate answer
  names `duplicate_mode` and the measured similarity.
- `idempotency_key` (1-128 characters, trimmed) makes a retry safe: the check
  runs before dedupe, supersede and the near-duplicate policy, so a second call
  with the same key in the same scope writes nothing and has no side effects. It
  answers `status: duplicate` with `id`/`duplicate_of` naming the stored row,
  `duplicate_mode: "idempotency_key"` and `idempotent_replay: true` (a batch
  entry carries the same key). A different scope has its own key space; a key
  on an expired, superseded or forgotten row no longer blocks the next write. A
  non-string, empty or over-long key is an argument error.
- Near-duplicate policy applies to an active memory at or above
  `NEAR_DUPLICATE_THRESHOLD` (0.75) that dedupe did not reject:
  `warn` (default: store, warn with id and similarity), `supersede` (replace
  it, list the old id in `superseded`) or `reject` (store nothing,
  `status: duplicate`, `duplicate_mode: "near"`). **Same-entity guard:** when
  the matched memory has a different entity, `supersede`/`reject` are not
  applied; the fact is stored and `remember`/`remember_batch` warn with
  `near-duplicate across entities` (both entities and a content preview).
  Similarity alone never deletes another subject's fact. When the new content
  differs in a number or a negation, the warning calls it a **possible
  contradiction**; an `auto_supersede` additionally says to verify it was an
  intended update (use `check_contradictions=true` for a real NLI verdict).
- `auto_supersede_near_duplicate=true` is an alias for
  `on_near_duplicate="supersede"`. It works in `remember_batch` per fact and at
  the top level, also under `dedupe="off"`.
- Near-duplicate warnings from `update_memory`/`update_batch` suggest an
  executable action instead of the write tool's `supersede=<id>` flag: "supersede
  it in a separate remember call (supersede=X), or rerun this update with
  `dedupe='off'`".
- When the write already carried a supersede directive (`supersede`,
  `supersede_content`) and the stored fact did not replace the near-dup match
  (blocked, gone, or a same-entity guard refusal), the warning appends a
  reconciliation note naming the directive, so a caller never assumes its
  instruction did not exist.
- `supersede`/`supersede_content` replace only an existing active memory; an
  unknown id or unmatched content warns and replaces nothing. Both accept a
  global id from a project write (the replacement hides the global fact in
  every scope and is reported as a cross-scope supersede);
  `supersede_content` matches the target scope first, then the global scope.
- `supersede_last` replaces the best-matching memory, not the newest. With an
  entity it uses that entity's memories from similarity 0.5; without one it
  searches the whole scope at the stricter 0.75 floor. If the topic does not
  match, the new fact is stored additionally and the answer warns. A topic
  guard compares the attribute words (entity, stopwords, numbers and time
  values removed), so a same-template fact about another attribute (port vs
  backup job) is refused even above the similarity floor. Without an entity a
  scope-wide match that belongs to a tagged entity is refused, so a broad
  write cannot silently replace another subject's fact.
- Replaced metadata is carried over by default (`metadata_mode="merge"`);
  explicitly passed keys win. `metadata_mode="replace"` drops the old keys.
  Graph-edge metadata (`relation_context`, `from`, `to`, `relation_type`,
  `weight`) is never carried: a relation-context row's edge fields do not
  describe the replacement fact.
- `mem_type` is optional on `remember`/`remember_batch`. An explicit type wins;
  otherwise a supersede carries the replaced memory's type and a fresh fact
  defaults to `"fact"`.
- A supersede closes the replaced row's validity window
  (`valid_to = COALESCE(valid_to, NOW())`), so `as_of` inside the old window
  still reads the old fact; a soft forget does not touch it.
- A cross-scope supersede is named: `cross_scope_superseded` plus a warning.
  The invalidated global fact disappears from every scope.
- Dedupe only covers the write target. If the exact content is already active
  in another scope, the write still succeeds but returns
  `cross_scope_duplicates` plus a warning: reads hide one of the copies, so the
  fork must not be silent (`remember` and each `remember_batch` entry).
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

## Temporal Validity, Provenance and Tags

- Validity window: `valid_from` (default: write time) and `valid_to` (default:
  open). `as_of` on a read keeps only rows with `valid_from <= as_of` and
  (`valid_to` is null or later than `as_of`). Rows that the other filters
  matched but the window hid are counted in `dropped_by_validity` (only
  reported with `as_of`), so an empty point-in-time read names the time filter.
- A validity window with `valid_to <= valid_from` is an argument error.
  `update_memory(valid_to=..., clear_valid_to=true)` edits it; a supersede
  closes an open window (`valid_to = COALESCE(valid_to, NOW())`), a soft forget
  does not.
- Provenance: `source` and `agent_id` are trimmed and capped at 200 characters
  (empty clears), `confidence` is 0..1 (out of range is an argument error).
  `source`/`agent_id`/`min_confidence` are read filters; a row without a
  confidence is never returned by `min_confidence`.
- Tags: normalized (trimmed, lower-cased, deduplicated), max 32, each 1-64
  characters. `tags` (AND) and `tags_any` (OR) are mutually exclusive read
  filters; `update_memory` merges (`tags_mode="merge"`) or replaces, and
  `clear_tags=true` empties the list.
- All three feature groups are available on every read tool that already
  carries `metadata_filter` (`recall`, `recall_batch`, `compile_context`,
  `list_memories`, `get_entity`, `get_entities`) and on `export_memories`.

## Retrieval

- `recall` searches the project table (when set) and the global table. An
  active copy of the same content in another scope is deduped; history stays
  visible with the matching flags. Identical active copies inside one scope
  (possible with `dedupe="off"`) stay in the result; `list_memories` lists
  them too.
- An exact literal hit is a first-class retriever: an identifier-like token
  (`ZEBRAFISH_TOKEN`, `app.py`, `v1.2.3`) or a quoted phrase is searched as a
  case-insensitive substring and fused with a stronger RRF weight. Such rows
  carry `_exact_match` (`"content"` or `"substring"`) and a relevance floor of
  0.95/0.85, so an exact id outranks a vaguely similar vector hit.
- Vector search is exact on scopes below 10,000 rows: no IVFFlat index is
  built there, so a nearby row is never missed by an unprobed list.
- FTS is OR-based; the text score is multiplied by query coverage, and a
  full-text-only hit below `FTS_MIN_COVERAGE` (0.3) is dropped. `min_similarity`
  filters **vector** hits only; full-text hits survive a bad embedding score.
  `recall` reports `dropped_by_similarity`, `dropped_by_coverage`,
  `dropped_by_decay`, `dropped_by_relevance`, `dropped_by_superseded`,
  `dropped_by_expired`, `dropped_by_validity` (with `as_of`),
  `dropped_by_cross_scope_duplicate`; an empty result
  names every filter that
  removed something. The strongest rejected vector hit is reported
  as `best_dropped_by_similarity` (id, similarity, scope, `content_preview`),
  so a barely-missed fact is visible instead of hidden behind a count; the
  empty-result warning names it and suggests `min_similarity=null`.
- `_similarity` is cosine similarity (0..1). Natural-language matches that
  share vocabulary score 0.6-0.95; vocabulary-shifting paraphrases 0.3-0.55
  (the hard eval pair scored 0.325); verbatim keywords 0.3-0.6.   The model
  card's unrelated-document example scores **0.50-0.61**, so "unrelated is
  below 0.3" is false; the measured noise floor is **0.288**. The default
  `min_similarity` is **0.30**, just above that floor; `null` disables it.
  `min_similarity=0.4` is still available as the old strict setting.
- `min_relevance` (0..1, optional) is the floor on the combined
  `_relevance_score`: it trims an online agent's window server-side, with its
  own `dropped_by_relevance` counter and empty-result reason. Graph-only hits
  carry no relevance score and are never dropped by it. Independently of it,
  a **full-text-only** hit (no vector score) below a relevance of **0.25** is
  dropped as stopword noise; exact literal hits (`_exact_match`) are exempt.
- When the first stage is ambiguous (best `_relevance_score` < **0.85** and
  margin to the runner-up < **0.10**), the top **10** candidates are reranked
  by a local cross-encoder (`mixedbread-ai/mxbai-rerank-base-v1`, Apache-2.0).
  `RERANK_ALWAYS=true` reranks every query; `RERANK_ENABLED=false` disables
  it. `recall` reports `reranked`; a failure adds `rerank_error` and the
  `reranking skipped: ...` warning and keeps the first-stage order.
- Ranking fields:
  - `_rrf_norm` (0..1) and raw `_rrf_score`.
  - `_relevance_score` (0..1): vector similarity + FTS rank, plus
    `_scope_boost` for project rows (max 0.05). `null` for graph-only hits.
    After reranking it is capped, not replaced:
    `min(first_stage, sigmoid(logit + RERANK_LOGIT_BIAS))`, so the reranker
    can demote but never promote a weak first-stage hit.
  - `_rerank_score` (0..1): per-candidate sigmoid of the cross-encoder logit,
    so a score does not depend on how many candidates were reranked; it is a
    relative signal, not an absolute relevance. Only present when reranking
    ran. The reranked block is ordered by the capped relevance, with
    `_rerank_score`/`_rrf_score` as tie-breakers.
  - `_history_penalty` (0.5) marks a superseded, forgotten or expired row that
    was included on purpose. Its relevance is halved, so the current fact wins
    a near-tie while history stays readable when it is the only match.
  - `_final_score` (0..1): relevance blended with decay (70/30) plus graph
    boost. Only present with `include_decay=true` or `graph_boost=true`.
  - `_distance` is raw cosine distance.
  - Graph hits carry `_graph_boost` (0.3 × weight), `_graph_source` and
    `_graph_weight`; a traversed edge's relation-context sentence carries
    `_graph_relation_context: true`.
- `graph_boost=true` traverses one hop, loads up to 5 memories per connected
  entity (including the edge's context sentence) and adds the boost.
  `graph_entity` is the expansion point and leaves the query unfiltered;
  `entity` alone also restricts the search to that entity. If only graph hits
  return, a warning says so. Graph-only hits fill the slots left by query
  hits; they never push a query match out of the page.
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
  an entity-scoped `recall` with `include_relation_context=true` also searches
  the sentences of that entity's edges (its own memories stay first-class);
  `include_relation_context=false` hides them from `recall`/`list_memories`.
- `init_context` returns the most recently changed memories (not a decay
  ranking). `detail="full"` returns all public columns without score
  annotations.
- The FTS dictionary defaults to English and is configurable via
  `FTS_LANGUAGE` (a PostgreSQL dictionary name; invalid values fall back to
  `english`). Index and query side always match; a changed dictionary migrates
  at startup.
- Embeddings are prompt-aware: queries use `mode="query"`, stored content uses
  `mode="document"`. After changing the model or prompts, re-embed:
  `docker compose exec loci-mcp python -m mcp_server.reembed`.

## Cognitive Decay

- Half-life: 90 days.
- `importance` (0..1) and `access_count` (capped at 20) slow decay.
- `include_decay=true` blends recency/frequency into the score; decay is at
  most 30% (`DECAY_WEIGHT = 0.3`), relevance stays at 70%.
- `min_decay_score` (0..1) filters on `_decay_norm`: importance × recency
  × an access factor of 0.9..1.0. A fresh memory with importance 1.0 scores
  0.9, importance 0.5 scores 0.45; a 90-day-old one is roughly half of that.
  The filter runs before the current query's access counters are counted.
  `_decay_score` is the raw, uncapped value.

## Contradictions

- Two paths, both pre-filter with pgvector before calling `/nli`:
  - `remember(check_contradictions=true)` checks the 15 nearest active
    memories. With an entity, that entity's memories get candidate priority
    and the rest go to cross-entity neighbours; without one, all entities
    compete and the answer sets `contradictions.cross_entity = true` plus a
    warning.
  - `contradictions_report` loads up to 200 active memories (entity-scoped or
    scope-wide; an entity scan covers that entity's memories **plus untagged
    neighbours** of the scope, so a tagged/untagged conflict stays visible),
    fetches 6 neighbours per memory per searched scope table
    (`CONTRADICTION_NEIGHBORS`) and NLI-checks the deduplicated pairs in a
    bounded pool (6 workers). Neighbours are ordered by vector distance and
    the most suspicious pairs are judged first. The whole scan runs on one
    wall-clock budget (`budget_seconds`, default 40s or
    `CONTRADICTION_SCAN_BUDGET_SECONDS`): the budget starts at the scan's
    first byte of work (loading candidates, a pgvector search per seed), so a
    load that consumed the budget returns before any NLI call with a warning
    ("candidate loading consumed the NLI time budget ..."), `budget_exhausted`
    and the unjudged pairs named in `nli_failures` instead of pushing the call
    past the client's transport timeout.
- Candidates are not restricted to `mem_type="fact"`, but synthetic
  relation-context sentences (`relation_context=true`) never compete: an edge's
  explanation must not be superseded by the fact it describes.
- A score of 0.7 or higher is a contradiction. Auto-supersede marks the old
  memory with `superseded_by = new_id`; with `resolve="newest"` the older
  memory of each same-entity pair is superseded. "Newest" is the creation
  time, so `update_memory(touch=true)` never changes the winner. Cross-entity
  pairs are never auto-resolved.
- A write's auto-supersede is bounded: only pairs at or above
  `contradiction_min_confidence` (default 0.70) are eligible, at most
  `max_contradiction_supersedes` memories (default 1) are invalidated, and
  `contradiction_mode="warn"` changes nothing. Confirmed pairs beyond the
  budget are listed in `contradictions.kept` with a warning, so one write
  cannot wipe several facts on a single NLI pass. The warning names an
  executable directive (`supersede='<existing_id>'`, one supersede argument
  per kept id, or `contradictions_report(resolve="newest")` for same-entity
  pairs), so an agent can resolve kept pairs on the next write without
  guessing ids.
- NLI judges whether both texts describe the same subject first. A differing
  attribute (environment, host, port, version, date/time, state) or an explicit
  negation is a contradiction only then; added or missing details are not.
  Different subjects stay neutral. Near-identical wording alone does not make
  two pattern instances the same subject.
- NLI outages are surfaced: `/nli` sets `error` when no score is a real
  judgment; the client raises. Unparseable JSON is retried once (logged);
  a reply that only slipped on formatting (a trailing comma or a raw
  control character) is parsed instead of failed. The
  inference side also retries with backoff (3 attempts) on 429/5xx, timeout,
  connection error and Groq's transient `400 json_validate_failed`/`tool_use_failed`,
  then falls back through provider-level `json_schema`/`json_object`/plain. A
   partial failure sets `partial_error`/`failed`; `contradictions_report`
   reports an incomplete scan and names `nli_failures`. On the write side a
   partial failure no longer discards the whole check: the candidates that
   WERE judged still produce contradictions/supersedes, the rest is reported
   as `contradictions.unjudged` (with `existing_id`, `existing_content`,
   `reason`) plus a `contradiction check incomplete` warning. Only a failure
   that leaves nothing judable (a whole outage, or a stale inference
   container that cannot name its failed indices) raises the skip: write
   tools keep the fact and warn (`contradiction check skipped: NLI
   unavailable: ...`) in `warnings` and `contradictions.skipped`, so an
   unjudged pair is never silently read as "no contradiction" (the write-side
   check travels through `nli_detailed`). Zero scores are always trustworthy.
- `contradictions_report` short-circuits when no LLM provider is configured
  and returns `nli_configured=false` plus a warning.
- `/nli` uses the configured LLM provider (`LLM_PROVIDER`: `groq`, `openai`,
  `anthropic`, `ollama`, `openai-compatible`, `auto`, `none`). Inside one
  request the documents are judged concurrently (`NLI_MAX_WORKERS`, default
  4, 1-16), so a multi-document check is not a serial round trip per
  document. OpenAI-style
  providers try `json_schema` + `strict`, then `json_object`, then a plain
  request; Anthropic uses `/messages`.

## Backup & Restore

- `export_memories` (JSON version `0.4`) keeps ids, `superseded_by`, TTL, the
  validity window (`valid_from`/`valid_to`), provenance (`source`/`agent_id`/
  `confidence`), `tags`, `pinned`, `deleted_at`, timestamps, `access_count`,
  project, and
  the graph (entities and weighted relations with scope and relation `context`).
  JSON carries the graph as a section; JSONL carries it as extra `record_type`
  lines (`entity` / `relation`), so both formats restore the same state. An
  `entity` export also carries that entity's relation-context sentences.
  The memory type is written only as `mem_type` (the duplicate `type` key of
  version 0.3 is gone); import still accepts a legacy `type` as a fallback.
- Default export excludes superseded/forgotten history and warns how many rows
  it left out. Expired rows are included with `expires_at` and an explicit
  warning. Forgotten rows carry `deleted_at` and `superseded_by: null`; the
  internal sentinel is never exported. `include_deleted=false` with
  `include_superseded=true` exports invalidated facts without the
  soft-forgotten ones; the default keeps the old behaviour.
- `import_memories` recreates project tables on demand and MERGEs
  entities/relations, so it is idempotent. JSONL `record_type` graph lines take
  the same path (relation `context` included). Records keep ids and timestamps.
  `dry_run=true` only validates and counts.
- A record whose id already exists in the target scope is always skipped;
  `skip_duplicates=false` only disables content dedupe for records without an
  id. A non-empty `skipped` count comes with `skipped_reasons`
  (`id_exists`, `duplicate_content`, `empty_content`, `invalid`). Broken
  metadata is reported in `errors` and skipped; a malformed JSONL line or a
  broken graph record (unknown `record_type`, relation without endpoints, bad
  weight) is collected as `skipped_reasons.invalid` with `jsonl line N`; the
  rest still imports.
- `graph_entities`/`graph_relations` count processed records;
  `*_created` counts records that did not exist before.
- Prefer passing `data` as the JSON string: some clients repack objects into
  `{"item": ...}` (the server tolerates it, but the string is safer).

## Configuration

| Env Var | Default | Description |
|---------|---------|-------------|
| `MEMORY_DATABASE_URL` | `postgresql://memory@localhost:5432/memory` | PostgreSQL connection string |
| `MEMORY_INFERENCE_URL` | `http://localhost:8001` | Inference API base URL |
| `MCP_TRANSPORT` | `stdio` | `stdio`, `sse` or `streamable-http` |
| `MCP_HOST` | `0.0.0.0` | Bind address (`sse`/`streamable-http`) |
| `MCP_PORT` | `8766` | Port (`sse`/`streamable-http`) |
| `MCP_AUTH_TOKEN` | – (required for HTTP) | Shared bearer token; every HTTP request must send `Authorization: Bearer <token>`. A non-loopback HTTP bind refuses to start without it |
| `MCP_ALLOW_INSECURE_HTTP` | `false` | Allow an unauthenticated non-loopback HTTP bind (trusted sandbox only) |
| `POSTGRES_PASSWORD` | – (required by compose) | Password for the bundled Postgres; compose builds the DSN from it |
| `FTS_LANGUAGE` | `english` | PostgreSQL text-search dictionary (invalid values fall back to `english`) |
| `RERANK_ENABLED` | `true` | Local cross-encoder reranking of ambiguous first-stage results |
| `RERANK_MODEL` | `mixedbread-ai/mxbai-rerank-base-v1` | Cross-encoder model; loaded by the inference service |
| `RERANK_MAX_LENGTH` | `256` | Token limit per query/document pair (64-512) |
| `RERANK_TOP_K` | `10` | Candidates rescored when the first stage is ambiguous (2-50) |
| `RERANK_TRIGGER_SCORE` | `0.85` | Best relevance at or above this value is trusted |
| `RERANK_TRIGGER_MARGIN` | `0.10` | Lead over the runner-up at or above this value is trusted |
| `RERANK_ALWAYS` | `false` | Rerank every query instead of only ambiguous ones |
| `RERANK_LOGIT_BIAS` | `1.0` | Offset applied before the sigmoid cap; the raw logits are not calibrated (-5..5) |
| `INFERENCE_DEVICE` | `auto` | `auto` (CUDA when available, else CPU), `cpu`, `cuda`; `cuda` still falls back without a visible device |
| `TORCH_INDEX_URL` | `https://download.pytorch.org/whl/cu126` | Build arg for the inference image; `https://download.pytorch.org/whl/cpu` builds the smaller CPU-only variant |
| `NLI_MAX_WORKERS` | `4` | Documents of one `/nli` request judged concurrently (1-16); higher values cut scan latency but raise provider concurrency |
| `LLM_PROVIDER` | `none` | `groq`, `openai`, `anthropic`, `ollama`, `openai-compatible`, `auto`, `none`. `none` disables NLI; egress to a hosted provider is opt-in (compose passes `none` explicitly). When it is unset, a configured provider key (or `LLM_BASE_URL` / `OLLAMA_BASE_URL`) still selects that provider so existing deployments keep working; a bare install stays on `none` |
| `LLM_MODEL` | provider default | NLI model |
| `LLM_REASONING_EFFORT` | `low` for Groq/gpt-oss | Reasoning budget (`low`/`medium`/`high`; `none` disables). A reasoning model otherwise spends the token budget before the JSON answer and returns empty content |
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
# set POSTGRES_PASSWORD and MCP_AUTH_TOKEN first, e.g.
python -c "import secrets; print(secrets.token_urlsafe(32))"
docker compose up -d

# GPU inference (NVIDIA Container Toolkit required; base compose stays on CPU)
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d

# Smaller CPU-only inference image
TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu docker compose build inference

# After changing embedding model or prompts
docker compose build inference loci-mcp && docker compose up -d
docker compose exec loci-mcp python -m mcp_server.reembed
```

Client configuration examples live in [`docs/clients.md`](docs/clients.md).
