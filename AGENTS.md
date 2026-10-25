# Loci MCP Server — Agent Instructions

Persistent memory MCP server for AI agents, backed by PostgreSQL + pgvector
(vector search) + Apache AGE (graph/Cypher).

This file is the machine-facing contract: the rules an agent must follow to
use or change the server correctly. Full parameter and return references live
in the linked documents; this file keeps the operational invariants.

- [`docs/tools.md`](docs/tools.md) — every tool parameter and answer payload
- [`docs/retrieval.md`](docs/retrieval.md) — retrieval, scoring, decay, NLI
- [`docs/scopes.md`](docs/scopes.md) — scope resolution and project lifecycle
- [`docs/backup.md`](docs/backup.md) — export/import contract
- [`docs/configuration.md`](docs/configuration.md) — env vars, auth, LLM,
  Docker, GPU, inference API
- [`docs/clients.md`](docs/clients.md) — client configuration
- [`docs/architecture.md`](docs/architecture.md) — modules, schema, request flow

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

## Tool Surface (29 tools)

- **Write:** `remember` (one fact, optional graph `links`), `remember_batch`
  (many facts, per-entry savepoints), `update_memory`/`update_batch`,
  `merge_memories` (fold 2-20 active facts of one scope), `forget` (soft or
  staged hard delete).
- **Read/search:** `recall`, `recall_batch` (1-25 independent queries),
  `compile_context` (token-bounded block), `list_memories`, `get_entity`,
  `get_entities` (batch across scopes), `list_entities`, `list_projects`,
  `init_context`.
- **Graph:** `relate`, `relate_batch`, `unrelate`, `delete_entity`,
  `find_connections`, `find_paths`, `list_relations`.
- **Maintenance:** `contradictions_report`, `export_memories`,
  `import_memories`, `verify_integrity`, `prune_memories`, `drop_project`,
  `server_health`.

`MCP_TOOLSET=core` exposes the 17 daily read/write/graph tools; the default
`full` exposes all 29. Resource `memory://stats` reports counts and health.

Valid `mem_type`: `fact`, `decision`, `learning`, `preference`, `pattern`,
`incident`, `runbook`, `context`, `observation`, `convention`, `command`,
`note`.

## Scopes

- Every fact belongs to exactly one scope: the shared global `memories` table or
  one project table. `project="global"` always means the shared scope, never
  "all scopes"; a blank project is refused.
- Write default: `scope="project"` when `project` is set, else global.
  `scope="global"` writes to the shared table even inside a project.
  `remember`/`remember_batch` report the scope they used and warn when a write
  had to create a previously unknown project scope, so a typo is visible.
- Read defaults:
  - `recall`, `get_entity`, `get_entities`, `init_context`: global when `project` is omitted;
    project + global when set. `recall` and `init_context` also accept `"*"`.
  - `get_entity`, `get_entities` and `delete_entity` reject `project="*"`.
  - `list_memories`, `list_entities`: all scopes when `project` is omitted or
    `"*"`; project + global when a project is set; `"global"` for only the
    shared scope. An implicit all-scope `list_memories` browse warns when more
    than one scope was searched.
  - `include_global=false` restricts a project read to project rows. An unknown
    project falls back to global and is named in `warnings`.
- A memory id is unique per scope. `update_memory` (also `touch`/`restore`) and
  `forget(memory_id=...)` refuse an id that exists in several scopes and name
  the scopes; pass `project` to select one. An explicit `project` — including
  the reserved `"global"` — bounds the lookup. `forget(memory_ids=[...])` is
  the bulk exception: without `project` it applies each id to every scope and
  reports each in `bulk_results`. A supersede target is the other exception:
  when the id is active in both the write scope and global, the write-scope
  row is replaced, the global row stays active, and a warning names the
  precedence and the untouched row; `cross_scope_superseded` is not set.
- `forget(entity=...)` removes the entity's graph node and memories from every
  scope unless `project` is given; with `project`, only that scope is touched.
  `forget(entity="*", project=...)` only empties that scope. `drop_project`
  removes the scope itself (`dry_run` previews,
  `global` is refused). `forget(query=...)` is a selector and needs an explicit
  `project`.
- `export_memories` exports every scope when `project` is omitted;
  `project="global"` exports only the shared scope. `import_memories` restores
  each record to its exported scope unless `project` overrides it.
- Same entity names can exist in different scopes; global facts stay visible
  from inside a project. Use `list_projects` to discover scopes.

## Write, Dedupe and Supersede

- `importance` defaults to 0.5, so decay applies from the first write.
- Dedupe: `exact` (default, content hash), `semantic` (vector similarity at
  `fuzzy_threshold`, default 0.85) or `off`. A duplicate names
  `duplicate_mode` and the measured similarity. Dedupe only covers the write
  target: identical active content in another scope still stores, but is
  reported as `cross_scope_duplicates` plus a warning. A batch with
  `skip_duplicates=false` stores the copy and warns when identical content is
  already active.
- `idempotency_key` (1-128 chars, trimmed) is checked before dedupe, supersede
  and the near-duplicate policy. A replay in the same scope writes nothing and
  answers `duplicate` with `id`/`duplicate_of`, `duplicate_mode:
  "idempotency_key"` and `idempotent_replay: true`; key spaces are per scope;
  a key on an expired, superseded or forgotten row no longer blocks. Non-string,
  empty or over-long keys are argument errors.
- Near duplicates (active memory at `NEAR_DUPLICATE_THRESHOLD` 0.75):
  `on_near_duplicate` is `warn` (default: store and warn), `supersede` (replace,
  list in `superseded`) or `reject` (store nothing, `duplicate`, mode `near`).
  Same-entity guard: a match on a different entity never triggers
  `supersede`/`reject`; the fact is stored and a `near-duplicate across
  entities` warning names both. A differing number or negation is called a
  possible contradiction and an automatic supersede warns to verify it was
  intended (`check_contradictions=true` gives an NLI verdict). The alias
  `auto_supersede_near_duplicate=true` means `supersede` and also works under
  `dedupe="off"`.
- When a write carried a supersede directive but did not replace the near-dup
  match, the warning names the directive, so a caller never assumes its
  instruction did not exist.
- `supersede` (id), `supersede_content` (exact wording) and `supersede_last`
  replace only an active memory. An unknown id or unmatched content warns and
  replaces nothing. A global id may be superseded from a project write; the
  replacement hides the global fact everywhere and is reported as
  `cross_scope_superseded` plus a warning.
- `supersede_content` searches the target scope first, then global.
  `supersede_last` replaces the best-matching memory, not the newest: entity
  memories from similarity 0.5, scope-wide from 0.75. A topic guard refuses a
  same-template fact about another attribute, and a scope-wide match owned by a
  tagged entity is refused; the new fact is then stored additionally with a
  warning.
- A supersede carries the old metadata by default (`metadata_mode="merge"`,
  explicit keys win; `replace` drops old keys). Graph-edge keys
  (`relation_context`, `from`, `to`, `relation_type`, `weight`) are never
  carried. `mem_type` defaults to the replaced memory's type, else `"fact"`.
- A supersede closes the old validity window (`valid_to =
  COALESCE(valid_to, NOW())`); a soft forget does not. Storing content that
  matches a superseded memory is allowed (an intentional revert) but warns with
  the old id and its replacement. A project write can invalidate a global fact
  via `check_contradictions` or an explicit global `supersede`.
- `update_memory` with content re-embeds. A content change that would duplicate
  an active memory answers `duplicate`, applies nothing, and names the blocking
  row (`id`/`duplicate_of` blocking, `memory_id` edited). An unknown id is
  `not_found` before the nothing-to-update check; an existing but
  soft-forgotten, superseded or expired id is `conflict` with the reason.
  Near-duplicate warnings suggest an executable follow-up (`supersede=<id>` in
  a separate remember call, or `dedupe="off"`).
- `update_memory`/`update_batch` default to `metadata_mode="replace"`;
  `clear_metadata_keys` removes stored keys. `remember` with a supersede
  defaults to `merge`.
- Batch writes (`remember_batch`, `update_batch`) use one savepoint per entry:
  invalid, unknown, ambiguous or duplicate entries are reported individually
  (`partial`, `count`/`succeeded`/`failed`) and the rest still apply. Batch
  entries name a blocking row in `duplicate_of`, symmetric with `remember`. A
  per-fact unknown key is refused in that entry only.
- `merge_memories` folds 2-20 active memories of one scope into a new fact. The
  new fact is written first, then every source is marked `superseded_by` it in
  the same transaction; sources stay untouched on failure. All sources must be
  active and in one scope; merged content is dedupe-checked.

## Graph

- `relate` creates a directed edge (`from_entity`, `to_entity`,
  `relation_type` letters/digits/`_`, `weight` 0..1, `context`). Omitting
  `weight` keeps an existing edge's weight (new edges default 1.0). Entities are
  auto-created; `context` is stored as a relation memory, and repeating the same
  edge/context reuses it (`context_reused`).
- `remember`/`remember_batch` accept `links` to create edges with the fact
  (`from_entity` defaults to the fact's entity). Malformed links are argument
  errors before the write; an edge failure rolls back to a savepoint and warns
  `links failed: ...` while the fact stays. Rejected or duplicate facts store
  no edges.
- `relate_batch` is all-or-nothing in one transaction (an invalid edge names its
  index; self-loops warn). `unrelate` deletes one edge, soft-forgets its context
  unless `keep_context=true`, and removes nodes left without an edge or memory.
- `delete_entity` uses `DETACH DELETE` (neighbours survive); `remove_memories`
  defaults false, `remove_relations` true (`false` is `conflict` while edges
  exist), `dry_run` is available. A name that only exists on memories is
  accepted only with `remove_memories=true`.
- `find_connections` reports direction and edge context with paging;
  `find_paths` traverses both directions, clamps `max_hops` to 10 with a
  warning and answers a 0-hop path for equal endpoints. `list_relations` browses edges across
  endpoints and scopes.
- Names are trimmed; blank names are argument errors; `$` or a backtick is
  refused (reserved by Cypher). Relation types allow letters, digits and `_`
  only.

## Deletion, Restore and Integrity

- `forget` selectors are `memory_id`, `memory_ids`, `entity`, `metadata_filter`,
  `contains` or semantic `query`; `soft=true` (default) hides via `deleted_at`
  and the forgotten sentinel. `memory_id` and `memory_ids` are mutually
  exclusive.
- `metadata_filter`, `contains` and `query` need an explicit `project`
  (`"global"` for the shared table), are mutually exclusive with ids, `entity`
  and each other, and answer `ok` with a "no memories matched" warning when
  nothing matches. `query` searches active rows only, does not bump
  `access_count`, keeps only vector matches at `min_similarity` (default 0.80;
  `limit` default 20), prints each hit in `bulk_results` with `content_preview`
  and `similarity` (`null` without a vector score), and never selects a
  keyword-only full-text hit.
- `entity` also removes that scope's graph node and memories; without
  `project` it touches every scope and reports `scopes_affected`. An entity
  with only soft-forgotten rows answers `ok` with a message, not `not_found`.
- A hard delete (`soft=false`) is staged: it sets `deleted_at` and
  `hard_deleted_at`, keeps status `deleted`, reports `undo_until`/`purge_after`,
  and stays restorable until `prune_memories` after the grace (default 7 days).
  `update_memory(restore=true)` revives a soft or recoverable hard delete,
  clears `deleted_at`/`hard_deleted_at`/the sentinel, never clears a real
  `superseded_by`, and is exclusive with other update fields; an existing id
  with nothing to restore answers `conflict`. Hard deletes clear dangling
  supersede links.
- `dry_run` reports `would_soft_forget`/`would_delete` and exactly the counts
  the real call would change, including `active_memories`/`already_forgotten`
  for `entity="*"`. `counts` splits memories/entities/relations.
- `prune_memories` only removes inactive rows and requires exactly one of
  `project` or `all_scopes=true`; `dry_run` defaults to true. It never touches
  active rows and clears supersede links that would dangle.
- `verify_integrity` finds rows hidden by a missing or self `superseded_by`,
  plus chains and expired active rows; `fix=true` clears only broken/self links
  and never deletes content or restores a valid supersede. After a fix, counts
  and `results` describe the repaired state.

## TTL, Validity, Provenance and Tags

- `expires_at` (ISO 8601) is accepted by `remember`, `remember_batch` (per
  fact), `update_memory` and `update_batch`; invalid values are argument
  errors. `clear_expires_at=true` removes the TTL.
- Expired rows are inactive: hidden from `recall`, `list_memories`,
  `get_entity` and `contradictions_report`, and excluded from dedupe, supersede
  and near-duplicate checks. They are not deleted; `include_expired=true`
  reveals them with `expired: true` (`include_deleted` alone never does).
  Exports keep them with `expires_at` and warn; they stay inactive after import.
- Validity window: `valid_from` (default write time), `valid_to` (default open).
  `as_of` keeps only rows with `valid_from <= as_of` and (`valid_to` null or
  later than `as_of`); rows matched by other filters but hidden by the window
  are counted in `dropped_by_validity` (reported only with `as_of`). A window
  with `valid_to <= valid_from` is an argument error; update/clear via
  `valid_to`/`clear_valid_to`.
- Provenance: `source`/`agent_id` are trimmed, max 200 chars, empty clears;
  `confidence` is 0..1 (out of range is an argument error). `source`,
  `agent_id` and `min_confidence` are read filters; a row without confidence is
  never returned by `min_confidence`.
- Tags: normalized (trimmed, lower-cased, deduplicated), max 32, each 1-64
  chars. `tags` (AND) and `tags_any` (OR) are mutually exclusive. Updates use
  `tags_mode` `replace` (default) or `merge`; `clear_tags=true` empties.
- These filters exist on every read tool that carries `metadata_filter`
  (`recall`, `recall_batch`, `compile_context`, `list_memories`, `get_entity`,
  `get_entities`) and on `export_memories`.

## Retrieval

- `recall` searches the project table (when set) and global. An active copy of
  the same content in another scope is deduped; history stays visible with its
  flags. Identical active copies inside one scope (possible with
  `dedupe="off"`) stay.
- Exact literals are a first-class retriever: an identifier-like token or quoted
  phrase is matched case-insensitively as a substring, fused with a stronger RRF
  weight, and marked `_exact_match`. Exact hits outrank a vague vector match
  while `_relevance_score` stays the genuine vector+FTS blend. `hidden_exact_match`
  names the flag when the literal exists only in a hidden row.
- Pinned facts (`pinned=true`) are the always-include set. `recall`
  (`include_pinned=true`) and `compile_context` (default) backfill them after
  the ranked hits, displacing the lowest non-pinned hit when the page is full;
  structural filters apply, but `min_similarity`,
  `min_relevance` and `min_decay_score` never drop a pin. A pin carries
  `_pinned: true`, `_relevance_score` 0.0 and is counted in `pinned_included`.
- Vector search is exact below 10,000 rows per scope (no IVFFlat index).
- FTS is OR-based with a query-coverage multiplier; a full-text-only hit below
  `FTS_MIN_COVERAGE` 0.3 is dropped. `min_similarity` filters vector hits only,
  so an exact keyword hit survives a bad embedding. `min_relevance` floors the
  combined score; graph-only hits are exempt, and a full-text-only hit below
  0.25 relevance is dropped unless it is an exact literal.
- Similarity: natural-language matches 0.6-0.95, vocabulary-shifting
  paraphrases 0.3-0.55, verbatim keywords 0.3-0.61. The measured noise floor is
  0.288, so `min_similarity` defaults to 0.30 (`null` disables; 0.4 is the old
  strict setting).
- Ambiguous first-stage results (best relevance < 0.85 and lead < 0.10) are
  reranked locally (`mixedbread-ai/mxbai-rerank-base-v1`, top 10). `reranked`
  reports it; a failure sets `rerank_error` and keeps the first-stage order.
  `RERANK_ALWAYS`/`RERANK_ENABLED` override the trigger.
- Key scores: `_similarity` (cosine), `_relevance_score` (vector+FTS+scope,
  `null` for graph-only), `_rerank_score`, `_history_penalty` (0.5 for history
  included on purpose), `_final_score` (relevance+decay+graph, only with
  `include_decay`/`graph_boost`), `_graph_boost`. Full field table:
  [`docs/retrieval.md`](docs/retrieval.md).
- `graph_boost=true` traverses one hop, loads up to 5 memories per connected
  entity (including edge context) and adds `0.3 × weight`. `graph_entity` is
  the expansion point and leaves the query unfiltered; `entity` alone also
  restricts the search. Graph-only hits fill leftover slots and never displace a
  query hit.
- `hybrid=false` uses one retriever (vector if available, else FTS).
  `updated_after` and `metadata_filter` narrow `recall`/`list_memories`; scalar
  metadata compares tolerantly (`"8001"` matches `8001`).
- Hidden-row semantics: superseded needs `include_superseded` (or
  `only_superseded` for only history), forgotten needs `include_deleted`
  (`forgotten: true`, `deleted_at`; the `__forgotten__` sentinel is hidden),
  expired needs `include_expired` (`expired: true`). The flags are orthogonal.
- Results are projected: internal `content_hash`/embedding are never returned;
  `detail="slim"` trims rows to session fields and an explicit `fields` list
  wins (`fields=["id","content"]` returns exactly those). Relation-context sentences
  live under `<from>_<relation>_<to>`; `get_entity` returns them for both
  endpoints, and with `entity`, `recall` also searches the relation-context
  sentences of that entity's edges; `include_relation_context=false` hides them.
- `recall` reports `candidates`, `scopes_searched`, `has_more`/`next_offset`,
  `confidence` (`high` >= 0.60, `medium` >= 0.45, `low`, `none`, `graph`) and
  `best_relevance`; a low-confidence answer warns. `dropped_by_*` counters
  (similarity, coverage, decay, relevance, superseded, expired, validity,
  cross-scope duplicate) explain an empty or short result; the strongest
  rejected vector hit is `best_dropped_by_similarity`. With `only_superseded`,
  dropped active rows are counted in `dropped_by_not_superseded` while
  `dropped_by_superseded` stays 0.
- `recall_batch` runs independent queries; a failed or empty query is reported
  in its own entry and named in the top-level `warnings`, never aborting the
  batch.
- `compile_context` packs `recall` hits into a token-bounded block.
  `max_tokens` is 256-32000 (default 4000), `include_pinned` defaults true,
  `diversity` is the MMR weight (default 0.3; 0.0 is a no-op) and
  `max_per_entity` is optional. With diversity, near-duplicates (identical hash
  or cosine >= 0.95) collapse; a pin is never evicted. `truncated` and
  `dropped_by_*` counters report omissions; an empty match is `ok` with a
  warning.
- `list_memories` adds `contains`, direct `memory_id`/`memory_ids` lookup,
  `order_by`, paging and `include_duplicates`. `memory_id` is a direct lookup;
  a hidden row answers `ok:false` and names the revealing flag (an ambiguous id
  is refused with `scopes`). `memory_ids` is the bulk form: hidden rows are
  listed in `hidden`, missing ids in `missing`, and ambiguous ids in `errors`;
  browse filters are reported as ignored. Cross-scope duplicates are
  hidden by default (`duplicates_hidden`); `total` follows that dedupe and
  `total_by_scope`/`listed_by_scope` split before/after. An empty filtered
  browse warns and names the filters that produced zero.
- `get_entity` reports `details`, `memories_by_scope` (an exact per-scope
  count), `total`, `duplicates_hidden` and deterministic
  `details.connections`. `scopes_with_entity` names other scopes on a hit or
  miss, so repeat with `project=<name>`. `get_entities` runs 1-25 lookups,
  allows a per-entry project, returns `partial` with per-entry
  `not_found`/`error`, and rejects `"*"`.
- `list_entities` hides synthetic relation-context entities; an unknown
  `entity_type` is an argument error naming the labels present; `count` is the
  page size, `total` the full match count. `offset` beyond the result set warns
  in `list_memories`, `list_entities` and `get_entity`.
- `init_context` returns the most recently changed memories, not a decay
  ranking. Without a project it summarizes global only and warns which other
  scopes exist (`project="*"` or `list_projects`); `project_stats` covers
  per-scope memories/total/superseded/forgotten/entities/relations.
- The FTS dictionary is `FTS_LANGUAGE` (default `english`; invalid values fall
  back) and migrates at startup. Embeddings are prompt-aware (`mode="query"`
  vs `"document"`); after changing model or prompts run
  `docker compose exec loci-mcp python -m mcp_server.reembed`.

## Cognitive Decay

- Half-life is 90 days. `importance` (0..1) and `access_count` (capped at 20)
  slow the decay; `recall` bumps `access_count` for its hits, and
  `update_memory(touch=true)` restarts recency without a content change.
- `include_decay=true` blends at most 30% decay with 70% relevance.
- `min_decay_score` (0..1) filters `_decay_norm` = importance × recency × an
  access factor of 0.9..1.0. A fresh memory with importance 1.0 scores 0.9
  (importance 0.5 scores 0.45); a 90-day-old one is roughly half. The filter
  runs before the current query's access counters.

## Contradictions and NLI

- Two paths, both vector-prefilter before `/nli`:
  `remember(check_contradictions=true)` checks the 15 nearest active memories
  (an entity's memories get priority; without an entity all entities compete
  and `contradictions.cross_entity` is set with a warning), while
  `contradictions_report` loads up to 200 active seeds (an `entity` scan also
  covers untagged neighbours), fetches 6 neighbours per seed per scope table and
  judges the deduplicated pairs in a bounded pool.
- Candidates include every `mem_type`, but synthetic relation-context sentences
  never compete.
- A score >= 0.7 is a contradiction. Write-side auto-supersede is bounded: only
  pairs >= `contradiction_min_confidence` (default 0.70), at most
  `max_contradiction_supersedes` (default 1), and `contradiction_mode="warn"`
  changes nothing. Confirmed pairs beyond that are listed in
  `contradictions.kept` with an executable directive.
- `contradictions_report(resolve="newest")` supersedes the older memory of each
  same-entity pair by creation time (`touch` does not change the winner).
  Cross-entity pairs, cross-scope pairs, pairs below `resolve_min_confidence`
  (0.80) and pairs without a shared content word stay active with a warning
  (`cross_scope: true` names a project/global pair); a same-scope resolve
  updates only that pair's table through the scope-aware supersede helper, so
  it closes `valid_to` too. `dry_run` reports `would_resolve` without writing.
- The whole report scan runs on `budget_seconds` (default 40s), covering
  candidate loading and NLI. On exhaustion it stops, sets `budget_exhausted`
  and lists unjudged pairs in `nli_failures`; `nli_failures_count`/
  `nli_failure_summary` and `nli_failure_kinds`
  (`budget`/`timeout`/`parse`/`outage`/`error`) summarize them. The 200-seed cap
  names overflow seeds as `kind: seed_limit`; `limit` only caps reported pairs.
- NLI first decides whether both texts describe the same subject. A differing
  attribute (environment, host, port, version, date/time, state) or an explicit
  negation is a contradiction only then; added or missing details are not.
- NLI failures are never "no contradiction": a response without a real judgment
  sets `error` and the client raises. Unparseable JSON is retried once
  (formatting slips are repaired), and provider calls retry and fall back. A
  partial write-side failure keeps judged verdicts and reports the rest as
  `contradictions.unjudged` plus a `contradiction check incomplete` warning;
  only a failure with nothing judable skips the check, keeps the fact and warns
  (`contradiction check skipped: NLI unavailable: ...`) in `warnings` and
  `contradictions.skipped`. Zero scores from a successful run are trustworthy.
- `contradictions_report` short-circuits without a configured provider
  (`nli_configured=false` plus a warning). Provider selection and worker counts
  are in [`docs/configuration.md`](docs/configuration.md); `/nli` judges
  documents concurrently (`NLI_MAX_WORKERS`, default 4).

## Backup & Restore

- `export_memories` (JSON version 0.4) keeps ids, `superseded_by`, TTL, validity
  window, provenance, tags, `pinned`, delete markers, timestamps,
  `access_count`, project and the graph (entities and weighted relations with
  scope and relation `context`). JSON carries the graph as a section; JSONL
  carries it as extra `record_type` (`entity`/`relation`) lines, so both formats
  restore the same state.
- The default export excludes superseded/forgotten history and warns. Expired
  rows are included by default (`include_expired=true`) with `expires_at` and a
  warning; they stay inactive after import. `include_deleted=false` with
  `include_superseded=true` exports invalidated facts without soft-forgotten
  ones. Forgotten rows carry `deleted_at` and `superseded_by: null`; the
  internal sentinel is never exported.
- `import_memories` recreates project tables on demand, MERGEs graph records,
  keeps ids and timestamps, and is idempotent. A record whose id exists in the
  target scope is always skipped; `skip_duplicates=false` only disables content
  dedupe for id-less records. Each record has its own savepoint: bad rows are
  reported in `errors` and the rest import; malformed JSONL/graph records are
  `skipped_reasons.invalid` with `jsonl line N`.
- Import clears dangling `superseded_by` links and self/internal
  (`__forgotten__`) references (`repaired_self_references`). A kept link whose
  target lives in another scope is reported as `cross_scope_supersede_links`
  plus a warning; `drop_project` of that missing scope clears it. A dry run
  warns when the real import would clear a link; `dry_run` adds `would_create`
  and writes nothing.
- Prefer passing `data` as the JSON string (some clients repack objects). Full
  contract: [`docs/backup.md`](docs/backup.md).

## Configuration, Running and Health

- Environment variables, defaults, HTTP auth, LLM providers, Docker, GPU and
  the inference API are documented in
  [`docs/configuration.md`](docs/configuration.md); client setups are in
  [`docs/clients.md`](docs/clients.md). Operational defaults:
  - `LLM_PROVIDER=none` by default, so NLI/contradiction checks are off until a
    provider is configured. `groq`, `openai` and `anthropic` send memory content
    to a hosted service; `data_egress` and a warning expose that.
  - `MCP_AUTH_TOKEN` is required for a non-loopback HTTP bind;
    `MCP_ALLOW_INSECURE_HTTP=true` is sandbox-only.
  - `HARD_DELETE_GRACE_DAYS` defaults to 7; `prune_memories`'s
    `hard_delete_grace_days` wins.
  - `FTS_LANGUAGE` defaults to `english`; invalid values fall back to it.
- `server_health` reports Postgres/inference/NLI status, `version`,
  `code_fingerprint` (stale-container detection), `started_at`, `toolset`,
  `llm_provider`/`llm_model`, `data_egress` and `startup_warnings`.
- After changing the embedding model or prompts, rebuild and re-embed:
  `docker compose build inference loci-mcp && docker compose up -d`, then
  `docker compose exec loci-mcp python -m mcp_server.reembed`.
