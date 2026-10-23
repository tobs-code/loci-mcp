# Tool Reference

Every tool of the Loci MCP server: what it does, which parameters it takes and
what it answers with. The short version is in the [README](../README.md); this
page is the contract.

> For agents: [`AGENTS.md`](../AGENTS.md) is the machine-facing version of this
> document. It carries the same semantics in a denser form.

## Contents

- [Response envelope](#response-envelope)
- [Write tools](#write-tools)
- [Read tools](#read-tools)
- [Graph tools](#graph-tools)
- [Maintenance tools](#maintenance-tools)
- [Resource](#resource)
- [Typical workflows](#typical-workflows)

## Response envelope

Every tool answers with the same shape, so an agent never has to guess:

```json
{"ok": true, "status": "stored", "id": "6dd84bf5092a4850",
 "error": null, "warnings": []}
```

| Key | Meaning |
|-----|---------|
| `ok` | `false` when the call failed, `true` otherwise |
| `status` | `stored`, `created`, `duplicate`, `updated`, `deleted`, `soft_forgotten`, `restored`, `not_found`, `conflict`, `partial`, `error`, `ok` |
| `id` | Id of the affected memory (`null` for collections and for failures) |
| `error` | Error message, `null` on success |
| `warnings` | Non-fatal notes (skipped checks, superseded facts) |

- Collection tools add `count` and `results`.
- Tools with extra payload add their own keys: `counts` (`forget`),
  `superseded` and `contradictions` (`remember`), `recent_memories`
  (`init_context`), `total`/`has_more`/`next_offset` (paging tools).
- Export and import payloads are in `data`; collection results are in `results`.
- **Failed calls never raise.** An invalid call (missing argument, wrong type)
  answers with the same envelope (`error` names the argument) instead of a raw
  validation error.
- **The schema is the contract.** An argument the tool does not accept (a
  forgotten or renamed parameter, e.g. `only_superseded` on `list_memories`)
  is refused with `unknown argument(s) ...` and the list of accepted names,
  not silently ignored.

## Write tools

| Tool | Description |
|------|-------------|
| `remember` | Store a fact/decision/learning (auto-embedding, `dedupe` mode, `supersede`/`supersede_content`/`supersede_last`, `on_near_duplicate`, `metadata_mode`, optional contradiction check, `links` for graph edges in the same write, `scope` global/project) |
| `remember_batch` | Store multiple facts (savepoints; top-level `skip_duplicates`; per fact: `project`, `scope`, `supersede`, `supersede_content`, `supersede_last`, `on_near_duplicate`, `auto_supersede_near_duplicate`, `expires_at`, `fuzzy_threshold`, `dedupe`; optional `check_contradictions`) |
| `update_memory` | Update content (re-embedding), importance, metadata, entity, type or TTL (`expires_at`/`clear_expires_at`); `touch=true` refreshes the decay recency without a content change, `restore=true` undoes a soft delete |
| `update_batch` | Update many memories in one call (savepoints); per entry `memory_id`, `content`, `importance`, `metadata`, `metadata_mode`, `clear_metadata_keys`, `entity`, `mem_type`, `expires_at`, `clear_expires_at`, `touch`, `restore`, `project`; bad entries are reported individually (`partial`), the rest is still applied |
| `forget` | Soft delete (`deleted_at` + sentinel) or hard delete by `memory_id`, `memory_ids` or entity; `entity` also removes the graph node and that scope's memories; `counts` splits memories/entities/relations |

### `remember`

| Parameter | Type | Meaning |
|-----------|------|---------|
| `content` | string | The fact to store (non-empty). |
| `mem_type` | enum | `fact` (default), `decision`, `learning`, `preference`, `pattern`, `incident`, `runbook`, `context`, `observation`, `convention`, `command`, `note`. |
| `entity` | string? | Entity this memory belongs to; connects it in the graph and enables `graph_boost` / `find_connections`. |
| `importance` | number | 0.0-1.0 (default 0.5). Scales the decay weight, so low-importance memories fade faster. Raise it for facts that must stay prominent. |
| `metadata` | object? | Free-form JSON stored with the memory. |
| `scope` | `global`\|`project`? | Where to store. Defaults to `project` when a project is given, else `global`. |
| `project` | string? | Project scope. `global` is reserved and means the shared scope. |
| `supersede` | string? | Memory id this fact replaces. |
| `supersede_content` | string? | Exact content of the memory this fact replaces. Matches the target scope first, then the global scope (a global match hides that fact in every scope and is reported as a cross-scope supersede). |
| `supersede_last` | bool | Replace the memory that matches the new fact best (most similar, not the newest) - of the entity, or across the whole scope when no entity is set. Only if the new fact is about the same topic (similarity >= 0.5; scope-wide without an entity >= 0.75). A topic guard compares the attribute words after removing the entity, stopwords, numbers and time values, so a same-template fact about another attribute (port vs backup job) is refused. Without an entity, a scope-wide best match that belongs to a **tagged** entity is refused: the fact is stored additionally and the warning names the entity/id to use for an explicit replacement. |
| `check_contradictions` | bool | NLI-check against the nearest active memories of the scope and auto-supersede contradictions; with an entity that entity's memories are candidates first, without one all entities compete (`contradictions.cross_entity`). |
| `links` | list? | Graph edges to create in the same write: `[{to_entity, relation_type, weight?, context?, from_entity?}]`; `from_entity` defaults to `entity`. Malformed links are an argument error before the write; a failing edge rolls back to a savepoint, warns (`links failed: ...`) and keeps the fact. Each edge is reported in `links` (`status: created`/`updated`/`duplicate`). |
| `dedupe` | enum | `exact` (default, content hash), `semantic` (also vector similarity), `off`. |
| `on_near_duplicate` | enum | Action for a reworded copy of an active memory (near-duplicate, similarity >= 0.75): `warn` (default, store and warn), `supersede` (replace it, list it in `superseded`) or `reject` (`duplicate`, nothing stored). Works under `dedupe="off"` when the policy is not `warn`. **Same-entity guard:** when the matched memory belongs to a different entity, the policy is not applied (the fact is stored and a `near-duplicate across entities` warning names both entities). A high vector similarity alone never deletes another subject's fact. If the content differs in a number or a negation, the warning calls it a **possible contradiction**; `supersede` additionally warns to verify it was an intended update (`check_contradictions=true` gives an NLI verdict). |
| `auto_supersede_near_duplicate` | bool | Backwards-compatible alias for `on_near_duplicate="supersede"`. |
| `metadata_mode` | enum | `merge` (default: keys the new fact does not set are carried over from the replaced memory) or `replace` (the old keys are dropped). |
| `expires_at` | string? | ISO 8601 TTL: the memory leaves the active set after this timestamp (hidden in reads, no duplicate candidate). The row itself stays until it is `forget`-den or superseded; exports keep it. Wrong format = named argument error. |
| `fuzzy_threshold` | number? | Similarity threshold for `dedupe="semantic"` (default 0.85). |

The answer carries `superseded` (the ids that were invalidated) and, with
`check_contradictions`, a `contradictions` object. A duplicate answer names the
mode that matched (`duplicate_mode`) and the measured similarity.

It also reports the scope the fact really landed in (`scope` and `project`).
That matters when `scope="global"` is combined with a project: the fact ends up
in the shared table, and later reads and deletes have to use `project="global"`.

`supersede` and `supersede_content` are checked: an id that does not exist (or
a content without an active match) produces a warning and replaces nothing -
the answer never claims a replacement that did not happen. An id may name a
memory of the global scope from inside a project; `supersede_content` matches
the target scope first and then the global scope, and a global match is
reported as a cross-scope supersede (see below).

A write that is still very similar to an active memory (cosine similarity
>= 0.75) is handled through `on_near_duplicate`: `warn` (default) stores the
fact and names the similar id in `warnings`; `supersede` replaces it; `reject`
answers `status: "duplicate"` with `duplicate_mode: "near"` and stores nothing.
`dedupe="off"` skips the check only for the `warn` policy, because that mode
explicitly asks for every write to land.

When the write invalidates a fact of the shared `memories` table (the
contradiction check or an explicit `supersede` of a global id), the fact
disappears from every scope. The answer carries `cross_scope_superseded` (the
ids) and a `cross-scope supersede` warning so that this is never silent.

Dedupe only checks the table the fact is written to. If the exact content is
already active in another scope, the write succeeds but returns
`cross_scope_duplicates` (ids and scopes) plus a warning: reads hide one of the
copies, so the fork is visible instead of looking like a lost write.
`remember_batch` reports the same per fact entry.

### `remember_batch`

`facts` is a list of objects; only `content` is required, everything else works
like in `remember` (including the per-fact `auto_supersede_near_duplicate`
flag). The top-level `skip_duplicates` flag controls content
dedupe, and `check_contradictions` applies to every fact. Invalid facts are
reported individually, the rest is still stored (`status: "partial"`). A fact
without `content` or one that is not an object at all does not abort the batch
- it is reported as an error entry next to the facts that were stored. Every
entry reports the `scope` and `project` it landed in, so a batch that mixes
scopes stays readable.

### `update_memory`

`memory_id`, `content`, `importance`, `metadata` (`metadata_mode`
`replace`/`merge`, `clear_metadata_keys`), `entity`, `mem_type`,
`expires_at`, `clear_expires_at`, `touch`, `restore`, `project`, `dedupe`
(`exact`/`semantic`/`off`), `fuzzy_threshold`.
Content changes are re-embedded; a change that would duplicate another active
memory is refused as `duplicate`, nothing is applied, and the answer names the
blocking row: `id`/`duplicate_of` are the blocking id, `memory_id` the edited
one (`dedupe="off"` bypasses the check). `metadata_mode="merge"` keeps stored
keys the new object does not set; `clear_metadata_keys` removes keys (also
without a new object). Without `project` the memory is
looked up in every scope; a memory id is unique per scope, so an id that
exists in several scopes is refused and the scopes are named (see
[Scopes](scopes.md)). `expires_at` sets a new TTL and
`clear_expires_at=true` removes it; pass only one of the two.

`touch=true` only bumps `updated_at`, so the decay recency starts over while
content, embedding and `access_count` stay untouched; it can be combined with
`importance`. `restore=true` is the inverse of `forget(soft=true)`: it clears
`deleted_at` and the forgotten sentinel of a soft-forgotten row, keeps a real
`superseded_by`, and is mutually exclusive with every other update field.
Unknown or hard-deleted ids answer `not_found`; an id that exists but is
soft-forgotten, superseded or expired answers `conflict` with the reason
(`touch` on such a row is not a silent `not_found` either). An existing id
with nothing to restore (active or superseded) also answers `conflict`.

### `update_batch`

`updates` is a list of objects with the same fields as `update_memory`
(`memory_id` is required; `touch=true` refreshes the decay age and can be
combined with the other fields; `restore=true` undoes a soft delete and
excludes the other fields). Every entry runs in its own savepoint, so an
invalid, unknown, ambiguous or duplicate entry is reported in its result entry and the
rest of the batch is still applied (`status: "partial"`). The answer carries
`count` and one result per entry; `ok` is only `false` when nothing could be
updated.

### `forget`

`memory_id`, `memory_ids` or `entity` (`entity_name` is kept as alias), `soft`
(default `true`), `project`, `dry_run`. `memory_id` and `memory_ids` are
mutually exclusive.

- `memory_ids` removes a list of ids in one call. Each id is validated
  individually and reported in `bulk_results`; without `project` the same id
  is removed from every scope, while the single-id form refuses an ambiguous
  id unless `project` selects one scope.
- `entity="*"` plus `project` empties a whole scope; `project` is required for
  that, so a stray call cannot wipe the database. The shared scope is
  addressed with `project="global"`.
- `dry_run=true` only reports the counts - and it counts exactly what the real
  call removes: without a `project` the entity is removed from every scope, so
  the preview counts the graph nodes and relations of every scope as well.
- The answer carries `counts` (memories/entities/relations separately) and the
  `project` it acted in; for `entity="*"` the counts also name
  `active_memories` and `already_forgotten`, so a soft dry run and a
  `drop_project` preview agree about rows that were already hidden. The
  single-id form also reports `scopes` (the scopes the id was found in). An
  entity deletion without `project` reports `scopes_affected` and warns when it
  touched more than one scope, so a broad delete is never silent.


## Read tools

| Tool | Description |
|------|-------------|
| `recall` | Hybrid vector + FTS (RRF fusion, project + global); `project="*"` searches every scope; `hybrid`, `min_similarity`, `limit`, `include_decay`/`min_decay_score`, `include_superseded`/`only_superseded`/`include_deleted`/`include_expired`, `include_global`, `graph_boost`/`graph_entity`, `include_relation_context`, `fields`, `updated_after`, `metadata_filter`, `offset`; hidden hits carry `forgotten`/`expired` markers |
| `recall_batch` | Run 1-25 independent searches in one call; `queries`, shared `project`, `include_global`, `limit`, `offset`, `hybrid`, `min_similarity`, decay/history/TTL flags, `graph_boost`/`entity`, `mem_type`, `fields`, `detail`; each `results` entry carries its own `query`, `count`, filter counters, `warnings` and `scopes_searched` |
| `list_memories` | Browse memories without a query (newest first); `entity`, `mem_type`, `project` (`"*"` = every scope), `include_global`, `fields`, `updated_after`, `metadata_filter`, `order_by`, `offset`, `memory_id` (direct lookup, singular `memory`), `memory_ids` (bulk direct lookup, max 200; `missing` and a warning, mutually exclusive with `memory_id`), `include_deleted`, `include_expired`, `include_superseded`, `include_relation_context`, `include_duplicates` (show every cross-scope copy; by default duplicates are hidden and counted in `duplicates_hidden`) |
| `get_entity` | Entity details (including graph connections) + all linked memories + the relation contexts of its edges; `include_global`, `include_relation_context`, `include_superseded`, `include_expired`, `fields`, `limit`, `offset`; reports `total` and `relation_context_count` |
| `list_entities` | Browse entities (`entity_type`, `project`, `include_global`, `offset`); `type` is 'graph' (raw AGE label in `label`) or 'memory-only'; an unknown `entity_type` is an argument error that names the labels present; synthetic `<from>_<relation>_<to>` entities are hidden |
| `list_projects` | Discover every scope; `include_empty` (default true), `include_counts` (default true); each row carries `project`, `scope`, `has_table`, `has_graph` and, with counts, the memory/entity/relation counts (without counts those keys are omitted) |
| `init_context` | Session bootstrap: recent memories (slim, near-duplicates removed, each with its `project`), counters, `projects`/`project_stats` for every scope, graph entity count, health; `detail="slim"\|"full"`, `include_global`, `include_empty_projects`; `project="*"` lists every scope; without a project it warns about the other scopes |

### `recall`

The search tool. `query` is required and non-empty.

| Parameter | Type | Meaning |
|-----------|------|---------|
| `query` | string | Search text, embedded as a query and matched against the vector and full-text indexes. |
| `project` | string? | Scope. Without it only global memories are searched; `"*"` searches every scope. |
| `limit` | int | Maximum number of results (default 10). |
| `hybrid` | bool | True (default) fuses vector and full-text results (RRF). False uses a single retriever. |
| `min_similarity` | number? | Drop vector hits below this cosine similarity (0..1, default 0.40). `null` disables the filter. |
| `include_decay` | bool | Blend recency/access into the score (max 30% of it). |
| `min_decay_score` | number? | Keep only memories with at least this normalized decay score (0..1). |
| `include_superseded` | bool | Also return superseded memories (history). |
| `only_superseded` | bool | Return only invalidated memories. |
| `graph_boost` | bool | Add memories of entities connected in the graph. |
| `graph_entity` | string? | Entity to expand from. With it the query search stays unfiltered. |
| `entity` | string? | Only return memories of this entity. |
| `mem_type` | enum? | Only return memories of this type. |
| `include_global` | bool | True (default) lets a project search also see global memories. |
| `fields` | list? | Only return these columns, e.g. `["id", "content"]`. Score fields include `_graph_relation_context` for relation-context hits. |
| `detail` | enum | `full` (default) or `slim`. `slim` returns id/content/entity/type/project/importance/timestamps plus the ranking scores, without metadata or history bookkeeping; an explicit `fields` list wins. |
| `updated_after` | string? | Only memories changed at or after this ISO date. |
| `metadata_filter` | object? | Only memories whose metadata contains these key/value pairs. Scalar values compare tolerantly (the string `"8001"` matches the stored number `8001`); objects and arrays use JSON containment. |
| `include_relation_context` | bool | True (default) includes the relation-context sentences of graph edges; False hides them. |
| `offset` | int | Skip this many ranked hits (default 0). The answer reports `has_more` and `next_offset`. |

The answer reports `candidates` (the hits that made it into the ranking),
`dropped_by_similarity` (the number of vector hits the filter removed),
`dropped_by_coverage` (the number of full-text hits that covered too little of
the query), plus `fulltext_only` when a hit came only from the full-text leg.
`dropped_by_decay` and `dropped_by_superseded` are reported whenever the
`min_decay_score` / `only_superseded` filter removed hits — also when other
hits survived, so a non-empty result is never quietly short.
`scopes_searched` names the scopes the query actually covered. See
[Retrieval](retrieval.md) for the score fields.

An empty result names every filter that removed something (`min_similarity`,
`min_decay_score`, `only_superseded`, the coverage floor) - not just the first
one, so the caller does not debug the wrong filter. The strongest rejected
vector hit is additionally reported as `best_dropped_by_similarity`
(`id`, `similarity`, `scope`, `content_preview`) and named in the warning, so
a barely-missed fact is visible instead of hidden behind a count.

A project that does not exist is not silently ignored: the search falls back
to the global scope and the answer says so in `warnings`.

### `recall_batch`

Run up to 25 independent `recall` searches in one call. A multi-intent turn
("how do we deploy and where are the logs?") otherwise forces the agent either
to concatenate both intents into one diluted embedding or to make several
round trips. The shared options (`project`, `include_global`, `limit`,
`offset`, `hybrid`, `min_similarity`, decay/history/TTL flags,
`graph_boost`/`entity`, `mem_type`, `fields`, `detail`) apply to every query;
`results` contains one entry per query with its own `query`, `count`,
`results`, filter counters, `warnings` and `scopes_searched`. An empty list or
a blank query is an argument error. A query that fails on its own (for
example an inference outage) is reported in that entry and named in the
top-level `warnings`; the remaining queries still run.

### `list_memories`

Browse without a search. `order_by` is `updated_at` (default), `created_at`,
`access_count` or `importance`. `contains` filters on a case-insensitive
literal substring - the right tool for "which memory contains exactly X",
where the semantic ranking of `recall` is not needed. `memory_id` is the
direct id lookup: the answer carries the single `memory` (and a one-item
`results`) with the `superseded_by`/`forgotten`/`expired` markers; a hidden
row answers `ok: false` and names the flag that reveals it, an id that exists
in several scopes is refused with `scopes`, and `project="*"` searches every
scope. Browse filters are reported as ignored for that lookup. `offset` pages
through a scope; the
answer reports `total`, `has_more` and `next_offset`. `include_deleted=true`
lists soft-forgotten rows and includes their `deleted_at`, which is how an agent
finds an id for `update_memory(restore=true)`. Forgotten rows carry the
machine-readable
`forgotten: true` flag; the `__forgotten__` sentinel is removed from
`superseded_by` in those rows (it was overwriting the real value). Superseded
history stays hidden unless `include_superseded=true` is also passed - the two
flags are orthogonal. `include_expired=true` lists rows past their TTL with
`expired: true`; `include_deleted` alone never shows them.

Without `project` (or with `project="*"`) every scope is listed;
`project="global"` lists only the shared scope. `total` is computed after the
cross-scope dedup, so `total`, `count`, `has_more` and `next_offset` always
describe the same row set. The answer also reports `scopes_searched` and splits
the rows into `total_by_scope` (before cross-scope dedup) and
`listed_by_scope` (after it), so a project browse that also sees global rows
cannot look like a single mixed list. `include_relation_context=false` hides the
relation-context sentences of the graph edges, which repeat what the edge
already says.


### `get_entity`

`entity` (`name` is kept as alias), `project`, `include_global`,
`include_relation_context` (default true), `include_superseded`, `fields`,
`limit` (default 100), `offset`.
Memory-only entities have no graph node;
`details.type` is then `memory-only`; a graph node's `details.type` is `graph`
with the raw AGE label of its scope in `details.label`. A missing entity
answers with `status: "not_found"`. The `memories` list is sorted by `updated_at`
descending, newest change first. It also contains the relation-context
sentences of the entity's edges - they are stored under the synthetic entity
`<from>_<relation>_<to>`, so without this they would not show up here at all.
Pass `include_relation_context=false` to hide them; the entity's own memories
stay.
The answer reports `total`, `count`, `offset`, `has_more`, `next_offset`,
`relation_context_count`, `scopes_searched` and `memories_by_scope`, and a
truncated answer says so in `warnings` instead of looking complete. `total`
also counts the relation-context sentences of the entity's edges. When the same
name exists in several scopes, `details.scopes_with_entity` names each
occurrence (`memory`/`graph`) and a warning says which scope `details`
describes. `project="*"` is rejected: the answer describes exactly one entity,
so there is no single scope to report; omit `project` for the global scope or
name one.

### `list_entities`

`entity_type`, `project`, `include_global`, `limit`, `offset`. Without
`project` (or with `project="*"`) every scope is listed. The list is sorted by
name, so `offset` is stable. Synthetic relation-context entities (`A_KNOWS_B`)
are hidden unless the name also carries a real memory. The same name can occur
in several scopes; each row carries its `project`, so they are distinct rows.

`entity_type` selects one kind of entity: `graph` (every graph node),
`memory-only`, or a raw AGE label (`Entity` for the shared scope,
`Project_<suffix>` for a project) for exactly one scope's nodes. The `type`
column is user-facing: `graph` with the raw AGE label in `label`, or
`memory-only`. `count` is the page size and `total` the full match count, and
an entity that has a graph node is never reported as memory-only.

### `init_context`

The session bootstrap. `project="*"` includes all projects, `detail="full"`
returns every column except the internal ones and the search score
annotations, `include_global=false` excludes global memories.

`projects` names every scope the answer covers, including the shared one
(`"global"`). `project_stats` lists each covered scope with its own
`memories`, `total_memories`, `superseded_memories`, `forgotten_memories`,
`entities` and `relations`; `include_empty_projects=false` drops scopes
without active memories. `active_memories` counts the active rows across the
covered
tables (a cross-scope copy can be counted in both), `total_memories` counts
every row of the scope - the difference is `superseded_memories` plus
`forgotten_memories`. `graph_entities_count` counts
the graph nodes of the covered scopes, not of the whole database.


## Graph tools

| Tool | Description |
|------|-------------|
| `relate` | Directed relation via Cypher (auto-creates entities); `project` selects the project graph |
| `relate_batch` | Many relations in one transaction (MERGE, idempotent, per-edge `weight`/`context`/`project`); an empty list is an error and self-loops are stored with a warning |
| `unrelate` | Delete a single directed relation; `keep_context` keeps the edge sentence; nodes left without an edge or a memory are removed (`orphan_nodes_removed`) |
| `find_connections` | Graph traversal (outgoing/incoming/both), results carry `direction`; `limit`/`offset` page and the answer reports `total`/`has_more`/`next_offset` |
| `find_paths` | Shortest paths between two existing entities (`project`-scoped); edges are traversed in both directions, so a path can run against an edge. Missing entities return `not_found`, no path returns a warning; `from_entity == to_entity` answers a 0-hop path and `max_hops > 10` is clamped with a warning |

- `relate` takes `from_entity`, `to_entity` (not `from`/`to`), `relation_type`
  and `weight` (0..1). The weight scales the retrieval boost of `graph_boost`.
  Repeating the same edge with the same `context` reuses the context memory
  (`context_reused: true`, same `context_memory` id) instead of creating a new
  version; a changed sentence still replaces the old context.
- `relate_batch` takes `edges: [{from_entity, to_entity, relation_type,
  weight?, context?, project?}]`.
- `unrelate` takes `from_entity`, `to_entity`, `relation_type`, `project` and
  `keep_context`. By default the context sentence of the edge is soft-forgotten;
  `keep_context=true` leaves it active. Endpoints that have no further edge and
  no memory row are deleted from the graph, so an empty node does not survive
  in `list_entities`.
- `find_connections` takes `entity` (`entity_name` is kept as alias),
  `direction`, `limit` and `offset`; `total` and `has_more` describe the full
  match set, and a missing entity names the scope it searched.

## Maintenance tools

| Tool | Description |
|------|-------------|
| `contradictions_report` | Detect conflicting facts (LLM NLI + vector pre-filter); per entity or scope-wide (all active memories compete, cross-entity pairs included, each pair carries the entity of both facts and `cross_entity`); an entity scan covers the entity's memories **plus untagged (entity-less) memories** as their neighbours, so a tagged/untagged conflict stays visible and cross-entity pairs are reported but never auto-resolved; `resolve="newest"` keeps the newer fact **of same-entity pairs only** and supersedes the older (newest is the creation time; a `touch` does not decide it); cross-entity pairs, pairs below `resolve_min_confidence` (default 0.8) and pairs without a shared content word stay active (named warning); `dry_run=true` reports `would_resolve` and per-pair `resolution.dry_run` without writing; `budget_seconds` (default 25s) bounds the NLI leg: when it is exhausted the scan stops early, sets `budget_exhausted` and names every unjudged pair in `nli_failures` (`fact_id`, `neighbor_ids`, `reason: skipped: NLI time budget exhausted`); `nli_failures_count` and `nli_failure_summary` give the compact scale without reading every entry; other partial NLI failures are named there too |
| `export_memories` | Backup as JSON/JSONL (`format`, `project`, `include_embeddings`, `include_superseded`, `include_deleted` (default follows `include_superseded`; `false` leaves soft-forgotten rows out), `include_expired`, `include_graph`, `entity`, `mem_type`, `updated_after`, `metadata_filter`); TTL and timestamps are preserved; payload in `data`; without `project` every scope is exported, `project="global"` only the shared scope; `entity` also narrows the graph and keeps the other endpoints of those edges; JSONL warns that `include_graph` is ignored; warns about excluded history and included expired rows; `include_expired=false` leaves expired rows out |
| `verify_integrity` | Report rows whose `superseded_by` points at a missing id or at themselves (both make a row unreadable forever), plus supersede chains and expired active rows; broken links and chains are listed in `results` (chains carry `chain_to`); `project`, `fix` (clear the broken links), `limit`; `fix=true` deletes no content and re-scans after the repair, so the counts and `results` describe the state AFTER the fix (`fixed` carries what was cleared) |
| `import_memories` | Restore from JSON/JSONL (`data` accepts object or string, `project` override, `format`, `skip_duplicates`, `fuzzy_threshold`, `dry_run`); IDs, TTL and timestamps are preserved; an id that already exists is always skipped; `dry_run` validates without writing and adds `would_create`; dangling supersede links are cleared with a warning, as are self-referencing or internal (`__forgotten__`) `superseded_by` values (`repaired_self_references`); `graph_entities`/`graph_relations` count processed records, `*_created` the new ones; a non-empty `skipped` comes with `skipped_reasons` and a warning |
| `drop_project` | Remove a project scope completely (memory table, registry entry, graph nodes, mirror rows); `dry_run` previews; counts name `active_memories`/`already_forgotten` so a preview can be compared with `forget(entity="*")`; the shared scope cannot be dropped |
| `prune_memories` | Permanently delete inactive rows only: `project` (default every scope), `older_than_days` (default 30), `include_expired`/`include_forgotten` (default true), `include_superseded` (default false, history is kept), `dry_run` (default true); active memories are never touched and supersede links that would dangle are cleared; `counts` carries `memories` and per-scope `by_scope`, the message says "Would remove" or "Removed" |
| `server_health` | Check Postgres + Inference connectivity; reports `version`, `code_fingerprint` (detects a stale container), `started_at` and the active `toolset` |

See [Backup & Restore](backup.md) for the export/import contract and
[Retrieval](retrieval.md) for how contradictions are detected.

## Toolset profiles

`MCP_TOOLSET=core` registers 14 tools: `remember`, `remember_batch`,
`update_memory`, `forget`, `recall`, `list_memories`, `get_entity`,
`list_entities`, `list_projects`, `init_context`, `relate`, `unrelate`,
`find_connections` and `find_paths`. The default `full` registers all 24, adding `update_batch`,
`relate_batch`, `contradictions_report`, `export_memories`, `import_memories`,
`verify_integrity`, `prune_memories`, `drop_project` and `server_health`. An
unknown value warns
and falls back to `full`; `server_health` reports the active `toolset`.

## Resource

| Resource | Description |
|----------|-------------|
| `memory://stats` | Memory/entity/relation counts + health |

## Typical workflows

Use `init_context` at session start. With a project set, it loads that project
and global context; follow it with a targeted `recall` when you need a
specific past decision:

```text
init_context(project="api")
recall(query="deployment rollback procedure", project="api", include_decay=true)
```

Store a fact with an entity when it should be discoverable by both search and
entity lookup. Add graph relations when you need explicit traversable links:

```text
remember(content="The API runs on Postgres", entity="API", project="api")
relate(from_entity="API", to_entity="Postgres", relation_type="USES", project="api")
find_connections(entity="API", project="api")
```

Use the returned memory `id` to update or forget that exact record. For backup,
pass the JSON export's `data` payload directly into import; re-import is
idempotent by default:

```text
update_memory(memory_id="<id>", content="Updated fact", project="api")
forget(memory_id="<id>", project="api", soft=true)
export_memories(format="json", project="api")
import_memories(data=<export data>, format="json")
```
