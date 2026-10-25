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
| `status` | `stored`, `created`, `merged`, `duplicate`, `updated`, `deleted`, `soft_forgotten`, `restored`, `not_found`, `conflict`, `partial`, `error`, `ok` |
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
| `remember_batch` | Store multiple facts (savepoints; top-level `skip_duplicates`; per fact: `project`, `scope`, `supersede`, `supersede_content`, `supersede_last`, `on_near_duplicate`, `auto_supersede_near_duplicate`, `expires_at`, `idempotency_key`, `valid_from`/`valid_to`, `source`/`agent_id`/`confidence`, `tags`, `fuzzy_threshold`, `dedupe`; optional `check_contradictions`) |
| `update_memory` | Update content (re-embedding), importance, metadata, entity, type, TTL (`expires_at`/`clear_expires_at`), the validity window (`valid_from`/`valid_to`/`clear_valid_to`), provenance (`source`/`agent_id`/`confidence`) or tags (`tags`/`tags_mode`/`clear_tags`); `touch=true` refreshes the decay recency without a content change, `restore=true` undoes a soft delete |
| `update_batch` | Update many memories in one call (savepoints); per entry `memory_id`, `content`, `importance`, `metadata`, `metadata_mode`, `clear_metadata_keys`, `entity`, `mem_type`, `expires_at`, `clear_expires_at`, `valid_from`, `valid_to`, `clear_valid_to`, `source`, `agent_id`, `confidence`, `tags`, `tags_mode`, `clear_tags`, `touch`, `restore`, `project`; bad entries are reported individually (`partial`), the rest is still applied |
| `merge_memories` | Fold 2-20 active memories of one scope into a new memory: `memory_ids`, `content`, `project`, `entity`, `importance`, `metadata`/`metadata_mode`, `mem_type`, `dedupe`/`fuzzy_threshold`, `dry_run`; every source is superseded by the new id (`status: merged`) |
| `forget` | Soft delete (`deleted_at` + sentinel) or staged hard delete by `memory_id`, `memory_ids`, entity, `metadata_filter`, `contains` or a semantic `query` (vector matches at or above `min_similarity`; keyword-only full-text hits are not selected); a hard delete sets `hard_deleted_at` and stays restorable (`undo_until`) until `prune_memories` purges it after the grace; `entity` also removes the graph node and that scope's memories; selectors need an explicit `project` and are mutually exclusive with ids, `entity` and each other; `counts` splits memories/entities/relations, `bulk_results` names selector hits with `content_preview`, `dry_run` reports `would_*` |

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
| `contradiction_mode` | enum | `supersede` (default: replace the contradicted memory) or `warn` (report the pair in `contradictions.kept` and change nothing). |
| `contradiction_min_confidence` | number | Minimum NLI confidence (0.0-1.0, default 0.70) for an automatic supersede. Lower-scored pairs stay active and are reported. |
| `max_contradiction_supersedes` | int | Maximum number of memories one write may invalidate (default 1). Further confirmed pairs stay active, are listed in `contradictions.kept` and named in a warning, so a single write cannot wipe several facts unconfirmed. |
| `links` | list? | Graph edges to create in the same write: `[{to_entity, relation_type, weight?, context?, from_entity?}]`; `from_entity` defaults to `entity`; unknown fields inside an edge object are refused. Malformed links are an argument error before the write; a failing edge rolls back to a savepoint, warns (`links failed: ...`) and keeps the fact. Each edge is reported in `links` (`status: created`/`updated`/`duplicate`). |
| `dedupe` | enum | `exact` (default, content hash), `semantic` (also vector similarity), `off`. |
| `on_near_duplicate` | enum | Action for a reworded copy of an active memory (near-duplicate, similarity >= 0.75): `warn` (default, store and warn), `supersede` (replace it, list it in `superseded`) or `reject` (`duplicate`, nothing stored). Works under `dedupe="off"` when the policy is not `warn`. **Same-entity guard:** when the matched memory belongs to a different entity, the policy is not applied (the fact is stored and a `near-duplicate across entities` warning names both entities). A high vector similarity alone never deletes another subject's fact. If the content differs in a number or a negation, the warning calls it a **possible contradiction**; `supersede` additionally warns to verify it was an intended update (`check_contradictions=true` gives an NLI verdict). When the write already carried `supersede`/`supersede_content` and the fact still did not replace that near-dup match, the warning appends a reconciliation note naming the directive, so a caller never assumes its instruction did not exist. |
| `auto_supersede_near_duplicate` | bool | Backwards-compatible alias for `on_near_duplicate="supersede"`. |
| `metadata_mode` | enum | `merge` (default: keys the new fact does not set are carried over from the replaced memory) or `replace` (the old keys are dropped). |
| `expires_at` | string? | ISO 8601 TTL: the memory leaves the active set after this timestamp (hidden in reads, no duplicate candidate). The row itself stays until it is `forget`-den or superseded; exports keep it. Wrong format = named argument error. |
| `idempotency_key` | string? | Client token, trimmed, 1-128 characters. A retry with the same key in the same scope writes nothing and answers `duplicate` with `id`/`duplicate_of` naming the stored row, `duplicate_mode: "idempotency_key"` and `idempotent_replay: true`; no dedupe, supersede or near-duplicate side effects run. Keys are per scope, shared by `remember` and `remember_batch`, and a key on an expired, superseded or forgotten row no longer blocks the next write. Non-string, empty or over-long keys are argument errors. |
| `valid_from` | string? | ISO 8601 start of the validity window (default: now). A fact is visible at `as_of` when `valid_from <= as_of` and `valid_to` is absent or later. Invalid values are argument errors. |
| `valid_to` | string? | ISO 8601 end of the validity window (default: open-ended). An explicit `valid_to` earlier than `valid_from` is refused. Superseding a memory closes its window (`valid_to = NOW()`) if it was open. |
| `source` | string? | Where the fact came from, trimmed, max 200 characters; empty means none. Not narrowed by recall, only filtered. |
| `agent_id` | string? | Which agent/tool wrote the fact, trimmed, max 200 characters; empty means none. |
| `confidence` | number? | How sure the writer is, 0.0-1.0. Stored as `confidence`; `min_confidence` filters on it. |
| `tags` | list? | Lower-case, trimmed, deduplicated tags (max 32, each 1-64 characters). A comma-separated string is accepted. `recall(tags=...)` requires all, `tags_any=...` at least one. |
| `fuzzy_threshold` | number? | Similarity threshold for `dedupe="semantic"` (default 0.85). |

The answer carries `superseded` (the ids that were invalidated) and, with
`check_contradictions`, a `contradictions` object with `found`, `superseded`
(only the ids this write actually replaced), `kept` (confirmed pairs left
active by the mode/confidence/budget guard), `mode`, `min_confidence` and
`max_supersedes`. A kept contradiction is also named in `warnings` with an
executable directive (`supersede='<existing_id>'`, one supersede argument per
kept id, or `contradictions_report(resolve="newest")` for same-entity pairs),
so an agent can resolve kept pairs without guessing ids. A duplicate answer
names the mode that matched (`duplicate_mode`) and the measured similarity. An
`idempotency_key` replay is a duplicate too, but it names the row the first call
stored (`idempotent_replay: true`, `duplicate_mode: "idempotency_key"`) instead
of a content match; it never rewrites or supersedes anything. When a partial NLI failure leaves some candidates unjudged (and
some judged), the object adds `unjudged` (`existing_id`, `existing_content`,
`reason`) and a `contradiction check incomplete` warning: the judged pairs
still count, the unverified ones are never silently read as "no
contradiction". Only a failure that leaves nothing judable skips the whole
check (`contradictions.skipped` + warning) and keeps the fact.

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
like in `remember` (including the per-fact `auto_supersede_near_duplicate` flag,
the per-fact `contradiction_mode`, `contradiction_min_confidence` and
`max_contradiction_supersedes`, per-fact `valid_from`/`valid_to`/`source`/
`agent_id`/`confidence`/`tags`, and per-fact `links` like in
[`remember`](#remember)). The top-level `skip_duplicates` flag controls
content dedupe, and `check_contradictions` applies to every fact; the three
contradiction options can be set at the top level or per fact. Invalid facts are
reported individually, the rest is still stored (`status: "partial"`). A fact
without `content` or one that is not an object at all does not abort the batch
- it is reported as an error entry next to the facts that were stored. A
malformed `links` edge is a per-fact prepared failure (the entry answers
`error` and `id: null`); the remaining facts still store. A stored fact
reports its created edges in `links`, and edge failures hang off the entry's
`warnings` (`links failed: ...`) while the fact survives. Every
entry reports the `scope` and `project` it landed in, so a batch that mixes
scopes stays readable. A duplicate entry names the blocking row in a dedicated
`duplicate_of` key (like `remember`, so batch entries and single calls stay
symmetric).

Each fact may carry its own `idempotency_key` (same rules and key space as
[`remember`](#remember)): a retried entry answers `duplicate` with
`idempotent_replay: true` and the id the first call stored, while the other
facts of the batch still run. Replays do not count as failures; a rejected or
duplicate entry creates no graph edges.

### `update_memory`

`memory_id`, `content`, `importance`, `metadata` (`metadata_mode`
`replace`/`merge`, `clear_metadata_keys`), `entity`, `mem_type`,
`expires_at`, `clear_expires_at`, `valid_from`, `valid_to`, `clear_valid_to`,
`source`, `agent_id`, `confidence`, `tags` (`tags_mode` `replace`/`merge`,
`clear_tags`), `touch`, `restore`, `project`, `dedupe`
(`exact`/`semantic`/`off`), `fuzzy_threshold`.
Content changes are re-embedded; a change that would duplicate another active
memory is refused as `ok: false` / `status: duplicate` / `applied: false`,
nothing is applied, and the answer names the
blocking row: `id`/`duplicate_of` are the blocking id, `memory_id` the edited
one (`dedupe="off"` bypasses the check). `metadata_mode="merge"` keeps stored
keys the new object does not set; `clear_metadata_keys` removes keys (also
without a new object). Without `project` the memory is
looked up in every scope; a memory id is unique per scope, so an id that
exists in several scopes is refused and the scopes are named (see
[Scopes](scopes.md)). `expires_at` sets a new TTL and
`clear_expires_at=true` removes it; pass only one of the two. `valid_to` and
`clear_valid_to` are the same pair for the validity window. `source=""` /
`agent_id=""` clear the field (`None` leaves it untouched). `tags_mode="merge"`
unions the new tags with the stored ones (later duplicates removed, still
capped at 32); `clear_tags=true` empties the list.

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
  individually and reported in `bulk_results` (`count` and the `scopes` it was
  found in); without `project` the same id is removed from every scope, while
  the single-id form refuses an ambiguous id unless `project` selects one
  scope. With `dry_run=true` the entries say `would_soft_forget` /
  `would_delete` instead of the real outcome, so a preview cannot be mistaken
  for a completed delete. An explicit `project` — including the reserved
  `"global"` — bounds the removal to that one table: copies of the id in other
  scopes are neither removed nor counted, and the single-id form is no longer
  ambiguous.
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
- `metadata_filter` (object, same comparison as `recall`) or `contains`
  (literal, case-insensitive substring) removes a selected set of active
  memories instead of ids. An explicit `project` is required (use
  `"global"` for the shared table), and neither selector can be combined with
  `memory_id`, `memory_ids`, `entity` or the other selector. `counts`,
  `scopes_affected` and `bulk_results` (`id`, `content_preview`, `reason`,
  `status`) name the hits; `dry_run=true` reports `would_soft_forget` /
  `would_delete` without writing, and a selector that matches nothing answers
  `ok` with a "no memories matched" warning rather than `not_found`.
- `query` is the semantic form of a selector: `limit` (default 20) and
  `min_similarity` (default 0.85) bound the search, it needs an explicit
  `project`, searches only active rows and does not bump `access_count`. Only
  vector matches are selected: a row needs a real vector score at or above
  `min_similarity`, so a keyword-only full-text hit is never removed (even with
  `min_similarity=0.0`). Each hit is reported in `bulk_results` with `id`,
  `content_preview` and `similarity` (`null` when the row has no vector score);
  no hit answers `ok` with a warning.
- `soft=false` no longer deletes immediately. The row gets `deleted_at` and
  `hard_deleted_at`, the status stays `deleted`, and the answer reports
  `undo_until` / `purge_after` plus a "recoverable until" warning.
  `update_memory(restore=true)` revives it; `prune_memories` removes it once
  the grace period (`HARD_DELETE_GRACE_DAYS`, default 7 days) has passed.


### `merge_memories`

`memory_ids` (2-20 active ids), `content` (required), `project`, `entity`,
`importance`, `metadata`, `metadata_mode` (`merge`/`replace`), `mem_type`,
`dedupe` (`exact`/`semantic`/`off`), `fuzzy_threshold`, `dry_run`.

All sources must be active (not superseded, soft-forgotten or expired) and live
in one scope: with `project` the named scope is used, without it the sources
name the scope and a mixed set answers `error` with `scopes`. An inactive or
unknown source answers `conflict` (`invalid` names the id and reason) and
nothing is written. The merged content is checked like a write (`duplicate`
when another active row has it); the new memory is written first and then every
source is marked `superseded_by` the new id in the same transaction, so a
failure leaves the sources untouched. `metadata_mode="merge"` (default)
combines the sources' metadata (later ids win) and lets explicit keys override
it, without carrying graph-edge metadata; `entity`, `mem_type` and
`importance` default to the sources' shared value (highest importance). The
answer is `status: merged` with `id`, `merged_ids`, `superseded`, `count` and
`project`; `dry_run=true` answers `would_create`/`would_supersede` and writes
nothing.


## Read tools

| Tool | Description |
|------|-------------|
| `recall` | Hybrid vector + FTS (RRF fusion, project + global); ambiguous top hits are reranked by a local cross-encoder (`reranked`/`rerank_error`); `project="*"` searches every scope; `hybrid`, `min_similarity`, `min_relevance`, `limit`, `include_decay`/`min_decay_score`, `include_superseded`/`only_superseded`/`include_deleted`/`include_expired`, `include_global`, `graph_boost`/`graph_entity`, `include_relation_context`, `fields`, `updated_after`, `as_of`, `tags`/`tags_any`, `source`, `agent_id`, `min_confidence`, `metadata_filter`, `offset`, `include_pinned`; hidden hits carry `forgotten`/`expired` markers |
| `recall_batch` | Run 1-25 independent searches in one call; `queries`, shared `project`, `include_global`, `limit`, `offset`, `hybrid`, `min_similarity`, `min_relevance`, decay/history/TTL flags, `graph_boost`/`entity`, `mem_type`, `as_of`/`tags`/`tags_any`/`source`/`agent_id`/`min_confidence`, `fields`, `detail`; each `results` entry carries its own `query`, `count`, filter counters, `warnings` and `scopes_searched` |
| `compile_context` | Ranked, token-bounded context block for a prompt: `query`, `project`, `include_global`, `max_tokens` (256-32000), `limit`, `entity`, `metadata_filter`, `updated_after`, `as_of`, `tags`/`tags_any`, `source`, `agent_id`, `min_confidence`, `hybrid`, `include_decay`, `graph_boost`/`graph_entity`, `include_pinned` (default true), `min_relevance`/`min_similarity`, `diversity` (MMR, default 0), `max_per_entity` (default none); reports `context`, `items`, `used_tokens`, `truncated`, `pinned_included`, `dropped_by_relevance`/`dropped_by_similarity`/`dropped_by_diversity`/`dropped_by_entity_cap`, `diversity` |
| `list_memories` | Browse memories without a query (newest first); `entity`, `mem_type`, `project` (`"*"` = every scope), `include_global`, `fields`, `updated_after`, `as_of`, `tags`/`tags_any`, `source`, `agent_id`, `min_confidence`, `metadata_filter`, `order_by`, `offset`, `memory_id` (direct lookup, singular `memory`), `memory_ids` (bulk direct lookup, max 200; `missing` and a warning, mutually exclusive with `memory_id`), `include_deleted`, `include_expired`, `include_superseded`, `include_relation_context`, `include_duplicates` (show every cross-scope copy; by default duplicates are hidden and counted in `duplicates_hidden`) |
| `get_entity` | Entity details (including graph connections) + all linked memories + the relation contexts of its edges; `include_global`, `include_relation_context`, `include_superseded`, `include_expired`, `as_of`/`tags`/`tags_any`/`source`/`agent_id`/`min_confidence`, `fields`, `limit`, `offset`; reports `total`, `relation_context_count` and `duplicates_hidden` (identical content in several scopes is listed once); an unknown project is named in `warnings` like `recall` |
| `get_entities` | Batch `get_entity` for 1-25 names (duplicates removed, blank names rejected); per-entry `details`/`memories`/`total`/`has_more`/`duplicates_hidden`, `succeeded`/`failed`; an unknown name is a `not_found` entry, not an abort; `project="*"` rejected |
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
| `min_similarity` | number? | Drop vector hits below this cosine similarity (0..1, default 0.30, just above the model's 0.288 noise floor). `null` disables the filter. |
| `min_relevance` | number? | Drop query hits below this combined `_relevance_score` (0..1, the blend of vector similarity and full-text rank). Graph-only hits carry no relevance score and are never dropped. `null` (default) applies no explicit floor, but a full-text-only hit below an implicit 0.25 noise floor is still dropped (exact `_exact_match` hits are exempt). |
| `include_decay` | bool | Blend recency/access into the score (max 30% of it). |
| `min_decay_score` | number? | Keep only memories with at least this normalized decay score (0..1). |
| `include_superseded` | bool | Also return superseded memories (history). |
| `only_superseded` | bool | Return only invalidated memories. |
| `include_pinned` | bool | Backfill the page with the scope's pinned facts (never dropped by a relevance/decay floor; they carry `_pinned` and are counted in `pinned_included`). |
| `graph_boost` | bool | Add memories of entities connected in the graph. |
| `graph_entity` | string? | Entity to expand from. With it the query search stays unfiltered. |
| `entity` | string? | Only return memories of this entity. |
| `mem_type` | enum? | Only return memories of this type. |
| `include_global` | bool | True (default) lets a project search also see global memories. |
| `fields` | list? | Only return these columns, e.g. `["id", "content"]`. Score fields include `_rerank_score` for reranked rows and `_graph_relation_context` for relation-context hits. |
| `detail` | enum | `full` (default) or `slim`. `slim` returns id/content/entity/type/project/importance/timestamps plus the ranking scores, without metadata or history bookkeeping; an explicit `fields` list wins. |
| `updated_after` | string? | Only memories changed at or after this ISO date. |
| `as_of` | string? | Read the scope at a point in time: only rows whose validity window contains this ISO timestamp. Out-of-window matches are counted in `dropped_by_validity`. |
| `tags` | list? | Only memories that carry **all** of these tags (AND). Mutually exclusive with `tags_any`. |
| `tags_any` | list? | Only memories that carry **at least one** of these tags (OR). Mutually exclusive with `tags`. |
| `source` | string? | Only memories with this exact `source` (trimmed). |
| `agent_id` | string? | Only memories with this exact `agent_id` (trimmed). |
| `min_confidence` | number? | Only memories whose `confidence` is set and at least this value (0..1). |
| `metadata_filter` | object? | Only memories whose metadata contains these key/value pairs. Scalar values compare tolerantly (the string `"8001"` matches the stored number `8001`); objects and arrays use JSON containment. |
| `include_relation_context` | bool | True (default) includes the relation-context sentences of graph edges; False hides them. |
| `offset` | int | Skip this many ranked hits (default 0). The answer reports `has_more` and `next_offset`. |

When the best hit is below **0.85** relevance and leads the runner-up by less
than **0.10**, the top candidates are rescored by the local cross-encoder
(`RERANK_TOP_K`, default 10). Reranked rows carry `_rerank_score` (a
per-candidate sigmoid of the cross-encoder logit) and a capped
`_relevance_score`; the answer reports `reranked`. A reranker failure adds
`rerank_error` and a `reranking skipped: ...` warning and leaves the first-stage
order unchanged.

The answer reports `candidates` (the hits that made it into the ranking),
`dropped_by_similarity` (the number of vector hits the filter removed),
`dropped_by_expired` (rows hidden because their TTL passed),
`dropped_by_validity` (rows the filters matched but whose validity window did
not contain `as_of`; reported only when `as_of` is passed),
`dropped_by_coverage` (the number of full-text hits that covered too little of
the query), plus `fulltext_only` when a hit came only from the full-text leg.
`dropped_by_relevance` and `dropped_by_decay` are reported whenever the
`min_relevance` / `min_decay_score` filter removed hits — also when other
hits survived, so a non-empty result is never quietly short. With
`only_superseded` the filter drops the ACTIVE rows, so they are counted in
`dropped_by_not_superseded` and `dropped_by_superseded` stays present as `0`.
`scopes_searched` names the scopes the query actually covered. See
[Retrieval](retrieval.md) for the score fields.

An empty result names every filter that removed something (`min_similarity`,
`min_relevance`, `min_decay_score`, `only_superseded`, the coverage floor,
the TTL) - not just the first
one, so the caller does not debug the wrong filter. The strongest rejected
vector hit is additionally reported as `best_dropped_by_similarity`
(`id`, `similarity`, `scope`, `content_preview`) and named in the warning, so
a barely-missed fact is visible instead of hidden behind a count.

A project that does not exist is not silently ignored: the search falls back
to the global scope and the answer says so in `warnings`.

The answer carries `confidence` and `best_relevance` (the best
`_relevance_score` of the page). `high` means the best hit scored at least
0.60, `medium` at least 0.45, `low` below that, `none` for an empty result and
`graph` when only graph-boosted hits (which have no similarity) returned. A
`low` answer adds a warning that the rows may not answer the query and names
`min_similarity=null` as the way to inspect weaker hits - a non-empty result is
never presented as a confident answer.

### `recall_batch`

Run up to 25 independent `recall` searches in one call. A multi-intent turn
("how do we deploy and where are the logs?") otherwise forces the agent either
to concatenate both intents into one diluted embedding or to make several
round trips. The shared options (`project`, `include_global`, `limit`,
`offset`, `hybrid`, `min_similarity`, `min_relevance`, decay/history/TTL flags,
`graph_boost`/`entity`, `mem_type`, `fields`, `detail`, plus the `as_of` /
`tags` / `tags_any` / `source` / `agent_id` / `min_confidence` filters) apply
to every query;
`results` contains one entry per query with its own `query`, `count`,
`results`, filter counters, `warnings` and `scopes_searched`. An empty list is
an argument error. A query that fails on its own (for example an inference
outage) is reported in that entry and named in the top-level `warnings`; the
remaining queries still run. The same holds for a blank query: its entry
answers `ok: false`, `status: "error"` and `count: 0` instead of discarding the
other intents of the batch.

### `compile_context`

`query`, `project`, `include_global`, `max_tokens` (256-32000, default 4000),
`limit` (default 50), `entity`, `metadata_filter`, `updated_after`, `hybrid`,
`include_decay` (default true), `graph_boost`/`graph_entity`, `include_pinned`
(default true), `min_relevance` / `min_similarity`, `diversity` (MMR weight,
default 0.3, `0.0` = strict no-op) and `max_per_entity` (default none), plus the `as_of` /
`tags` / `tags_any` / `source` / `agent_id` / `min_confidence` filters. Runs
the same hybrid search as `recall` (same scope, filter and dedupe semantics) and
packs
the ranked hits into one `context` string in ranking order; with `include_pinned`
the pinned facts lead the block and no relevance/decay floor drops them. With
`diversity > 0` near-duplicates (identical content hash or cosine >= 0.95)
collapse and the block prefers hits unlike those already chosen;
`max_per_entity` caps one entity's contribution but never evicts a pin. The
estimate is
conservative (about four characters per token plus a fixed per-item overhead),
so `used_tokens` never exceeds `max_tokens`; a hit or pin that does not fit
stops the block and sets `truncated`. Each `items` entry carries `id`,
`content`, `entity`, `mem_type`, `project`, `pinned`, `_relevance_score` and
`_final_score`; the answer reports `pinned_included`, `diversity` and the
`dropped_by_relevance` / `dropped_by_similarity` / `dropped_by_diversity` /
`dropped_by_entity_cap` counters. An
empty result answers `ok` with `count: 0`, an empty `context` and a warning.

### `list_memories`

Browse without a search. `order_by` is `updated_at` (default), `created_at`,
`access_count`, `importance` or `pinned` (pins first, then newest).
`contains` filters on a case-insensitive
literal substring - the right tool for "which memory contains exactly X",
where the semantic ranking of `recall` is not needed. `memory_id` is the
direct id lookup: the answer carries the single `memory` (and a one-item
`results`) with the `superseded_by`/`forgotten`/`expired` markers; a hidden
row answers `ok: false` and names the flag that reveals it, an id that exists
in several scopes is refused with `scopes`, and `project="*"` searches every
scope. Browse filters are reported as ignored for that lookup. `memory_ids`
looks up several ids at once (max 200): rows that exist but are hidden by
their state are listed in `hidden` with `reasons` and `reveal_flags` (for
example `include_deleted`), ids that exist nowhere are listed in `missing`, and
both are named in `warnings` - a hidden row is never reported as missing.
`offset` pages
through a scope; the
answer reports `total`, `has_more` and `next_offset`. `include_deleted=true`
lists soft-forgotten rows and includes their `deleted_at`, which is how an agent
finds an id for `update_memory(restore=true)`. Forgotten rows carry the
machine-readable
`forgotten: true` flag; the `__forgotten__` sentinel is removed from
`superseded_by` in those rows (it was overwriting the real value). Superseded
history stays hidden unless `include_superseded=true` is also passed - the two
flags are orthogonal. `include_expired=true` lists rows past their TTL with
`expired: true`; `include_deleted` alone never shows them. The `as_of` / `tags`
/ `tags_any` / `source` / `agent_id` / `min_confidence` filters narrow a browse
exactly like `recall`; a tags/tags_any pair is refused as mutually exclusive,
and a direct `memory_id`/`memory_ids` lookup reports them as ignored.

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
`limit` (default 100), `offset`, and the `as_of` / `tags` / `tags_any` /
`source` / `agent_id` / `min_confidence` filters (which also narrow `total`, so
the count describes the same filtered set as the page).
Memory-only entities have no graph node;
`details.type` is then `memory-only`; a graph node's `details.type` is `graph`
with the raw AGE label of its scope in `details.label`. A missing entity
answers with `status: "not_found"`; when the name lives in another scope, the
answer carries `scopes_with_entity` (`{project: "memory"/"graph"}`) and the
error/warning say to repeat the call with `project=<name>`, so a cross-scope
lookup needs no second discovery round trip. The `memories` list is sorted by `updated_at`
descending, newest change first. It also contains the relation-context
sentences of the entity's edges - they are stored under the synthetic entity
`<from>_<relation>_<to>`, so without this they would not show up here at all.
Pass `include_relation_context=false` to hide them; the entity's own memories
stay.
The answer reports `total`, `count`, `offset`, `has_more`, `next_offset`,
`relation_context_count`, `scopes_searched` and `memories_by_scope` (a real
`COUNT(*)` per scope, so it does not change with `limit`), and a
truncated answer says so in `warnings` instead of looking complete. `total`
also counts the relation-context sentences of the entity's edges. When the same
name exists in several scopes, `details.scopes_with_entity` names each
occurrence (`memory`/`graph`) and a warning says which scope `details`
describes. `details.connections` is ordered by weight (strongest first), then
target name and relation type, so repeated calls return the same order.
`project="*"` is rejected: the answer describes exactly one entity,
so there is no single scope to report; omit `project` for the global scope or
name one.

### `get_entities`

Batch form of `get_entity`: `entities` (1-25 items, duplicates removed, blank
names are an argument error), `project`, `include_global`, `include_superseded`,
`include_expired`, `include_relation_context`, `fields`, `limit_per_entity`
(default 50), `offset_per_entity`, plus the same `as_of` / `tags` / `tags_any` /
`source` / `agent_id` / `min_confidence` filters. Each item is a trimmed name
(uses the tool-level `project`) or an object `{"entity": "API", "project":
"api"}` (`name` is accepted as an alias), so one call can span scopes. A
duplicate `(entity, project)` pair is removed, and an object with an unknown key
is an `error` entry naming the key without aborting the rest. Each `results`
entry answers exactly like a single `get_entity` call (`ok`, `status`, `entity`,
`project`, `details`, `memories`, `total`, `has_more`, `duplicates_hidden`,
`scopes_searched`, `warnings`, plus `relation_context_count` and
`scopes_with_entity` when present). An unknown name is a `not_found` entry
instead of aborting the call. The envelope reports `count`, `succeeded`,
`failed`, the sorted `scopes`, and `status` is `partial` when some entries
failed; a call where every name is unknown answers `ok: false` with
`not_found`. `project="*"` is rejected for the same reason as in `get_entity`,
also per entry.

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
| `relate` | Directed relation via Cypher (auto-creates entities); `project` selects the project graph; repeating an existing edge without `context` reports the stored sentence's `context_memory` with `context_reused: true` |
| `relate_batch` | Many relations in one transaction (MERGE, idempotent, per-edge `weight`/`context`/`project`); an empty list is an error and self-loops are stored with a warning |
| `unrelate` | Delete a single directed relation; `keep_context` keeps the edge sentence; nodes left without an edge or a memory are removed (`orphan_nodes_removed`) |
| `delete_entity` | Delete one graph node with `DETACH DELETE`; `remove_memories` (default `false`) also soft-forgets the entity's memories, `remove_relations` (default `true`) may be set to `false` only when the node has no edge (otherwise `conflict` with `relations: <n>`); `dry_run` reports `would_*` counters; neighbours survive |
| `find_connections` | Graph traversal (outgoing/incoming/both), results carry `direction`; `limit`/`offset` page and the answer reports `total`/`has_more`/`next_offset` |
| `find_paths` | Shortest paths between two existing entities (`project`-scoped); edges are traversed in both directions, so a path can run against an edge. Each relation carries `relation_type` and the edge's `context` sentence when one was stored. Missing entities return `not_found`, no path returns a warning; `from_entity == to_entity` answers a 0-hop path and `max_hops > 10` is clamped with a warning |
| `list_relations` | Browse edges without a start entity: `entity` (either endpoint), `relation_type`, `project` (omitted or `"*"` = every scope, `"global"` = shared only); sorted by weight then endpoints, with `total`/`has_more`/`next_offset` |

- `relate` takes `from_entity`, `to_entity` (not `from`/`to`), `relation_type`
  and `weight` (0..1). Omitting `weight` keeps the weight of an existing edge
  (1.0 for a new one), so adding a context sentence does not reset it. The
  weight scales the retrieval boost of `graph_boost`.
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
  `direction`, `limit` and `offset`; each result carries the edge's `context`
  sentence when one was stored, `total` and `has_more` describe the full
  match set, and a missing entity names the scope it searched.
- `list_relations` pages through edges directly; `project` omitted (or `"*"`)
  covers every scope, `"global"` only the shared graph. Each edge carries its
  `context` sentence when one was stored, like `find_connections`.
- Entity names are trimmed before use. A blank or whitespace-only name is an
  argument error, and a name containing `$` or a backtick is refused: those
  characters are reserved by the Cypher layer, so accepting them would create
  a node that no subsequent lookup can address. Relation types are restricted
  to letters, digits and `_`.

## Maintenance tools

| Tool | Description |
|------|-------------|
| `contradictions_report` | Detect conflicting facts (LLM NLI + vector pre-filter); per entity or scope-wide (all active memories compete, cross-entity pairs included, each pair carries the entity of both facts and `cross_entity`); an entity scan covers the entity's memories **plus untagged (entity-less) memories** as their neighbours, so a tagged/untagged conflict stays visible and cross-entity pairs are reported but never auto-resolved; `resolve="newest"` keeps the newer fact **of same-entity pairs only** and supersedes the older (newest is the creation time; a `touch` does not decide it); cross-entity pairs, pairs below `resolve_min_confidence` (default 0.8) and pairs without a shared content word stay active (named warning); `dry_run=true` reports `would_resolve` and per-pair `resolution.dry_run` without writing; `budget_seconds` (default 40s) is the wall-clock budget of the whole scan (candidate loading plus the NLI leg, so the call stays inside the client timeout): a load that consumed it returns before any NLI call, and when the judge leg runs out the scan returns the pairs judged so far, sets `budget_exhausted` and names every unjudged pair in `nli_failures` (`fact_id`, `neighbor_ids`, `reason: skipped: NLI time budget exhausted`); `nli_failures_count` and `nli_failure_summary` give the compact scale without reading every entry; `nli_failure_kinds` splits them into `budget`/`timeout`/`parse`/`outage`/`error`, and each failure entry carries the same `kind`; other partial NLI failures are named there too |
| `export_memories` | Backup as JSON/JSONL (`format`, `project`, `include_embeddings`, `include_superseded`, `include_deleted` (default follows `include_superseded`; `false` leaves soft-forgotten rows out), `include_expired`, `include_graph`, `entity`, `mem_type`, `updated_after`, `metadata_filter`); TTL and timestamps are preserved; payload in `data`; without `project` every scope is exported, `project="global"` only the shared scope; `entity` also narrows the graph, keeps the other endpoints of those edges and includes their relation-context sentences; JSONL carries the same graph as extra `record_type` (`entity`/`relation`) lines (relation `context` included), so both formats restore the same state; warns about excluded history and included expired rows; `include_expired=false` leaves expired rows out |
| `verify_integrity` | Report rows whose `superseded_by` points at a missing id or at themselves (both make a row unreadable forever), plus supersede chains and expired active rows; broken links and chains are listed in `results` (chains carry `chain_to`); `project`, `fix` (clear the broken links), `limit`; `fix=true` deletes no content and re-scans after the repair, so the counts and `results` describe the state AFTER the fix (`fixed` carries what was cleared) |
| `import_memories` | Restore from JSON/JSONL (`data` accepts an object, a list of records or a string, `project` override, `format`, `skip_duplicates`, `fuzzy_threshold`, `dry_run`); IDs, TTL and timestamps are preserved; an id that already exists is always skipped; JSONL `record_type` graph lines are restored through the same idempotent MERGE path (relation `context` included); every record runs in its own savepoint, so one broken record (bad embedding, invalid row) is reported in `errors` and the rest still imports; a malformed JSONL line or broken graph record (unknown `record_type`, relation without endpoints, bad weight) is collected as `skipped_reasons.invalid` (named as `jsonl line N`) instead of aborting the file; `dry_run` validates without writing and adds `would_create`; dangling supersede links are cleared with a warning, as are self-referencing or internal (`__forgotten__`) `superseded_by` values (`repaired_self_references`); a kept supersede link whose target lives in another scope is reported as `cross_scope_supersede_links` plus a warning (cross-scope supersedes are valid, so the link stays - if that scope is never restored, the rows remain history-hidden and `drop_project` of the missing scope clears the link), and a dry run warns when the real import would clear a link (a target not stored anywhere); `graph_entities`/`graph_relations` count processed records, `*_created` the new ones; a non-empty `skipped` comes with `skipped_reasons` and a warning |
| `drop_project` | Remove a project scope completely (memory table, registry entry, graph nodes, mirror rows); `dry_run` previews; counts name `active_memories`/`already_forgotten` so a preview can be compared with `forget(entity="*")`; the shared scope cannot be dropped |
| `prune_memories` | Permanently delete inactive rows only: pass exactly one of `project` (a name, or `"global"` for the shared table) and `all_scopes=true` (neither answers "pass project=<name> or all_scopes=true", both is an error); `older_than_days` (default 30), `hard_delete_grace_days` (default 7 or `HARD_DELETE_GRACE_DAYS`), `include_expired`/`include_forgotten` (default true), `include_superseded` (default false, history is kept), `dry_run` (default true); active memories are never touched, hard-deleted rows are removed once their grace has passed, and supersede links that would dangle are cleared; `counts` carries `memories`, per-scope `by_scope` and `pending_purge`, `scopes_pruned` names the scopes, the message says "Would remove" or "Removed" |
| `server_health` | Check Postgres + Inference connectivity; reports `version`, `code_fingerprint` (detects a stale container), `started_at` and the active `toolset` |

See [Backup & Restore](backup.md) for the export/import contract and
[Retrieval](retrieval.md) for how contradictions are detected.

## Toolset profiles

`MCP_TOOLSET=core` registers 17 tools: `remember`, `remember_batch`,
`update_memory`, `forget`, `recall`, `compile_context`, `list_memories`,
`get_entity`, `list_entities`, `list_projects`, `init_context`, `relate`,
`delete_entity`, `unrelate`, `find_connections`, `find_paths` and
`list_relations`. The default `full` registers all 29, adding `update_batch`,
`merge_memories`, `get_entities`, `relate_batch`, `contradictions_report`,
`export_memories`, `import_memories`, `verify_integrity`, `prune_memories`,
`drop_project` and `server_health`. An
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
