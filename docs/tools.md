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
| `status` | `stored`, `created`, `duplicate`, `updated`, `deleted`, `soft_forgotten`, `restored`, `not_found`, `partial`, `error`, `ok` |
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

## Write tools

| Tool | Description |
|------|-------------|
| `remember` | Store a fact/decision/learning (auto-embedding, `dedupe` mode, `supersede`/`supersede_content`/`supersede_last`, optional contradiction check, `scope` global/project) |
| `remember_batch` | Store multiple facts (savepoints; per fact: `project`, `scope`, `supersede`, `fuzzy_threshold`, `dedupe`; optional `check_contradictions`) |
| `update_memory` | Update content (re-embedding), importance or metadata |
| `update_batch` | Update many memories in one call (savepoints); per entry `memory_id`, `content`, `importance`, `metadata`, `entity`, `mem_type`, `project`; bad entries are reported individually (`partial`), the rest is still applied |
| `forget` | Soft delete (`deleted_at` + sentinel) or hard delete; `entity` also removes the graph node and that scope's memories; `counts` splits memories/entities/relations |
| `restore_memory` | Bring a soft-forgotten memory back; only rows with `deleted_at` set can be restored |
| `touch_memory` | Bump `updated_at` without changing the content (refreshes the decay recency); optional `importance` |

### `remember`

| Parameter | Type | Meaning |
|-----------|------|---------|
| `content` | string | The fact to store (non-empty). |
| `mem_type` | enum | `fact` (default), `decision`, `learning`, `preference`, `pattern`, `incident`, `runbook`, `context`, `observation`, `convention`, `command`, `note`. |
| `entity` | string? | Entity this memory belongs to; connects it in the graph and enables `graph_boost` / `find_connections`. |
| `importance` | number | 0.0-1.0 (default 1.0). Scales the decay weight, so low-importance memories fade faster. |
| `metadata` | object? | Free-form JSON stored with the memory. |
| `scope` | `global`\|`project`? | Where to store. Defaults to `project` when a project is given, else `global`. |
| `project` | string? | Project scope. `global` is reserved and means the shared scope. |
| `supersede` | string? | Memory id this fact replaces. |
| `supersede_content` | string? | Exact content of the memory this fact replaces. |
| `supersede_last` | bool | Replace the memory of the same entity that matches the new fact best (most similar, not the newest) - only if the new fact is about the same topic (similarity >= 0.5), otherwise the fact is stored additionally and a warning is returned. |
| `check_contradictions` | bool | NLI-check against the nearest active memories of the scope and auto-supersede contradictions; with an entity that entity's memories are candidates first, without one all entities compete (`contradictions.cross_entity`). |
| `dedupe` | enum | `exact` (default, content hash), `semantic` (also vector similarity), `off`. |
| `fuzzy_threshold` | number? | Similarity threshold for `dedupe="semantic"` (default 0.85). |

The answer carries `superseded` (the ids that were invalidated) and, with
`check_contradictions`, a `contradictions` object. A duplicate answer names the
mode that matched (`duplicate_mode`) and the measured similarity.

It also reports the scope the fact really landed in (`scope` and `project`).
That matters when `scope="global"` is combined with a project: the fact ends up
in the shared table, and later reads and deletes have to use `project="global"`.

`supersede` and `supersede_content` are checked: an id that does not exist (or
a content without an active match) produces a warning and replaces nothing -
the answer never claims a replacement that did not happen.

A stored write that is still very similar to an active memory (cosine
similarity >= 0.75) is reported in `warnings` as a near-duplicate, naming the
similar id and the measured similarity. The fact is stored anyway - only the
silent fork is removed. `dedupe="off"` skips the check, because that mode
explicitly asks for every write to land.

When the write invalidates a fact of the shared `memories` table (the
contradiction check or an explicit `supersede` of a global id), the fact
disappears from every scope. The answer carries `cross_scope_superseded` (the
ids) and a `cross-scope supersede` warning so that this is never silent.

### `remember_batch`

`facts` is a list of objects; only `content` is required, everything else works
like in `remember`. Invalid facts are reported individually, the rest is still
stored (`status: "partial"`). A fact without `content` or one that is not an
object at all does not abort the batch - it is reported as an error entry
next to the facts that were stored. Every entry reports the `scope` and
`project` it landed in, so a batch that mixes scopes stays readable.

### `update_memory`

`memory_id`, `content`, `importance`, `metadata`, `entity`, `mem_type`,
`project`. Content changes are re-embedded. Without `project` the memory is
looked up in every scope; a memory id is unique per scope, so an id that
exists in several scopes is refused and the scopes are named (see
[Scopes](scopes.md)).

### `update_batch`

`updates` is a list of objects with the same fields as `update_memory`
(`memory_id` is required). Every entry runs in its own savepoint, so an
invalid, unknown or ambiguous entry is reported in its result entry and the
rest of the batch is still applied (`status: "partial"`). The answer carries
`count` and one result per entry; `ok` is only `false` when nothing could be
updated.

### `forget`

`memory_id` or `entity` (`entity_name` is kept as alias), `soft` (default
`true`), `project`, `dry_run`.

- `entity="*"` plus `project` empties a whole scope; `project` is required for
  that, so a stray call cannot wipe the database. The shared scope is
  addressed with `project="global"`.
- `dry_run=true` only reports the counts - and it counts exactly what the real
  call removes: without a `project` the entity is removed from every scope, so
  the preview counts the graph nodes and relations of every scope as well.
- The answer carries `counts` (memories/entities/relations separately) and
  `scopes` (the scopes the id was found in).

### `restore_memory`

`memory_id`, `project`. Clears `deleted_at` and the forgotten sentinel, so the
memory is visible to `recall`/`list_memories` again. Find the id with
`list_memories(include_deleted=true)`; active or hard-deleted ids answer with
`not_found`. Without `project` an id that exists in several scopes is refused
(`scopes` names them).

### `touch_memory`

`memory_id`, optional `importance`, `project`. Bumps `updated_at` without
touching content or embedding, so the cognitive decay treats the fact as
recent again. The content stays byte-identical (unlike `update_memory`, which
re-embeds). An unknown id answers `not_found`; without `project` the id is
looked up in every scope (an ambiguous id is refused, like `update_memory`).

## Read tools

| Tool | Description |
|------|-------------|
| `recall` | Hybrid vector + FTS (RRF fusion, project + global); `project="*"` searches every scope; `hybrid`, `min_similarity`, `limit`, `include_decay`/`min_decay_score`, `include_superseded`/`only_superseded`, `include_global`, `graph_boost`/`graph_entity`, `include_relation_context`, `fields`, `updated_after`, `metadata_filter`, `offset` |
| `list_memories` | Browse memories without a query (newest first); `entity`, `mem_type`, `project` (`"*"` = every scope), `include_global`, `fields`, `updated_after`, `metadata_filter`, `order_by`, `offset`, `memory_id`, `include_deleted`, `include_superseded`, `include_relation_context` |
| `get_entity` | Entity details (including graph connections) + all linked memories + the relation contexts of its edges; `include_global`, `include_superseded`, `fields`, `limit`, `offset` |
| `list_entities` | Browse entities (`entity_type`, `project`, `include_global`, `offset`); `type` is 'graph' (raw AGE label in `label`) or 'memory-only'; synthetic `<from>_<relation>_<to>` entities are hidden |
| `init_context` | Session bootstrap: recent memories (slim, near-duplicates removed, each with its `project`), counters, projects, graph entity count, health; `detail="slim"\|"full"`, `include_global`; `project="*"` lists every scope |
| `list_projects` | List the known project scopes with their sizes (memories, entities, relations) |

### `recall`

The search tool. `query` is required and non-empty.

| Parameter | Type | Meaning |
|-----------|------|---------|
| `query` | string | Search text, embedded as a query and matched against the vector and full-text indexes. |
| `project` | string? | Scope. Without it only global memories are searched; `"*"` searches every scope. |
| `limit` | int | Maximum number of results (default 10). |
| `hybrid` | bool | True (default) fuses vector and full-text results (RRF). False uses a single retriever. |
| `min_similarity` | number? | Drop vector hits below this cosine similarity (0..1, default 0.35). `null` disables the filter. |
| `include_decay` | bool | Blend recency/access into the score (max 30% of it). |
| `min_decay_score` | number? | Keep only memories with at least this normalized decay score (0..1). |
| `include_superseded` | bool | Also return superseded memories (history). |
| `only_superseded` | bool | Return only invalidated memories. |
| `graph_boost` | bool | Add memories of entities connected in the graph. |
| `graph_entity` | string? | Entity to expand from. With it the query search stays unfiltered. |
| `entity` | string? | Only return memories of this entity. |
| `mem_type` | enum? | Only return memories of this type. |
| `include_global` | bool | True (default) lets a project search also see global memories. |
| `fields` | list? | Only return these columns, e.g. `["id", "content"]`. |
| `updated_after` | string? | Only memories changed at or after this ISO date. |
| `metadata_filter` | object? | Only memories whose metadata contains these key/value pairs. |
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
one, so the caller does not debug the wrong filter.

A project that does not exist is not silently ignored: the search falls back
to the global scope and the answer says so in `warnings`.

### `list_memories`

Browse without a search. `order_by` is `updated_at` (default), `created_at`,
`access_count` or `importance`. `contains` filters on a case-insensitive
literal substring - the right tool for "which memory contains exactly X",
where the semantic ranking of `recall` is not needed. `memory_id` looks a single
id up directly in the selected scope(s). `offset` pages through a scope; the
answer reports `total`, `has_more` and `next_offset`. `include_deleted=true`
lists soft-forgotten rows and includes their `deleted_at`, which is how an agent
finds an id for `restore_memory`. Forgotten rows carry the machine-readable
`forgotten: true` flag; the `__forgotten__` sentinel is removed from
`superseded_by` in those rows (it was overwriting the real value). Superseded
history stays hidden unless `include_superseded=true` is also passed - the two
flags are orthogonal.

Without `project` (or with `project="*"`) every scope is listed;
`project="global"` lists only the shared scope. `total` is computed after the
cross-scope dedup, so `total`, `count`, `has_more` and `next_offset` always
describe the same row set. `include_relation_context=false` hides the
relation-context sentences of the graph edges, which repeat what the edge
already says.

### `get_entity`

`entity` (`name` is kept as alias), `project`, `include_global`,
`include_superseded`, `fields`, `limit` (default 100), `offset`.
Memory-only entities have no graph node;
`details.type` is then `memory-only`; a graph node's `details.type` is `graph`
with the raw AGE label of its scope in `details.label`. A missing entity
answers with `status: "not_found"`. The `memories` list is sorted by `updated_at`
descending, newest change first. It also contains the relation-context
sentences of the entity's edges - they are stored under the synthetic entity
`<from>_<relation>_<to>`, so without this they would not show up here at all.
The answer reports `total`, `count`, `offset`, `has_more` and `next_offset`,
and a truncated answer says so in `warnings` instead of looking complete.
`total` also counts the relation-context sentences of the entity's edges.
`project="*"` is rejected: the answer describes exactly one entity, so there
is no single scope to report; omit `project` for the global scope or name one.

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
`memory-only`. `count` and `total` always agree, and an entity that has a
graph node is never reported as memory-only.

### `init_context`

The session bootstrap. `project="*"` includes all projects, `detail="full"`
returns every column except the internal ones and the search score
annotations, `include_global=false` excludes global memories.

`projects` names every scope the answer covers, including the shared one
(`"global"`). `active_memories` counts the rows a search would return,
`total_memories` counts every row of the scope - the difference is
`superseded_memories` plus `forgotten_memories`. `graph_entities_count` counts
the graph nodes of the covered scopes, not of the whole database.

### `list_projects`

Lists every scope with `memories`, `total_memories`, `superseded_memories`,
`forgotten_memories`, `entities` and `relations`. `include_empty=false` hides
scopes without active memories.

## Graph tools

| Tool | Description |
|------|-------------|
| `relate` | Directed relation via Cypher (auto-creates entities); `project` selects the project graph |
| `relate_batch` | Many relations in one transaction (MERGE, idempotent, per-edge `weight`/`context`/`project`) |
| `unrelate` | Delete a single directed relation |
| `find_connections` | Graph traversal (outgoing/incoming/both), results carry `direction` |
| `find_paths` | Shortest paths between two existing entities (`project`-scoped); edges are traversed in both directions, so a path can run against an edge. Missing entities return `not_found`, no path returns a warning |

- `relate` takes `from_entity`, `to_entity` (not `from`/`to`), `relation_type`
  and `weight` (0..1). The weight scales the retrieval boost of `graph_boost`.
- `relate_batch` takes `edges: [{from_entity, to_entity, relation_type,
  weight?, context?, project?}]`.
- `unrelate` takes `from_entity`, `to_entity`, `relation_type`, `project`.
- `find_connections` takes `entity` (`entity_name` is kept as alias) and
  reports the searched scope when the entity is missing.

## Maintenance tools

| Tool | Description |
|------|-------------|
| `contradictions_report` | Detect conflicting facts (LLM NLI + vector pre-filter); per entity or scope-wide (all active memories compete, cross-entity pairs included, each pair carries the entity of both facts and `cross_entity`); `resolve="newest"` keeps the newer fact **of same-entity pairs only** and supersedes the older, cross-entity pairs stay active (warning); a partially failed NLI names the unjudged pairs in `nli_failures` (`fact_id`, `neighbor_ids`, `reason`) |
| `export_memories` | Backup as JSON/JSONL (`format`, `project`, `include_embeddings`, `include_superseded`, `include_graph`); payload in `data`; warns when the default export leaves superseded/forgotten rows out |
| `import_memories` | Restore from JSON/JSONL (`data` accepts object or string, `project` override, `format`, `skip_duplicates`, `fuzzy_threshold`); IDs are preserved; an id that already exists is always skipped; `graph_entities`/`graph_relations` count processed records, `*_created` the new ones; a non-empty `skipped` comes with `skipped_reasons` and a warning |
| `drop_project` | Remove a project scope completely (memory table, registry entry, graph nodes, mirror rows); `dry_run` previews; the shared scope cannot be dropped |
| `server_health` | Check Postgres + Inference connectivity; reports `version`, `code_fingerprint` (detects a stale container) and `started_at` |

See [Backup & Restore](backup.md) for the export/import contract and
[Retrieval](retrieval.md) for how contradictions are detected.

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
