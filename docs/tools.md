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
| `status` | `stored`, `created`, `duplicate`, `updated`, `deleted`, `soft_forgotten`, `not_found`, `partial`, `error`, `ok` |
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
| `forget` | Soft delete (`deleted_at` + sentinel) or hard delete; `entity` also removes the graph node and that scope's memories; `counts` splits memories/entities/relations |

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
| `supersede_last` | bool | Replace the newest memory of the same entity - only if the new fact is about the same topic (similarity >= 0.5), otherwise the fact is stored additionally and a warning is returned. |
| `check_contradictions` | bool | NLI-check against the existing memories of the same entity and auto-supersede contradictions. |
| `dedupe` | enum | `exact` (default, content hash), `semantic` (also vector similarity), `off`. |
| `fuzzy_threshold` | number? | Similarity threshold for `dedupe="semantic"` (default 0.9). |

The answer carries `superseded` (the ids that were invalidated) and, with
`check_contradictions`, a `contradictions` object. A duplicate answer names the
mode that matched (`duplicate_mode`) and the measured similarity.

### `remember_batch`

`facts` is a list of objects; only `content` is required, everything else works
like in `remember`. Invalid facts are reported individually, the rest is still
stored (`status: "partial"`).

### `update_memory`

`memory_id`, `content`, `importance`, `metadata`, `entity`, `mem_type`,
`project`. Content changes are re-embedded. Without `project` the memory is
looked up in every scope; a memory id is unique per scope, so an id that
exists in several scopes is refused and the scopes are named (see
[Scopes](scopes.md)).

### `forget`

`memory_id` or `entity` (`entity_name` is kept as alias), `soft` (default
`true`), `project`, `dry_run`.

- `entity="*"` plus `project` empties a whole scope; `project` is required for
  that, so a stray call cannot wipe the database.
- `dry_run=true` only reports the counts.
- The answer carries `counts` (memories/entities/relations separately) and
  `scopes` (the scopes the id was found in).

## Read tools

| Tool | Description |
|------|-------------|
| `recall` | Hybrid vector + FTS (RRF fusion, project + global); `hybrid`, `min_similarity`, `include_decay`/`min_decay_score`, `include_superseded`/`only_superseded`, `include_global`, `graph_boost`/`graph_entity`, `fields`, `updated_after`, `metadata_filter` |
| `list_memories` | Browse memories without a query (newest first); `entity`, `mem_type`, `project`, `include_global`, `fields`, `updated_after`, `metadata_filter`, `order_by`, `offset` |
| `get_entity` | Entity details (including graph connections) + all linked memories; `include_global`, `include_superseded`, `fields` |
| `list_entities` | Browse entities (`entity_type`, `project`, `include_global`, `offset`); memory-only entities carry `type="memory-only"` |
| `init_context` | Session bootstrap: recent memories (slim, near-duplicates removed, each with its `project`), counters, projects, graph entity count, health; `detail="slim"\|"full"`, `include_global` |
| `list_projects` | List the known project scopes with their sizes (memories, entities, relations) |

### `recall`

The search tool. `query` is required and non-empty.

| Parameter | Type | Meaning |
|-----------|------|---------|
| `query` | string | Search text, embedded as a query and matched against the vector and full-text indexes. |
| `project` | string? | Scope. Without it only global memories are searched. |
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

The answer reports `candidates` and `filtered_by_similarity`, plus
`fulltext_only` when a hit came only from the full-text leg. See
[Retrieval](retrieval.md) for the score fields.

### `list_memories`

Browse without a search. `order_by` is `updated_at` (default), `created_at`,
`access_count` or `importance`. `contains` filters on a case-insensitive
literal substring - the right tool for "which memory contains exactly X",
where the semantic ranking of `recall` is not needed. `offset` pages through a
scope; the answer reports `total`, `has_more` and `next_offset`.

### `get_entity`

`entity` (`name` is kept as alias), `project`, `include_global`,
`include_superseded`, `fields`. Memory-only entities have no graph node;
`details.type` is then `memory-only`. A missing entity answers with
`status: "not_found"`.

### `list_entities`

`entity_type`, `project`, `include_global`, `limit`, `offset`. The list is
sorted by name, so `offset` is stable.

### `init_context`

The session bootstrap. `project="*"` includes all projects, `detail="full"`
returns every column except the internal ones and the search score
annotations, `include_global=false` excludes global memories.

### `list_projects`

Lists every scope with `memories`, `total_memories`, `entities` and
`relations`. `include_empty=false` hides scopes without active memories.

## Graph tools

| Tool | Description |
|------|-------------|
| `relate` | Directed relation via Cypher (auto-creates entities); `project` selects the project graph |
| `relate_batch` | Many relations in one transaction (MERGE, idempotent, per-edge `weight`/`context`/`project`) |
| `unrelate` | Delete a single directed relation |
| `find_connections` | Graph traversal (outgoing/incoming/both), results carry `direction` |
| `find_paths` | Shortest paths between two existing entities (`project`-scoped); missing entities return `not_found` |

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
| `contradictions_report` | Detect conflicting facts for **one entity** (LLM NLI + vector pre-filter); `entity` is required |
| `export_memories` | Backup as JSON/JSONL (`format`, `project`, `include_embeddings`, `include_superseded`, `include_graph`); payload in `data` |
| `import_memories` | Restore from JSON/JSONL (`data` accepts object or string, `project` override, `format`, `skip_duplicates`, `fuzzy_threshold`); IDs are preserved |
| `drop_project` | Remove a project scope completely (memory table, registry entry, graph nodes, mirror rows); `dry_run` previews |
| `server_health` | Check Postgres + Inference connectivity |

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
