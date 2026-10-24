# Backup & Restore

`export_memories` and `import_memories` move a complete memory store between
systems — including the graph.

## Export

```text
export_memories(format="json", project="api")
```

| Parameter | Meaning |
|-----------|---------|
| `format` | `json` (default) returns an object with the records and the graph; `jsonl` returns one record per line (no graph). |
| `project` | Scope to export. Without it **every** scope is exported and each record carries its `project`. |
| `include_embeddings` | Include the vector of every memory (large). |
| `include_superseded` | Include superseded (invalidated) memories. |
| `include_graph` | Include entities and relations (JSON exports only). |
| `entity` | Only records of this entity. |
| `mem_type` | Only records of this type. |
| `updated_after` | Only records changed at or after this ISO date. |
| `metadata_filter` | Only records whose metadata contains these key/value pairs. |

The payload is returned in `data` and can be passed to `import_memories`
unchanged.

The JSON format (version `0.3`, `"type": "memory_export"`) contains IDs,
`superseded_by`, TTL (`expires_at`), `deleted_at`, timestamps, `access_count`,
the project of every record and — for JSON — the complete graph (entities with
scope plus relations with weight and scope). JSONL is flat: one record per
line, no graph.

The default export leaves superseded/forgotten history out and says how many
rows it excluded. Expired rows are included with their `expires_at` and an
explicit warning, so they cannot silently become active again on import. A
forgotten row carries `deleted_at` and `superseded_by: null`; the internal
`__forgotten__` sentinel is never exported (writing it back would make the row
look superseded by a fake memory id).

## Import

```text
import_memories(data=<export data>, format="json")
```

| Parameter | Meaning |
|-----------|---------|
| `data` | The export payload, as the object, as a list of records, or as its JSON/JSONL string. |
| `project` | Target scope for all records. Without it every record is restored into the scope it was exported from. |
| `format` | `json` (default) or `jsonl`. |
| `skip_duplicates` | Skip records whose content already exists when the record has **no** id (default `true`). A record whose id already exists is always skipped - that is what makes a re-import idempotent. |
| `fuzzy_threshold` | Similarity threshold for semantic dedup of records without an id. |
| `dry_run` | Validate and count without writing anything; `would_create` names how many records an import would add. |

Records keep their `id`, TTL (`expires_at`), timestamps and `superseded_by`, so
a re-import is idempotent and restores the exact state. A `superseded_by` link
whose target is missing from the backup is cleared with a warning (otherwise the
row would stay unreadable forever; `verify_integrity` reports the same case).
A kept link whose target lives in ANOTHER scope (a valid cross-scope supersede
that was not part of this import) is reported as `cross_scope_supersede_links`
plus a warning naming record and scope: the link stays, but the restored rows
are history-hidden inside this scope until that scope is imported too
(`drop_project` of the missing scope clears the link then). A `dry_run` lists
the links it found and warns when the real import would clear one (a target
that exists nowhere), so a preview never looks cleaner than the import it
describes.
Project tables are recreated on
demand, entities and relations are upserted with MERGE. `graph_entities` and
`graph_relations` count the processed records; `graph_entities_created` and
`graph_relations_created` count the ones that did not exist before (MERGE
updates an existing node/edge instead of duplicating it).

Re-importing the same file reports the existing ids as `skipped`. A non-empty
`skipped` comes with `skipped_reasons` (`id_exists`, `duplicate_content`,
`empty_content`, `invalid`) and a warning, so an idempotent re-import is
distinguishable from a backup that silently lost rows. Records that fail
validation (bad metadata, blank project, broken row) are counted in `invalid`
and listed in `errors`; they are a subset of `skipped`, so
`imported + skipped` still equals the number of records.

Every record runs in its own savepoint, so one broken row does not abort the
file: a record whose embedding has the wrong type or dimension, or whose insert
fails, is reported in `errors` (named by its id) and the remaining records still
import. The same holds for a malformed JSONL line: it is collected as
`skipped_reasons.invalid` with `jsonl line N` in `errors` while the valid lines
are imported.

## Pass the payload as a string

The string form is the safer one. Some MCP clients repack a JSON array inside
a free-form object into `{"item": ...}`, which is still valid JSON but no
longer a list. The server tolerates that shape, but why rely on it:

```text
export_memories(format="json", project="api")
import_memories(data=<the JSON string of the export payload>, format="json")
```

## A backup that keeps the history

`include_superseded` defaults to `false`, so a plain export contains only the
active memories. It also leaves soft-forgotten rows out, and it warns when it
carries expired rows. The answer names how many superseded, forgotten and
expired rows were excluded (`warnings`), so a backup is never silently
incomplete. For a real backup take
the history with it:

```text
export_memories(format="json", include_superseded=true, include_embeddings=true)
```

Without the embeddings the memories have to be embedded again on import,
which is fine but slower and can change the vectors slightly.
