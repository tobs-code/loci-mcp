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

The payload is returned in `data` and can be passed to `import_memories`
unchanged.

The JSON format (version `0.2`, `"type": "memory_export"`) contains IDs,
`superseded_by`, `deleted_at`, timestamps, `access_count`, the project of every
record and — for JSON — the complete graph (entities with scope plus relations
with weight and scope). JSONL is flat: one record per line, no graph.

## Import

```text
import_memories(data=<export data>, format="json")
```

| Parameter | Meaning |
|-----------|---------|
| `data` | The export payload, as the object or as its JSON string. |
| `project` | Target scope for all records. Without it every record is restored into the scope it was exported from. |
| `format` | `json` (default) or `jsonl`. |
| `skip_duplicates` | Skip records whose id already exists instead of inserting them again (default `true`). |
| `fuzzy_threshold` | Similarity threshold for semantic dedup of records without an id. |

Records keep their `id`, timestamps and `superseded_by`, so a re-import is
idempotent and restores the exact state. Project tables are recreated on
demand, entities and relations are upserted with MERGE.

Re-importing the same file reports the existing ids as `skipped`;
`skip_duplicates=false` inserts them again instead.

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
active memories. It also leaves soft-forgotten rows out. The answer names how
many superseded and forgotten rows were excluded
(`warnings`), so a backup is never silently incomplete. For a real backup take
the history with it:

```text
export_memories(format="json", include_superseded=true, include_embeddings=true)
```

Without the embeddings the memories have to be embedded again on import,
which is fine but slower and can change the vectors slightly.
