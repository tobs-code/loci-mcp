# Architecture

## Stack

| Part | Technology |
|------|------------|
| Storage | PostgreSQL 17 |
| Vector search | pgvector v0.8.1 (768d, cosine) |
| Graph | Apache AGE 1.7.0 (Cypher over `memory_graph`) |
| Server | Python + FastMCP, Streamable HTTP or stdio |
| Embeddings | Jina v5 nano (239M, local; CUDA when available, CPU fallback) |
| Contradiction check | `/nli` against a configurable LLM provider |

## Schema

| Table | Purpose |
|-------|---------|
| `memories` | Global text content with vector embedding (768d) + FTS + decay |
| `mem_<suffix>` | Project-specific memories (created dynamically on demand) |
| `projects` | Registry suffix → original project name |
| `entities` | Graph nodes (people, concepts, places...) with `project` scope |
| `relations` | Graph edges between entities (with `project` scope) |

The AGE graph `memory_graph` is created automatically via `create_graph()`.

## Direct SQL

The data is plain Postgres, so you can query it directly.

### Vector search

```sql
SELECT id, content, metadata
FROM memories
ORDER BY embedding <=> '[0.1, 0.2, ...]'
LIMIT 5;
```

Small scopes (fewer than 10,000 rows) are scanned exactly. An IVFFlat index
is built only once a table is large enough to benefit: on a tiny table it
probes a single list and can miss nearby rows entirely, which silently hides
near-duplicates and contradiction candidates. Project tables follow the same
rule.

### Graph queries

```sql
SELECT * FROM cypher('memory_graph', $$
    MATCH (p:Person)-[:KNOWS]->(friend)
    RETURN p.name, friend.name
$$) AS (name agtype, friend_name agtype);
```

## Module map

| File | Purpose |
|------|---------|
| `Dockerfile` | Custom Postgres image with pgvector + AGE |
| `docker-compose.yml` | Container orchestration |
| `init.sql` | Schema + extensions + AGE graph |
| `mcp_server/` | Python MCP server (FastMCP + pgvector + AGE) |
| `mcp_server/server.py` | Public facade: registers the tools and starts the transports |
| `mcp_server/app.py` | FastMCP instance, server instructions and the envelope-aware tool classes |
| `mcp_server/config.py` | Memory types, scopes and scoring constants |
| `mcp_server/envelope.py` | Uniform response envelope shared by all tools |
| `mcp_server/projection.py` | Row projection (`fields`) and field validation |
| `mcp_server/db.py` | Schema, migrations, connection handling |
| `mcp_server/store.py` | `MemoryStore`: connection, cursor, savepoint and transaction lifetime plus the write/lookup operations |
| `mcp_server/store_rows.py` | Entity/graph/relation-context reads of a scope, taking the store |
| `mcp_server/store_updates.py` | Partial row updates, restore and entity/context forget, taking the store |
| `mcp_server/store_maintenance.py` | Supersede-link repair, pruning, mirror counts and scope DDL, taking the store |
| `mcp_server/write_pipeline.py` | The single write order: `WriteIntent` → `WriteOutcome`, used by `remember` and `remember_batch` |
| `mcp_server/scope_catalog.py` | Scope naming: project slugs, scope labels, table names and registry lookup |
| `mcp_server/schema_migrator.py` | Project-table DDL and full-text/vector index migration |
| `mcp_server/scope.py` | `ScopeSelection` / `resolve_scope`: read-scope resolution |
| `mcp_server/lifecycle.py` | Active/TTL predicates, visibility SQL and row counts |
| `mcp_server/scoring.py` | Decay, RRF helpers and ID/content hashing |
| `mcp_server/duplicates.py` | Exact/semantic duplicates, subject guards and cross-scope dedupe |
| `mcp_server/browse.py` | The single browse pipeline: visibility, cross-scope dedupe, paging and page counters |
| `mcp_server/contradiction_scan.py` | Candidate selection, budgeted NLI judging and resolution planning |
| `mcp_server/storage.py` | Supersede, duplicate and access helpers shared by the adapters |
| `mcp_server/search.py` | `SearchRequest`, vector + FTS RRF search and graph boost |
| `mcp_server/graph.py` | MCP adapter for the relation tools (schemas and envelope) over `GraphStore` |
| `mcp_server/graph_store.py` | `GraphStore`: sole owner of the graph lifecycle (Cypher, mirror rows, relation-context memories, orphan cleanup) |
| `mcp_server/backup.py` | Export/import of memories and the graph |
| `mcp_server/tools/` | Write, read, scope and maintenance MCP tools |
| `mcp_server/inference_gateway.py` | One inference interface: `HttpInference` (service adapter) and `FakeInference` (deterministic tests) |
| `mcp_server/inference_client.py` | HTTP transport for `/embed`, `/rerank`, `/nli`, `/health` |
| `mcp_server/reembed.py` | Re-embed all memories after a model/prompt change |
| `tests/` | Integration tests against an isolated `loci_test` database |
| `inference/Dockerfile` | Inference service with sentence-transformers |
| `inference/app.py` | FastAPI server (`/embed`, `/nli`, `/health`) |
| `inference/llm.py` | Provider abstraction for Groq, OpenAI, Anthropic, Ollama and OpenAI-compatible APIs |
| `AGENTS.md` | Full tool documentation for AI agents |

## Seams

Six modules own a decision that used to be repeated by their callers. Each one
takes its adapter as an argument, so a test can replace Postgres, the graph or
the inference service without patching an import path.

| Seam | Interface | Callers / adapters |
|------|-----------|--------------------|
| `inference_gateway.py` | `embed`, `rerank`, `nli`, `nli_detailed`, `health` | `HttpInference` (service) and `FakeInference` (tests) |
| `store.py` (+ `store_rows` / `store_updates` / `store_maintenance`) | `MemoryStore`: connection and savepoint lifetime, plus memory-table operations | All persistence; fake store in tests |
| `write_pipeline.py` | `WriteIntent` → `WriteOutcome` | `remember`, `remember_batch` |
| `browse.py` | `BrowseRequest` → `BrowsePage`; `paginate`, `dedupe_rows`, `project_rows` | `recall`, `list_memories`, `get_entity`, `list_entities`, `init_context` |
| `contradiction_scan.py` | `ContradictionScan.scan` / `plan_resolution` | `contradictions_report`, `remember(check_contradictions)` |
| `graph_store.py` | `GraphStore`: nodes, edges, relation-context memories, orphans | `graph.py` tools, `search.py` boost, `backup.py`, `server.py`, `write_pipeline.py` |

The seams own ordering and policy; the tool modules keep validation, argument
wording and envelope rendering. None of the seams opens a second connection:
the caller passes the store (or cursor) and owns the transaction.

## Request flow

```
Agent ──MCP──> FastMCP tool ──> envelope
                    │
                    ├─ write  ──> write_pipeline ──> MemoryStore ──> Postgres
                    ├─ read   ──> search + browse ──> MemoryStore ──> pgvector + FTS
                    ├─ graph  ──> GraphStore ──> Apache AGE
                    └─ any    ──> inference_gateway ──> /embed, /rerank, /nli
```

Every tool returns the same envelope, so a caller never has to branch on the
shape of the answer. See [Tool Reference](tools.md).

## Testing

Tests run against an isolated `loci_test` database (created and dropped
automatically) and a local fake inference server, so no LLM key is needed.
The seams above also have unit tests that inject a fake store, a fake
inference gateway and a fake clock, so ordering, paging and budget rules are
exercised without a database or network.

```bash
pip install -r requirements-dev.txt
python -m pytest
```

`python -m pyflakes mcp_server/ tests/` is expected to stay clean.
