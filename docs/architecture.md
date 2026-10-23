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
| `mcp_server/store.py` | `MemoryStore`: connection, cursor and transaction lifetime |
| `mcp_server/scopes.py` | Project slugs, scope labels and per-project tables |
| `mcp_server/scope.py` | `ScopeSelection` / `resolve_scope`: read-scope resolution |
| `mcp_server/lifecycle.py` | Active/TTL predicates, visibility SQL and row counts |
| `mcp_server/scoring.py` | Decay, RRF helpers and ID/content hashing |
| `mcp_server/duplicates.py` | Exact/semantic duplicates, subject guards and cross-scope dedupe |
| `mcp_server/storage.py` | Store, supersede and access helpers |
| `mcp_server/search.py` | `SearchRequest`, vector + FTS RRF search and graph boost |
| `mcp_server/graph.py` | Apache AGE relation tools (orchestration only) |
| `mcp_server/graph_store.py` | Sole owner of Cypher and graph mirror SQL |
| `mcp_server/backup.py` | Export/import of memories and the graph |
| `mcp_server/tools/` | Write, read, scope and maintenance MCP tools |
| `mcp_server/inference_client.py` | HTTP client for `/embed`, `/nli`, `/health` |
| `mcp_server/reembed.py` | Re-embed all memories after a model/prompt change |
| `tests/` | Integration tests against an isolated `loci_test` database |
| `inference/Dockerfile` | Inference service with sentence-transformers |
| `inference/app.py` | FastAPI server (`/embed`, `/nli`, `/health`) |
| `inference/llm.py` | Provider abstraction for Groq, OpenAI, Anthropic, Ollama and OpenAI-compatible APIs |
| `AGENTS.md` | Full tool documentation for AI agents |

## Request flow

```
Agent ──MCP──> FastMCP tool ──> envelope
                    │
                    ├─ embed()  ──> inference /embed ──> vector
                    ├─ search   ──> pgvector + FTS ──> RRF fusion ──> projection
                    ├─ graph    ──> Apache AGE (Cypher)
                    └─ postgres ──> per-project tables + global table
```

Every tool returns the same envelope, so a caller never has to branch on the
shape of the answer. See [Tool Reference](tools.md).

## Testing

Tests run against an isolated `loci_test` database (created and dropped
automatically) and a local fake inference server, so no LLM key is needed.

```bash
pip install -r requirements-dev.txt
python -m pytest
```

`python -m pyflakes mcp_server/ tests/` is expected to stay clean.
