# Configuration

## Environment variables

| Env Var | Default | Description |
|---------|---------|-------------|
| `MEMORY_DATABASE_URL` | `postgresql://memory@localhost:5432/memory` | Postgres DSN (supply the password here) |
| `POSTGRES_PASSWORD` | - (required by compose) | Password for the bundled Postgres; compose also builds the DSN from it |
| `POSTGRES_USER` / `POSTGRES_DB` | `memory` | Bundled Postgres user and database |
| `MEMORY_INFERENCE_URL` | `http://localhost:8001` | Inference API |
| `MCP_TRANSPORT` | `stdio` | `stdio`, `sse` or `streamable-http` |
| `MCP_HOST` | `0.0.0.0` | Bind address (`sse`/`streamable-http` only) |
| `MCP_PORT` | `8766` | Port (`sse`/`streamable-http` only) |
| `MCP_AUTH_TOKEN` | - (required for HTTP) | Shared bearer token; every HTTP request must send `Authorization: Bearer <token>` |
| `MCP_ALLOW_INSECURE_HTTP` | `false` | Allow an unauthenticated non-loopback HTTP bind. Sandbox only; never in production |
| `FTS_LANGUAGE` | `english` | PostgreSQL text-search dictionary; invalid names fall back to `english` |
| `MCP_TOOLSET` | `full` | `full` (28 tools) or `core` (16 daily read/write/graph tools); an unknown value warns and falls back to `full` |
| `INFERENCE_DEVICE` | `auto` | `auto` (CUDA when available, else CPU), `cpu` or `cuda`; `cuda` still falls back if no device is visible |
| `TORCH_INDEX_URL` | `https://download.pytorch.org/whl/cu126` | Build arg for the inference image; use `https://download.pytorch.org/whl/cpu` for a CPU-only image |
| `RERANK_ENABLED` | `true` | Local cross-encoder reranking of ambiguous first-stage results |
| `RERANK_MODEL` | `mixedbread-ai/mxbai-rerank-base-v1` | Cross-encoder model; the inference service loads it, pass the same value to both containers |
| `RERANK_MAX_LENGTH` | `256` | Token limit per query/document pair (64-512) |
| `RERANK_TOP_K` | `10` | Candidates submitted to the reranker when the first stage is ambiguous (2-50) |
| `RERANK_TRIGGER_SCORE` | `0.85` | A best relevance at or above this value is trusted without reranking |
| `RERANK_TRIGGER_MARGIN` | `0.10` | A lead over the runner-up at or above this value is trusted without reranking |
| `RERANK_ALWAYS` | `false` | Rerank every query instead of only ambiguous ones |
| `RERANK_LOGIT_BIAS` | `1.0` | Offset before the sigmoid cap; the raw reranker logits are not calibrated (-5..5) |
| `LLM_PROVIDER` | `none` | `groq`, `openai`, `anthropic`, `ollama`, `openai-compatible`, `auto` or `none`. Egress is opt-in: `none` (the default) disables NLI |
| `NLI_MAX_WORKERS` | `4` | Documents of one `/nli` request judged concurrently (1-16). Higher values cut scan latency but raise provider concurrency |
| `LLM_MODEL` | provider default | NLI model (empty = provider default) |
| `LLM_REASONING_EFFORT` | `low` for Groq/gpt-oss | Reasoning budget (`low`/`medium`/`high`; `none` disables). gpt-oss otherwise spends the token budget on hidden reasoning and returns empty content |
| `LLM_API_KEY` | – | Generic key; overrides the provider-specific key |
| `LLM_BASE_URL` | provider default | Base URL for OpenAI-compatible or Ollama endpoints |
| `GROQ_API_KEY` | – | Groq key (backwards compatible) |
| `GROQ_MODEL` | `openai/gpt-oss-20b` (compose) | Groq model (backwards compatible) |
| `OPENAI_API_KEY` | – | Used when `LLM_PROVIDER=openai` |
| `ANTHROPIC_API_KEY` | – | Used when `LLM_PROVIDER=anthropic` |
| `OLLAMA_BASE_URL` | `http://host.docker.internal:11434/v1` | Ollama endpoint |
| `CONTRADICTION_SCAN_BUDGET_SECONDS` | `40` | Wall-clock budget for the whole `contradictions_report` scan (candidate loading plus the NLI leg); on exhaustion the scan stops early and names unjudged pairs in `nli_failures` |
| `CONTRADICTION_NEIGHBORS` | `6` | Vector neighbours fetched per memory per searched scope table (1-20); the closest pairs are judged first |
| `CONTRADICTION_NLI_WORKERS` | `6` | Bounded concurrency of the NLI calls (1-16) |

## HTTP authentication

The `streamable-http` and `sse` transports expose every tool, including
`drop_project`, `prune_memories` and `import_memories`. They are protected by a
shared bearer token:

```dotenv
MCP_AUTH_TOKEN=<generate with: python -c "import secrets; print(secrets.token_urlsafe(32))">
```

Every request must send `Authorization: Bearer <MCP_AUTH_TOKEN>`; requests
without it are answered with HTTP 401 before they reach a tool. `stdio` is
unaffected (the client owns the process).

Startup is fail-closed: an HTTP transport bound to a non-loopback address
without a token refuses to start. `MCP_ALLOW_INSECURE_HTTP=true` disables that
check for a trusted sandbox. Docker Compose publishes the MCP, Postgres and
inference ports on `127.0.0.1` only; to expose the MCP port to another machine,
change the mapping to `"8766:8766"` and keep the token set.

The Docker Compose file requires `POSTGRES_PASSWORD` as well and refuses to
start while it is empty.

## LLM providers

`/nli` uses one provider at a time. The default is Groq; the legacy
`GROQ_API_KEY`/`GROQ_MODEL` variables keep working.

| Provider | Default model | Default endpoint |
|----------|---------------|------------------|
| `groq` | `openai/gpt-oss-20b` | `https://api.groq.com/openai/v1` |
| `openai` | `gpt-4o-mini` | `https://api.openai.com/v1` |
| `anthropic` | `claude-3-5-haiku-latest` | `https://api.anthropic.com/v1` |
| `ollama` | `llama3.2` | `http://host.docker.internal:11434/v1` |
| `openai-compatible` | – (set `LLM_MODEL`) | – (set `LLM_BASE_URL`) |

```dotenv
# OpenAI
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...

# Anthropic
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...

# Ollama on the host
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://host.docker.internal:11434/v1

# Any OpenAI-compatible endpoint (LM Studio, vLLM, OpenRouter, ...)
LLM_PROVIDER=openai-compatible
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_API_KEY=...
LLM_MODEL=...
```

The default is `LLM_PROVIDER=none`: contradiction detection stays off until a
provider is chosen, so memory content is not sent anywhere by accident.
`LLM_PROVIDER=auto` picks the first configured key (Groq, OpenAI, Anthropic),
then `LLM_BASE_URL`, then `OLLAMA_BASE_URL`. Outside the bundled compose file
(which passes `none` explicitly) an omitted `LLM_PROVIDER` still detects a
configured provider key, so existing deployments keep working; a bare install
stays on `none`. `LLM_PROVIDER=none` disables
contradiction detection. Without a usable provider `/nli` returns placeholder
`0.0` scores together with an `error`; the MCP client raises instead of
treating those zeros as judgments. `contradictions_report` short-circuits when
health reports no configured provider and returns `nli_configured=false` plus
a warning; write tools keep the fact and warn
(`contradiction check skipped: NLI unavailable: ...`) in `warnings` and
`contradictions.skipped`. A partial NLI failure with judged candidates keeps
the verdict and reports the unjudged ones as `contradictions.unjudged` plus a
`contradiction check incomplete` warning; only a failure that leaves nothing
judable skips the check.

### Data egress

`groq`, `openai` and `anthropic` are hosted services. A contradiction check
sends the two compared memory texts to the provider (`/nli` builds
`Fact A: ...` / `Fact B: ...` and posts it to the provider endpoint). This is
the only path that moves stored content off the machine: embeddings and
reranking run locally. With the default `LLM_PROVIDER=none` no content leaves
the machine at all.

Use `LLM_PROVIDER=ollama` or `openai-compatible` with a local endpoint, or
`LLM_PROVIDER=none`, to keep memory content local. `server_health` reports the
active `llm_provider`, and `data_egress` is `true` when that provider is a
hosted one; it adds a `data egress` warning in that case.

Stored content still leaves the machine when a tool result is returned to the
calling agent - that is the client's transport, not this server.

Embeddings always run locally - no key needed. The default image is CUDA
enabled and uses an NVIDIA GPU when one is exposed; without a GPU it runs on
CPU.

## Docker

| Service | Image / base | Notes |
|---------|--------------|-------|
| **Postgres** | `postgres:17-bookworm` + pgvector v0.8.1 + Apache AGE 1.7.0 | `shm_size: 1g` — AGE needs more shared memory |
| **Inference** | FastAPI + sentence-transformers + PyTorch CUDA 12.6 | GPU via the compose override, CPU fallback, no GPU required |
| **Loci MCP** | Python FastMCP server with uvicorn | Streamable HTTP transport |

After changing the embedding model or prompts, rebuild both images and then
re-embed:

```bash
docker compose build inference loci-mcp && docker compose up -d
docker compose exec loci-mcp python -m mcp_server.reembed
```

On startup, the MCP server runs idempotent migrations (`deleted_at` backfill,
sentinel migration, dedup of duplicate AGE edges).

## GPU

The inference image ships a CUDA 12.6 build of PyTorch and picks the first
NVIDIA GPU automatically (`INFERENCE_DEVICE=auto`). The base compose file does
not pass a GPU into the container, so `docker compose up -d` stays on CPU. To
expose the GPU:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

This needs a driver and the NVIDIA Container Toolkit (`nvidia-smi` works and
`docker info` lists the `nvidia` runtime). The override only adds `gpus: all`
to `inference`; the image and the other services are unchanged.

`INFERENCE_DEVICE=cpu` forces the CPU path; `cuda` requests the GPU and still
falls back to CPU when no device is visible. Check what is active:

```bash
curl -s localhost:8001/health | jq '{device, embedder_device, cuda_device, reranker_device}'
```

The image is built from `https://download.pytorch.org/whl/cu126` (build arg
`TORCH_INDEX_URL`). A CPU-only image is smaller and needs no CUDA libraries:

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu docker compose build inference
```

On an RTX 2080 (8 GB) versus the same container on CPU, one text embeds in
36-49 ms instead of 375-485 ms, and reranking ten documents takes 131 ms
instead of 2.0 s. The CUDA image also runs on CPU-only hosts.

## Inference API

The MCP server talks to a small local HTTP service that does the embeddings
and calls the LLM. It can be used directly:

```python
import requests

# Embedding (Jina v5 nano, 239M) -> 768d normalized vector
# mode: "query" (search query) or "document" (stored content, default)
r = requests.post("http://localhost:8001/embed", json={
    "texts": ["This is a test", "Another text"],
    "mode": "document"
})
vectors = r.json()["embeddings"]

# NLI / contradiction detection (configured LLM provider)
r = requests.post("http://localhost:8001/nli", json={
    "query": "Max lives in Berlin",
    "documents": ["Max lives in Hamburg"]
})
contradiction_scores = r.json()["contradiction_scores"]
# `error` is set when the scores are NOT a real judgment (provider down or
# not configured) - treat zeros from an errored run as "unknown", never as
# "no contradiction".
error = r.json().get("error")
```

`/health` reports the status of both services, including `rerank_model`,
`rerank_loaded` and `rerank_error`, plus the active device (`device`,
`embedder_device`, `cuda_device`, `reranker_device`).

## Scoring constants

The tuning knobs live in `mcp_server/config.py`:

| Constant | Value | Meaning |
|----------|-------|---------|
| `DECAY_HALF_LIFE_DAYS` | 90 | How fast memories fade. |
| `DECAY_ACCESS_CAP` | 20 | Accesses that count for the decay bonus. |
| `DECAY_WEIGHT` | 0.3 | Maximum share of the score that decay may take. |
| `FTS_WEIGHT` | 0.6 | Weight of the full-text signal in the relevance score. |
| `RRF_K` | 60 | RRF smoothing constant. |
| `DEFAULT_MIN_SIMILARITY` | 0.30 | Default vector-hit floor, just above the embedding model's 0.288 noise floor. |
| `DEFAULT_DUPLICATE_THRESHOLD` | 0.85 | Similarity threshold of `dedupe="semantic"`. |
| `NEAR_DUPLICATE_THRESHOLD` | 0.75 | Similarity at which a stored write is reported as a near-duplicate in `warnings`. |
| `SUPERSEDE_SIMILARITY_MIN` | 0.5 | Topic similarity below which `supersede_last` refuses. |
| `PROJECT_SCOPE_BOOST` | 0.05 | Tie-breaker in favour of project hits. |
