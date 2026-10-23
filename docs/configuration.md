# Configuration

## Environment variables

| Env Var | Default | Description |
|---------|---------|-------------|
| `MEMORY_DATABASE_URL` | `postgresql://memory:geheim@localhost:5432/memory` | Postgres DSN |
| `MEMORY_INFERENCE_URL` | `http://localhost:8001` | Inference API |
| `MCP_TRANSPORT` | `stdio` | `stdio`, `sse` or `streamable-http` |
| `MCP_HOST` | `0.0.0.0` | Bind address (`sse`/`streamable-http` only) |
| `MCP_PORT` | `8766` | Port (`sse`/`streamable-http` only) |
| `FTS_LANGUAGE` | `english` | PostgreSQL text-search dictionary; invalid names fall back to `english` |
| `MCP_TOOLSET` | `full` | `full` (24 tools) or `core` (14 daily read/write/graph tools); an unknown value warns and falls back to `full` |
| `INFERENCE_DEVICE` | `auto` | `auto` (CUDA when available, else CPU), `cpu` or `cuda`; `cuda` still falls back if no device is visible |
| `TORCH_INDEX_URL` | `https://download.pytorch.org/whl/cu126` | Build arg for the inference image; use `https://download.pytorch.org/whl/cpu` for a CPU-only image |
| `RERANK_ENABLED` | `true` | Local cross-encoder reranking of ambiguous first-stage results |
| `RERANK_MODEL` | `mixedbread-ai/mxbai-rerank-base-v1` | Cross-encoder model; the inference service loads it, pass the same value to both containers |
| `RERANK_MAX_LENGTH` | `256` | Token limit per query/document pair (64-512) |
| `RERANK_TOP_K` | `10` | Candidates submitted to the reranker when the first stage is ambiguous (2-50) |
| `RERANK_TRIGGER_SCORE` | `0.85` | A best relevance at or above this value is trusted without reranking |
| `RERANK_TRIGGER_MARGIN` | `0.10` | A lead over the runner-up at or above this value is trusted without reranking |
| `RERANK_ALWAYS` | `false` | Rerank every query instead of only ambiguous ones |
| `LLM_PROVIDER` | `groq` | `groq`, `openai`, `anthropic`, `ollama`, `openai-compatible`, `auto` or `none` |
| `LLM_MODEL` | provider default | NLI model (empty = provider default) |
| `LLM_API_KEY` | – | Generic key; overrides the provider-specific key |
| `LLM_BASE_URL` | provider default | Base URL for OpenAI-compatible or Ollama endpoints |
| `GROQ_API_KEY` | – | Groq key (backwards compatible) |
| `GROQ_MODEL` | `openai/gpt-oss-20b` (compose) | Groq model (backwards compatible) |
| `OPENAI_API_KEY` | – | Used when `LLM_PROVIDER=openai` |
| `ANTHROPIC_API_KEY` | – | Used when `LLM_PROVIDER=anthropic` |
| `OLLAMA_BASE_URL` | `http://host.docker.internal:11434/v1` | Ollama endpoint |
| `CONTRADICTION_SCAN_BUDGET_SECONDS` | `25` | Wall-clock budget for the NLI leg of `contradictions_report`; on exhaustion the scan stops early and names unjudged pairs in `nli_failures` |
| `CONTRADICTION_NEIGHBORS` | `8` | Vector neighbours fetched per memory per searched scope table (1-20) |
| `CONTRADICTION_NLI_WORKERS` | `4` | Bounded concurrency of the NLI calls (1-16) |

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

`LLM_PROVIDER=auto` picks the first configured key (Groq, OpenAI, Anthropic),
then `LLM_BASE_URL`, then `OLLAMA_BASE_URL`. `LLM_PROVIDER=none` disables
contradiction detection. Without a usable provider `/nli` returns placeholder
`0.0` scores together with an `error`; the MCP client raises instead of
treating those zeros as judgments. `contradictions_report` short-circuits when
health reports no configured provider and returns `nli_configured=false` plus
a warning; write tools keep the fact and warn
(`contradiction check skipped: NLI unavailable: ...`) in `warnings` and
`contradictions.skipped`.

Embeddings always run locally — no key needed. The default image is CUDA
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
