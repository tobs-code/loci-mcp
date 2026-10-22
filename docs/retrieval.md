# Retrieval & Scoring

How a query becomes a ranked list, and what every score field means.

## The two retrievers

`recall` combines two retrievers:

1. **Vector search** (pgvector, cosine distance) finds memories that mean the
   same thing as the query.
2. **Full-text search** (PostgreSQL FTS) finds memories that contain the
   query's words.

Both are needed. The embedding model handles paraphrases ("how do we ship
releases" → "the release train ships every second Tuesday"), but it is bad at
identifiers and foreign words — a bare keyword like `ZEBRAFISH_TOKEN` only
scores 0.30-0.61, while a natural-language question that matches scores
0.62-0.95. The full-text leg catches exactly that case.

The full-text query is OR-based (`_fts_query`) and weighted with
`FTS_WEIGHT = 0.6`, so a single common word cannot outrank the best semantic
hit. The normalized text score is multiplied by **query coverage** (matched
query lexemes / all query lexemes after stemming and stopword removal), so a
document that matched one of three words cannot look like a perfect full-text
match. The retriever uses the English dictionary (`FTS_LANGUAGE`); index and
query side always match, a changed dictionary is migrated at startup.

## Read scope defaults

- `recall` and `get_entity` use the global scope when `project` is omitted;
  with a project they include that project's **and** global memories by
  default.
- `list_memories` and `list_entities` list all scopes when `project` is
  omitted, and project + global by default when it is set.
- `init_context` uses the global scope when `project` is omitted, project +
  global when set, and all scopes with `project="*"`.
- Pass `include_global=false` for project-only reads.

`recall` searches the project table (if `project` is set) **and** the global
`memories` table, so global facts remain visible inside a project. Project
hits receive a small scope tie-breaker. When the same content exists in both
scopes, the project copy wins.

## Score fields

| Field | Meaning |
|-------|---------|
| `_similarity` | `1 - cosine distance` (0..1), the same scale as `min_similarity`. |
| `_distance` | Raw cosine distance to the query. |
| `_text_score` | Normalized rank of the full-text hit (1.0 = best). |
| `_relevance_score` | Normalized 0..1 combination of both signals; `null` for graph-only hits. |
| `_rrf_score` / `_rrf_norm` | Raw RRF fusion value and its normalized form. |
| `_decay_score` / `_decay_norm` | Raw decay value (uncapped) and the normalized 0..1 score. |
| `_final_score` | The ranking score in 0..1: relevance blended with decay plus the graph boost. |
| `_graph_boost` / `_graph_source` / `_graph_weight` | Describe a graph hit: the boost, the connected entity and the edge weight. |

`_relevance_score` is the number to compare hits by. It combines the two
retrievers as independent chances to match, so a memory that *both* retrievers
found scores higher than either signal alone. Plain RRF cannot do that: it
ranks by position, so the third hit is almost as good as the first (0.97 vs
1.0) even when its similarity is 0.42 against 0.71.

## The similarity filter

`min_similarity` defaults to **0.35** and filters the **vector** retriever
only. A full-text hit is kept even when the embedding model scores it badly —
that is the case the full-text leg exists for.

`recall` reports `candidates` and `dropped_by_similarity` (the number of vector
hits removed) and warns when the filter removed everything, so an empty result
is never silent.

With `min_similarity=null` the filter is disabled and near-zero hits can
appear as "relevant". When any returned hit has `_similarity < 0.35`, the
response warns about it (`the min_similarity filter is disabled: N of M
result(s) ...`), so agents notice the weak hits instead of reading them as
confident matches.

Measured with the current embedding model:

| Query | Similarity |
|-------|------------|
| natural-language question that matches | 0.62-0.95 |
| bare keyword occurring verbatim in the memory | 0.30-0.61 |
| unrelated fact | below ~0.30 |

## Cognitive decay

Memories age with a half-life of 90 days.

- `importance` (0.0-1.0) and `access_count` (max. 20) slow the decay down.
- `include_decay=true` in `recall` blends recency/frequency into the score.
- Decay contributes at most 30% of the score (`DECAY_WEIGHT = 0.3`); the
  remaining 70% is relevance, so recency cannot drown out semantics.
- `min_decay_score` (0..1) filters on the normalized decay (`_decay_norm`):
  `importance x recency` plus at most a 10% access-frequency bonus. A
  brand-new memory with `importance=1.0` scores ~0.9, a 90-day-old one ~0.5,
  so `min_decay_score=0.9` keeps fresh, important facts. The filter runs
  before the access counters of the current query are counted.
- `_decay_score` keeps the raw value (uncapped, can exceed 1.0).

## Graph boost

`graph_boost=true` traverses one hop in the scope graph, loads up to 5
memories per connected entity and adds `_graph_boost = 0.3 * edge weight` on
top of the blended relevance/decay score. Edge weights are limited to 0.0-1.0,
so a single edge cannot dominate the ranking. Graph hits have no similarity
and count as relevance 0 — they are lifted by the boost only.

The expansion point is `graph_entity`; `entity` is the fallback. With
`graph_entity` the query search stays unfiltered ("query hits + graph
neighbourhood"), with only `entity` the search is also restricted to that
entity — which drops unrelated query hits.

## Embeddings are prompt-aware

Queries are sent with `mode="query"`, stored content with `mode="document"`
(Jina retrieval prefixes). Mixing the two makes scores incomparable, so
re-embed after changing the model or prompts:

```bash
docker compose exec loci-mcp python -m mcp_server.reembed
```

## Contradiction detection

Two paths, both using **vector pre-filtering** to avoid massive API calls.

### Path A: `remember` with `check_contradictions=true`

1. The new fact's embedding is computed (already available for the store).
2. **Vector pre-filter**: pgvector `<=>` finds the 15 nearest neighbours.
3. **NLI per candidate**: `/nli` (the configured LLM) checks each candidate for
   a real contradiction.
4. At a score >= 0.7 the old memory is automatically invalidated via
   `superseded_by = new_id`.

Every memory type is a candidate: a decision that contradicts a pattern is
just as harmful as a fact that contradicts a fact, and restricting the
comparison to `fact` hid most real conflicts.

### Path B: `contradictions_report`

1. All **active** memories of the entity are loaded (max 200; already
   superseded pairs are done and not re-checked).
2. **Vector self-join**: for each memory, the 20 nearest neighbours are fetched
   via `ORDER BY embedding <=>`.
3. Only the deduplicated candidate pairs are checked via `/nli` — one call per
   memory, not one call per pair.
4. Pairs with a confidence > 0.7 count as a contradiction.
5. `resolve="newest"` (opt-in) keeps the most recently updated memory of each
   pair active and supersedes the older one; the pair reports `resolution` with
   the `superseded` and `kept` ids.

The quality of the detection depends on the embedding model
(`jina-embeddings-v5`), which provides good similarity values for semantically
similar texts. Distracting prefixes such as `Batch-X: ` in the content destroy
the similarity and should be avoided. If necessary, increase
`neighbors_per_fact` in `contradictions_report`.

Without a usable LLM configuration `/nli` returns `0.0` for every pair; the
contradiction tools report a warning and `nli_configured=false` instead of
presenting that as a clean contradiction scan.

## Superseding

- `supersede` replaces a memory by id.
- `supersede_content` replaces the memory with the same wording.
- `supersede_last` replaces the newest memory of an entity, but only when the
  new fact is about the same topic (cosine similarity >= 0.5). Otherwise the
  old memory stays active, the new fact is stored additionally and the
  response carries a warning — replacing an unrelated fact would destroy it
  silently.

Metadata of replaced memories is not lost: keys the new fact does not set are
merged into the new memory (explicitly passed metadata always wins). The
response lists the carried keys in a warning (`metadata carried over from
the replaced memories: ...`).

`remember` also warns when the new content matches a **superseded** memory: the
fact is still stored as active (an intentional revert is legitimate), but the
response names the old id and its replacement, so the fork is visible instead
of silently reintroducing an invalidated fact.

## Dedupe modes

| Mode | Detects |
|------|---------|
| `exact` (default) | Identical content hash |
| `semantic` | Additionally vector similarity above `fuzzy_threshold` (default 0.9) |
| `off` | Nothing — every call stores a new memory |

A duplicate answer names the mode that matched (`duplicate_mode`) and the
measured similarity.

## The bootstrap is not a search

`init_context` returns the **most recently changed** memories of the scope,
never a decay-ranked mix: a freshly stored fact always shows up, even if it
was never read before. `detail="full"` returns every column except the
internal ones and the search score annotations — the bootstrap is sorted by
the last update, not by a score.
