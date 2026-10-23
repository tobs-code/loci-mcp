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
  omitted or `project="*"`, and project + global by default when a project
  name is set.
- `project="global"` means the **shared scope** in every read tool - it is not
  "all scopes". In the two browse tools it is stricter than omitting the
  parameter; in `recall` it is the same (only the global scope is searched).
- `init_context` uses the global scope when `project` is omitted, project +
  global when set, and all scopes with `project="*"`.
- `recall` accepts `project="*"` too: it then searches every scope at once.
  `get_entity` rejects `"*"`: its answer describes exactly one entity.
- Cross-scope dedup only drops an **active** copy in another scope. Superseded
  and soft-forgotten rows are history and stay visible; two active rows in the
  same scope (allowed by `dedupe="off"`) both stay as well.
- Pass `include_global=false` for project-only reads.

A project that does not exist is never silently ignored: the read tools
(`recall`, `list_memories`, `list_entities`, `init_context`) fall back to the
global scope and say so in `warnings`, so a typo in the project name cannot
look like a successful search.

`recall` searches the project table (if `project` is set) **and** the global
`memories` table, so global facts remain visible inside a project. Project
hits receive a small scope tie-breaker (`_scope_boost`, up to 0.05,
compressed with the relevance: `0.05 × (1 − relevance)`, so a strong hit stays
below 1.0 and comparable). When the same content exists in
both scopes, the project copy wins.

## Score fields

| Field | Meaning |
|-------|---------|
| `_similarity` | `1 - cosine distance` (0..1), the same scale as `min_similarity`. |
| `_distance` | Raw cosine distance to the query. |
| `_text_score` | Normalized rank of the full-text hit (1.0 = best). |
| `_relevance_score` | Normalized 0..1 combination of both signals; `null` for graph-only hits. |
| `_scope_boost` | The compressed boost a project row gets on top of `_relevance_score`: `0.05 × (1 − relevance)`, at most 0.05 for a weak hit. |
| `_rrf_score` / `_rrf_norm` | Raw RRF fusion value and its normalized form. |
| `_decay_score` / `_decay_norm` | Raw decay value (uncapped) and the normalized 0..1 score. |
| `_final_score` | The ranking score in 0..1: relevance blended with decay plus the graph boost. Without `include_decay` the base is the plain relevance, so `graph_boost` cannot lower a query hit. Only present with `include_decay=true` or `graph_boost=true`. |
| `_graph_boost` / `_graph_source` / `_graph_weight` | Describe a graph hit: the boost, the connected entity and the edge weight. |

`_relevance_score` is the number to compare hits by. It combines the two
retrievers as independent chances to match, so a memory that *both* retrievers
found scores higher than either signal alone. Plain RRF cannot do that: it
ranks by position, so the third hit is almost as good as the first (0.97 vs
1.0) even when its similarity is 0.42 against 0.71.

Without `include_decay` and `graph_boost` there is no `_final_score`: the
ranking is then `_relevance_score`, and the result is ordered by it.

## The similarity filter

`min_similarity` defaults to **0.40** and filters the **vector** retriever
only. A full-text hit is kept even when the embedding model scores it badly —
that is the case the full-text leg exists for.

`recall` reports `candidates` (the hits that made it into the ranking),
`dropped_by_similarity` (the number of vector hits the filter removed) and
`dropped_by_coverage` (the number of full-text hits the coverage filter
removed). `dropped_by_decay` and `dropped_by_superseded` are reported
whenever the `min_decay_score` / `only_superseded` filter removed hits — also
when other hits survived, so a non-empty result is never quietly short. When
the result is empty, the warning names **every** filter that removed
something, so the caller does not have to guess which one was responsible.
In that warning the strongest rejected vector hit is named explicitly, and the
payload carries `best_dropped_by_similarity` (`id`, `similarity`, `scope`,
`content_preview`): a fact that missed the cutoff by a hair is shown instead
of hiding behind the `dropped_by_similarity` counter.

## The full-text coverage floor

The full-text leg is OR-based: a document that contains one common word of a
six-word query is a hit. The coverage factor (matched query lexemes / all
query lexemes) scales its score down, but a hit that only the full-text leg
found has no similarity to vouch for it. Such a hit is therefore dropped
when it covers less than **0.3** of the query (`FTS_MIN_COVERAGE`), i.e. for
queries of four or more lexemes a single-word match no longer survives.

This keeps a genuine keyword hit ("ZEBRAFISH_TOKEN" in a longer query) and
removes the noise hit: the query *"requests per minute limit"* returns the
memory *"The pricing tier ... is $500 per month"* only through the word "per"
(actually not a stopword for the english dictionary, coverage 1/4 = 0.25).
Since `FTS_MIN_COVERAGE = 0.3` it is dropped.
Actually not a real keyword hit: hits that the vector retriever found are never
dropped by this filter.

With `min_similarity=null` the filter is disabled and near-zero hits can
appear as "relevant". When any returned hit has `_similarity < 0.40`, the
response warns about it (`the min_similarity filter is disabled: N of M
result(s) ...`), so agents notice the weak hits instead of reading them as
confident matches.

## The searched scopes

`recall` reports `scopes_searched`: the scopes the query actually covered.
Without a `project` only the shared scope is searched, so an agent that finds
nothing there is told which other scopes exist and that `project="*"` covers
them. `get_entity` reports the same plus `memories_by_scope`;
`list_memories` adds `scopes_searched`, `total_by_scope` (before cross-scope
dedup) and `listed_by_scope` (after it), so a mixed global + project browse is
never mistaken for one flat scope.

`recall(detail="slim")` trims the rows to the fields a session usually needs
(id, content, entity, type, project, importance, timestamps and scores) while
the default `detail="full"` keeps metadata and history bookkeeping. An explicit
`fields` list still wins over `detail`.

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
  `importance x recency` times an access factor of 0.9..1.0. A brand-new
  memory with `importance=1.0` scores 0.9 (`importance=0.5` scores 0.45), a
  90-day-old one roughly half of that, so `min_decay_score=0.9` keeps only
  fresh, top-importance facts. The filter runs before the access counters of
  the current query are counted.
- `_decay_score` keeps the raw value (uncapped, can exceed 1.0).

## Graph boost

`graph_boost=true` traverses one hop in the scope graph, loads up to 5
memories per connected entity and adds `_graph_boost = 0.3 * edge weight` on
top of the relevance (blended with decay only when `include_decay=true`).
Edge weights are limited to 0.0-1.0, so a single edge cannot dominate the
ranking. Graph hits have no similarity and count as relevance 0 — they are
lifted by the boost only. Query hits keep their slots: graph-only additions
fill the remaining page space and never displace a real query match.

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
   a real contradiction. The comparison carries the entity labels of both
   facts, so the model knows whether they describe the same subject.
4. At a score >= 0.7 the old memory is automatically invalidated via
   `superseded_by = new_id`.

Every memory type is a candidate: a decision that contradicts a pattern is
just as harmful as a fact that contradicts a fact, and restricting the
comparison to `fact` hid most real conflicts.

Entity tagging is not required. With an entity, that entity's memories get the
15 candidate seats first and the leftovers go to cross-entity neighbours, so a
differently tagged memory cannot hide a conflict. Without an entity all
memories of the scope compete; the response sets `contradictions.cross_entity`
and warns about the broader comparison, so it is never silent.

### Path B: `contradictions_report`

1. All **active** memories of the given entity are loaded (max 200; already
   superseded pairs are done and not re-checked). Without an entity the scan
   is scope-wide: all active memories compete, so pairs across different
   entities are reported too.
2. **Vector self-join**: for each memory, the 8 nearest neighbours per
   searched scope table (project + global) are fetched via
   `ORDER BY embedding <=>` (`CONTRADICTION_NEIGHBORS`, 1-20).
3. Only the deduplicated candidate pairs are checked via `/nli` — one call per
   memory, not one call per pair. The calls run in a bounded pool (4 workers,
   `CONTRADICTION_NLI_WORKERS`) and are reported in candidate order.
4. The NLI leg has a wall-clock budget (`budget_seconds`, default 25s or
   `CONTRADICTION_SCAN_BUDGET_SECONDS`). When it runs out the scan stops early,
   returns the pairs judged so far, sets `budget_exhausted: true` and lists
   every unchecked pair in `nli_failures` with
   `reason: "skipped: NLI time budget exhausted"`. The transport call no
   longer dies with a timeout, so a partial scan is visible and machine
   readable. Narrow the scan (`entity`, lower `limit`) or raise the budget to
   check more.
5. Pairs with a confidence > 0.7 count as a contradiction. Each pair names
   the entity of both facts and a `cross_entity` flag.
6. `resolve="newest"` (opt-in) keeps the most recently updated memory of
   each pair of the **same entity** active and supersedes the older one; the
   pair reports `resolution` with the `superseded` and `kept` ids. Both
   untagged facts count as the same subject. **Cross-entity pairs are never
   auto-resolved**: recency decides nothing when the two facts belong to
   different subjects — the newer fact is not necessarily the correct one
   ("staging API on port 9443" vs "inference service on port 8001" was
   reported as 0.95 by a field-by-field prompt, and an unconditional
   resolve destroyed the innocent Inference fact). Cross-entity pairs stay
   active, carry `cross_entity: true` and the scan warns
   `resolve skipped for N cross-entity pair(s) ...`.

The quality of the detection depends on the embedding model
(`jina-embeddings-v5`), which provides good similarity values for semantically
similar texts. Distracting prefixes such as `Batch-X: ` in the content destroy
the similarity and should be avoided. If necessary, raise
`CONTRADICTION_NEIGHBORS` (or pass a larger `budget_seconds`, so more of the
candidate groups are actually judged).

The NLI prompt first asks the model to decide whether both facts describe
the **same subject** at all; a differing attribute (environment, host, port,
version, date/time, state) or an explicit negation counts as a contradiction
**only then**. Two facts about different subjects stay neutral even when
their attribute values differ — a staging API on one port never contradicts
a database service on another port. Merely added or missing details are not
contradictions either. The `/nli` contract carries optional
`query_entity`/`document_entities` labels, and both contradiction paths send
the entity each fact is stored under, so the model knows the subjects and
gets an explicit hint when they differ. Vague prompts produced false
negatives - a vague prompt judged "deploys to staging" vs "deploys to
production" as neutral, because both could be true at the same time
(Oct 2026); the opposite extreme - attributing differing attributes blindly
to the same subject - produced cross-entity false positives (Oct 2026).

Without a usable LLM configuration `/nli` returns placeholder `0.0` scores
together with an `error`; the MCP client raises instead of treating those
zeros as judgments. `contradictions_report` short-circuits when health reports
no configured provider and returns `nli_configured=false` plus a warning; the
write tools keep the fact and warn
(`contradiction check skipped: NLI unavailable: ...`) in `warnings` and in
`contradictions.skipped`. The same error contract applies to an NLI
**outage**: `/nli` sets an `error` field instead of silently returning zeros,
and the client raises. `contradictions_report` marks the scan as incomplete
(`warnings: ["scan incomplete: ... - the result is not a clean contradiction
scan"]`) when the loop aborts mid-way.

## Superseding

- `supersede` replaces a memory by id.
- `supersede_content` replaces the memory with the same wording.
- `supersede_last` replaces the memory of an entity that matches the new fact
  best (most similar, not the newest), but only when the new fact is about the
  same topic (cosine similarity >= 0.5). Otherwise the old memory stays
  active, the new fact is stored additionally and the response carries a
  warning — replacing an unrelated fact would destroy it silently.

Metadata of replaced memories is not lost by default: keys the new fact does
not set are merged into the new memory (explicitly passed metadata always
wins). The response lists the carried keys in a warning (`metadata carried
over from the replaced memories: ...`). Pass `metadata_mode="replace"` to drop
the old keys instead.

`remember` also warns when the new content matches a **superseded** memory: the
fact is still stored as active (an intentional revert is legitimate), but the
response names the old id and its replacement, so the fork is visible instead
of silently reintroducing an invalidated fact.

A fact that is still very similar to an active memory is a **near-duplicate**
(cosine similarity >= 0.75). `on_near_duplicate` decides what happens: `warn`
(default) stores it and names the similar id and the measured similarity;
`supersede` replaces the old memory; `reject` answers `duplicate` and stores
nothing. `auto_supersede_near_duplicate=true` is the backwards-compatible
alias for `supersede`. `dedupe="off"` skips the check only for `warn`.

A write that invalidates a fact of the shared `memories` table (the
contradiction check or an explicit `supersede` of a global id) hides it from
every scope. That is reported as `cross_scope_superseded` plus a
`cross-scope supersede` warning.

## Dedupe modes

| Mode | Detects |
|------|---------|
| `exact` (default) | Identical content hash |
| `semantic` | Additionally vector similarity above `fuzzy_threshold` (default 0.85) |
| `off` | Nothing — every call stores a new memory |

A duplicate answer names the mode that matched (`duplicate_mode`) and the
measured similarity.

## The bootstrap is not a search

`init_context` returns the **most recently changed** memories of the scope,
never a decay-ranked mix: a freshly stored fact always shows up, even if it
was never read before. `detail="full"` returns every column except the
internal ones and the search score annotations — the bootstrap is sorted by
the last update, not by a score.
