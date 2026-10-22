import json
import logging
from datetime import datetime, timezone
from typing import Any

from mcp_server.config import FTS_LANGUAGE, FTS_WEIGHT, PROJECT_SCOPE_BOOST, RRF_K
from mcp_server.db import get_conn
from mcp_server.scopes import _ag_str, _project_names, _retrieval_tables, _scope_label
from mcp_server.scoring import (
    _blend_scores,
    _clamp01,
    _decay_norm,
    _decay_score,
    _fts_query,
    _now,
    _relevance_score,
)

logger = logging.getLogger(__name__)

def _extra_conditions(
    updated_after: str | None = None,
    metadata_filter: dict | None = None,
) -> tuple[list[str], list[Any]]:
    """Build the optional filter conditions shared by all retrieval paths."""
    conditions: list[str] = []
    params: list[Any] = []
    if updated_after:
        stamp = _parse_timestamp(updated_after)
        if stamp is None:
            raise ValueError(
                f"invalid updated_after {updated_after!r}: use an ISO date "
                f"like 2026-10-01 or 2026-10-01T12:00:00Z"
            )
        conditions.append("updated_at >= %s")
        params.append(stamp)
    if metadata_filter:
        conditions.append("metadata @> %s::jsonb")
        params.append(json.dumps(metadata_filter))
    return conditions, params


def _parse_timestamp(value: str):
    """Parse an ISO-8601 date/datetime into a timezone-aware datetime."""
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        stamp = datetime.fromisoformat(text)
    except ValueError:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp


def _search_memories_rrf(
    cur,
    query_embedding: list[float] | None,
    query_text: str | None,
    mem_type: str | None,
    entity: str | None,
    limit: int,
    project: str | None,
    include_decay: bool = False,
    include_embedding: bool = False,
    include_superseded: bool = False,
    min_similarity: float | None = None,
    k: int = RRF_K,
    tables: list[str] | None = None,
    updated_after: str | None = None,
    metadata_filter: dict | None = None,
    stats: dict[str, Any] | None = None,
) -> list[dict]:
    """
    Reciprocal Rank Fusion over vector + FTS results.

    Retrieval runs against the project table (if any) and the global table;
    project rows win when both contain the same content. Scores are normalized
    so that decay can reorder results without drowning out relevance.

    `min_similarity` only applies to the vector retriever. A full-text hit is
    kept even when its embedding is a poor match: that is exactly the case
    the full-text leg exists for (an identifier, a file name, a foreign word
    that the embedding model handles badly). The number of candidates that
    the filter removed is reported in `stats`.
    """
    tables = tables if tables is not None else _retrieval_tables(cur, project)
    if not tables:
        return []

    max_distance = None if min_similarity is None else 1.0 - min_similarity
    fts_query = _fts_query(query_text)
    # Query lexemes after stemming/stopword removal. They are used to measure
    # how much of the query a full-text hit actually covers: an OR query makes
    # a single common word look like a perfect match otherwise.
    fts_lexemes: list[str] = []
    if fts_query and query_text:
        cur.execute(
            f"SELECT tsvector_to_array(to_tsvector('{FTS_LANGUAGE}', %s))",
            (query_text,),
        )
        row = cur.fetchone()
        fts_lexemes = list(row[0] or []) if row else []
    extra_conditions, extra_params = _extra_conditions(updated_after,
                                                        metadata_filter)

    # (score, id, table) tuples, merged across tables before ranking
    vec_candidates: list[tuple[float, str, str]] = []
    # (rank, matched_lexemes, id, table)
    text_candidates: list[tuple[float, int, str, str]] = []
    dropped_by_similarity = 0

    for table in tables:
        conditions = ["TRUE"]
        params: list[Any] = []
        if include_superseded:
            # Superseded history is retrievable, intentionally deleted rows are not.
            conditions.append("deleted_at IS NULL")
        else:
            conditions.append("superseded_by IS NULL AND deleted_at IS NULL")
        if mem_type:
            conditions.append("mem_type = %s")
            params.append(mem_type)
        if entity:
            conditions.append("entity = %s")
            params.append(entity)
        conditions.extend(extra_conditions)
        params.extend(extra_params)
        where = " AND ".join(conditions)

        # 1. Vector Search (Top 50 per table)
        if query_embedding is not None:
            sql = (
                f"SELECT id, embedding <=> %s::vector AS distance FROM {table} "
                f"WHERE {where} AND embedding IS NOT NULL"
            )
            vec_params: list[Any] = [query_embedding] + params
            # The distance filter is applied below, not in SQL: only then the
            # number of rejected candidates can be reported to the caller.
            sql += " ORDER BY distance ASC LIMIT 50"
            cur.execute(sql, vec_params)
            for row_id, dist in cur.fetchall():
                dist = float(dist)
                if max_distance is not None and dist > max_distance:
                    dropped_by_similarity += 1
                    continue
                vec_candidates.append((dist, row_id, table))

        # 2. FTS Search (Top 50 per table), OR-joined terms
        if fts_query:
            sql = (
                f"SELECT id, ts_rank(to_tsvector('{FTS_LANGUAGE}', content), "
                f"websearch_to_tsquery('{FTS_LANGUAGE}', %s)) AS rank, "
                f"(SELECT COUNT(*) FROM unnest(%s::text[]) AS lex "
                f"WHERE to_tsvector('{FTS_LANGUAGE}', content) @@ "
                f"plainto_tsquery('{FTS_LANGUAGE}', lex)) AS matched "
                f"FROM {table} "
                f"WHERE to_tsvector('{FTS_LANGUAGE}', content) @@ "
                f"websearch_to_tsquery('{FTS_LANGUAGE}', %s)"
            )
            fts_params: list[Any] = [fts_query, fts_lexemes, fts_query]
            # No similarity filter here: a full-text hit stays a hit even when
            # the embedding model scores it poorly.
            sql += f" AND {where} ORDER BY rank DESC LIMIT 50"
            cur.execute(sql, fts_params + params)
            text_candidates.extend(
                (float(rank), int(matched or 0), row_id, table)
                for row_id, rank, matched in cur.fetchall()
            )

    vec_candidates.sort(key=lambda c: c[0])
    text_candidates.sort(key=lambda c: -c[0])

    # 3. RRF Fusion
    vector_results: dict[str, float] = {}
    text_results: dict[str, float] = {}
    similarities: dict[str, float] = {}
    text_scores: dict[str, float] = {}
    table_of: dict[str, str] = {}

    for rank, (distance, row_id, table) in enumerate(vec_candidates[:50], 1):
        vector_results[row_id] = 1.0 / (k + rank)
        similarities[row_id] = 1.0 - distance
        table_of.setdefault(row_id, table)
    for rank, (_, _, row_id, table) in enumerate(text_candidates[:50], 1):
        text_results[row_id] = FTS_WEIGHT / (k + rank)
        table_of.setdefault(row_id, table)

    max_text_rank = max((r for r, _, _, _ in text_candidates[:50]), default=0.0)
    lexeme_count = max(1, len(fts_lexemes))
    for rank, (ts_rank, matched, row_id, _table) in enumerate(text_candidates[:50], 1):
        # Normalized full-text evidence: the best text hit is 1.0, weaker
        # hits fall off with their text rank. Multiply by query coverage so a
        # hit that matched only one word of a multi-word query cannot look
        # like a perfect match.
        rank_norm = (ts_rank / max_text_rank) if max_text_rank else 0.0
        coverage = min(1.0, matched / lexeme_count) if fts_lexemes else 1.0
        text_scores[row_id] = rank_norm * coverage

    fused_scores = {
        row_id: vector_results.get(row_id, 0.0) + text_results.get(row_id, 0.0)
        for row_id in set(vector_results) | set(text_results)
    }
    if not fused_scores:
        if stats is not None:
            stats.update({"candidates": 0,
                          "dropped_by_similarity": dropped_by_similarity,
                          "fulltext_only": 0})
        return []

    max_rrf = max(fused_scores.values()) or 1.0
    # Do not cut before cross-table dedup: many duplicate contents could
    # otherwise hide unique rows that would fit into the limit.
    top_ids = sorted(fused_scores, key=lambda x: -fused_scores[x])

    # 4. Load details per table (project first), dropping cross-table duplicates
    rows: list[dict] = []
    seen_hashes: set[str] = set()
    project_names = _project_names(cur)
    for table in tables:
        ids = [row_id for row_id in top_ids if table_of.get(row_id) == table]
        if not ids:
            continue
        cur.execute(f"SELECT * FROM {table} WHERE id = ANY(%s)", (ids,))
        colnames = [desc[0] for desc in cur.description]
        by_id = {r[0]: dict(zip(colnames, r)) for r in cur.fetchall()}
        for row_id in ids:
            d = by_id.get(row_id)
            if d is None:
                continue
            if table == "memories":
                d["_project"] = "global"
            else:
                d["_project"] = project_names.get(table[4:], table[4:])
            c_hash = d.get("content_hash") or ""
            if c_hash and c_hash in seen_hashes:
                continue
            if c_hash:
                seen_hashes.add(c_hash)
            d["_rrf_score"] = fused_scores.get(row_id, 0.0)
            d["_rrf_norm"] = d["_rrf_score"] / max_rrf
            d["_similarity"] = round(similarities[row_id], 4) if row_id in similarities else None
            d["_distance"] = round(1.0 - d["_similarity"], 4) if d["_similarity"] is not None else None
            if row_id in text_scores:
                d["_text_score"] = round(text_scores[row_id], 4)
            # Relevance combines both signals, so a memory found by the
            # full-text retriever alone is still ranked above a weak vector
            # match instead of being flattened into the same RRF band.
            relevance = _relevance_score(similarities.get(row_id, 0.0),
                                         text_scores.get(row_id, 0.0))
            if project and table != "memories":
                relevance = min(1.0, relevance + PROJECT_SCOPE_BOOST)
                d["_scope_boost"] = PROJECT_SCOPE_BOOST
            d["_relevance_score"] = round(relevance, 4)
            if not include_embedding:
                d.pop("embedding", None)
            if include_decay:
                d["_decay_score"] = _decay_score(
                    d.get("created_at"), d.get("updated_at"),
                    d.get("access_count", 0), d.get("importance", 1.0),
                )
                d["_decay_norm"] = _decay_norm(
                    d.get("created_at"), d.get("updated_at"),
                    d.get("access_count", 0), d.get("importance", 1.0),
                )
            rows.append(d)

    if stats is not None:
        stats.update({
            "candidates": len(fused_scores),
            "dropped_by_similarity": dropped_by_similarity,
            "fulltext_only": sum(1 for row_id in fused_scores
                                 if row_id not in similarities),
        })

    # 5. Re-rank
    if include_decay:
        for r in rows:
            r["_final_score"] = _clamp01(
                _blend_scores(r["_relevance_score"], r["_decay_norm"])
            )
        rows.sort(key=lambda x: -x.get("_final_score", 0))
    else:
        rows.sort(key=lambda x: (-x.get("_relevance_score", 0.0),
                                 -x.get("_rrf_score", 0.0)))

    return rows[:limit]



def _search_memories(
    cur,
    query_embedding: list[float] | None,
    query_text: str | None,
    mem_type: str | None,
    entity: str | None,
    limit: int,
    project: str | None,
    include_decay: bool = False,
    include_embedding: bool = False,
    include_superseded: bool = False,
    min_similarity: float | None = None,
    hybrid: bool = True,  # Default: RRF aktivieren
    tables: list[str] | None = None,
    updated_after: str | None = None,
    metadata_filter: dict | None = None,
    recent_first: bool = False,
    stats: dict[str, Any] | None = None,
) -> list[dict]:
    """
    Wrapper for search functions. Uses RRF when hybrid=True, otherwise legacy search.

    `tables` overrides the table set (default: project table + global).
    `recent_first` keeps the "newest first" order of a browse instead of
    re-ranking by decay score - the session bootstrap needs recency, not the
    most frequently accessed memories.
    """
    if hybrid and (query_embedding is not None or query_text):
        return _search_memories_rrf(
            cur, query_embedding, query_text, mem_type, entity, limit, project,
            include_decay, include_embedding, include_superseded, min_similarity,
            tables=tables, updated_after=updated_after,
            metadata_filter=metadata_filter, stats=stats,
        )

    # Legacy fallback for individual retrievers
    tables = tables if tables is not None else _retrieval_tables(cur, project)
    if not tables:
        return []

    max_distance = None if min_similarity is None else 1.0 - min_similarity
    has_vec = query_embedding is not None
    fts_query = _fts_query(query_text)
    extra_conditions, extra_params = _extra_conditions(updated_after,
                                                        metadata_filter)

    rows: list[dict] = []
    seen_hashes: set[str] = set()
    dropped_by_similarity = 0
    project_names = _project_names(cur)
    # Fetch more than the caller asked for: the similarity filter is applied
    # in Python, so the limit must not cut the candidates before filtering.
    fetch_limit = max(limit, 50)
    for table in tables:
        conditions: list[str] = ["TRUE"]
        params: list[Any] = []
        params_vec: list[Any] = []

        if has_vec:
            col = "*, embedding <=> %s::vector AS distance"
            conditions.append("embedding IS NOT NULL")
            order = "ORDER BY distance ASC"
            params_vec.append(query_embedding)
        elif fts_query:
            col = (f"*, ts_rank(to_tsvector('{FTS_LANGUAGE}', content), "
                   f"websearch_to_tsquery('{FTS_LANGUAGE}', %s)) AS fts_rank")
            params_vec.append(fts_query)
            order = "ORDER BY fts_rank DESC"
        else:
            col = "*"
            order = "ORDER BY updated_at DESC"

        if fts_query:
            conditions.append(
                f"to_tsvector('{FTS_LANGUAGE}', content) @@ "
                f"websearch_to_tsquery('{FTS_LANGUAGE}', %s)"
            )
            params.append(fts_query)
        if mem_type:
            conditions.append("mem_type = %s")
            params.append(mem_type)
        if entity:
            conditions.append("entity = %s")
            params.append(entity)
        conditions.extend(extra_conditions)
        params.extend(extra_params)

        if include_superseded:
            conditions.append("deleted_at IS NULL")
        else:
            conditions.append("superseded_by IS NULL AND deleted_at IS NULL")

        where = " AND ".join(conditions)
        cur.execute(
            f"SELECT {col}, created_at, updated_at, access_count, importance "
            f"FROM {table} WHERE {where} {order} LIMIT %s",
            params_vec + params + [fetch_limit],
        )

        colnames = [desc[0] for desc in cur.description] if cur.description else []
        for row in cur.fetchall():
            d = dict(zip(colnames, row))
            if table == "memories":
                d["_project"] = "global"
            else:
                d["_project"] = project_names.get(table[4:], table[4:])
            if "distance" in d and d["distance"] is not None:
                d["_distance"] = float(d.pop("distance"))
                if max_distance is not None and d["_distance"] > max_distance:
                    dropped_by_similarity += 1
                    continue
                d["_similarity"] = round(1.0 - d["_distance"], 4)
            else:
                d.pop("distance", None)
            if "fts_rank" in d:
                d["_fts_rank"] = float(d.pop("fts_rank") or 0.0)
            c_hash = d.get("content_hash") or ""
            if c_hash and c_hash in seen_hashes:
                continue
            if c_hash:
                seen_hashes.add(c_hash)
            if not include_embedding:
                d.pop("embedding", None)
            if include_decay:
                d["_decay_score"] = _decay_score(
                    d.get("created_at"), d.get("updated_at"),
                    d.get("access_count", 0), d.get("importance", 1.0),
                )
                d["_decay_norm"] = _decay_norm(
                    d.get("created_at"), d.get("updated_at"),
                    d.get("access_count", 0), d.get("importance", 1.0),
                )
            rows.append(d)

    if has_vec:
        rows.sort(key=lambda x: x.get("_distance", 1.0))
    elif fts_query:
        max_rank = max((r.get("_fts_rank", 0.0) for r in rows), default=0.0)
        for row in rows:
            rank = row.pop("_fts_rank", 0.0)
            row["_text_score"] = round(rank / max_rank, 4) if max_rank else 0.0
            row["_relevance_score"] = row["_text_score"]
        rows.sort(key=lambda x: -x.get("_relevance_score", 0.0))
    else:
        rows.sort(key=lambda x: x.get("updated_at") or _now(), reverse=True)

    if has_vec:
        for row in rows:
            row["_relevance_score"] = row.get("_similarity", 0.0)
    if stats is not None:
        stats.update({
            "candidates": len(rows),
            "dropped_by_similarity": dropped_by_similarity,
            "fulltext_only": 0,
        })
    if include_decay:
        for r in rows:
            relevance = r.get("_relevance_score")
            if relevance is not None:
                r["_final_score"] = _clamp01(
                    _blend_scores(relevance, r["_decay_norm"])
                )
            else:
                # Nothing relevant to blend with: recency/frequency only.
                r["_final_score"] = r["_decay_norm"]
        if not recent_first:
            rows.sort(key=lambda x: -x.get("_final_score", 0))

    return rows[:limit]



def _search_memories_graph_boost(
    existing_results: list[dict],
    entity: str,
    limit: int,
    project: str | None,
    include_decay: bool = False,
    graph_boost_weight: float = 0.3,  # 30% boost for graph candidates
    include_global: bool = True,
    updated_after: str | None = None,
    metadata_filter: dict | None = None,
) -> list[dict]:
    """
    Augments search results with memories from connected entities via the graph.
    Merges graph candidates with RRF scores.
    """
    tables = None
    name_json = json.dumps(entity)
    graph_entities: list[tuple[str, float]] = []
    graph_results: list[dict] = []

    # Dedicated connection: a failing Cypher call must not abort
    # the running recall transaction.
    conn = get_conn()
    try:
        cur = conn.cursor()
        tables = _retrieval_tables(cur, project)
        if project and not include_global:
            tables = [table for table in tables if table != "memories"]

        # 1. Find connected entities via Apache AGE (any edge direction/type)
        cur.execute("SET search_path TO ag_catalog, public")
        label = _scope_label(project)
        cur.execute(
            f"SELECT * FROM cypher('memory_graph', $$ "
            f"MATCH (a:{label} {{name: {name_json}}})-[r]-(b) "
            f"RETURN b.name, coalesce(r.weight, 1.0) LIMIT 10 $$) AS "
            f"(name agtype, weight agtype)"
        )
        graph_entities = [
            (_ag_str(name), float(str(weight)))
            for name, weight in cur.fetchall()
        ]

        # 2. Fetch memories of connected entities
        extra_conditions, extra_params = _extra_conditions(updated_after,
                                                            metadata_filter)
        existing_ids = {r["id"] for r in existing_results}
        project_names = _project_names(cur)
        for entity_name, weight in graph_entities:
            for table in tables:
                cur.execute(
                    f"SELECT * FROM {table} "
                    f"WHERE entity = %s AND superseded_by IS NULL AND deleted_at IS NULL"
                    + ("" if not extra_conditions
                       else " AND " + " AND ".join(extra_conditions))
                    + " LIMIT 5",
                    [entity_name] + extra_params,
                )
                for row in cur.fetchall():
                    d = dict(zip([desc[0] for desc in cur.description], row))
                    d.pop("embedding", None)
                    if d["id"] in existing_ids:
                        continue
                    # Report the project name (not the scope kind), like every
                    # other read path does.
                    d["_project"] = ("global" if table == "memories"
                                     else project_names.get(table[4:], table[4:]))
                    d["_graph_weight"] = weight
                    d["_graph_source"] = entity_name
                    # Graph candidates are not retrieved by relevance: they have
                    # no similarity and no RRF contribution. Mark them as such
                    # instead of leaving the score fields undefined.
                    d["_similarity"] = None
                    d["_distance"] = None
                    d["_relevance_score"] = None
                    d["_rrf_score"] = 0.0
                    d["_rrf_norm"] = 0.0
                    if include_decay:
                        d["_decay_score"] = _decay_score(
                            d.get("created_at"), d.get("updated_at"),
                            d.get("access_count", 0), d.get("importance", 1.0),
                        )
                        d["_decay_norm"] = _decay_norm(
                            d.get("created_at"), d.get("updated_at"),
                            d.get("access_count", 0), d.get("importance", 1.0),
                        )
                    existing_ids.add(d["id"])
                    graph_results.append(d)

        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.warning("Graph boost failed: %s", e)
        return existing_results
    finally:
        conn.close()

    # 3. Fuse with existing results
    # Keep the boost relative to the best score so graph candidates cannot
    # jump to the top on weight alone.
    max_rrf = max((r.get("_rrf_score", 0.0) for r in existing_results), default=0.0)
    if max_rrf <= 0:
        max_rrf = 1.0 / 61.0

    all_results = existing_results + graph_results
    for r in all_results:
        rrf_norm = r.get("_rrf_norm")
        if rrf_norm is None:
            rrf_norm = r.get("_rrf_score", 0.0) / max_rrf
            r["_rrf_norm"] = rrf_norm
        if "_graph_weight" in r:
            r["_graph_boost"] = graph_boost_weight * r["_graph_weight"]
        graph_boost = r.get("_graph_boost", 0.0)
        decay_norm = r.get("_decay_norm", 0.0) if include_decay else 0.0
        # Graph candidates were not retrieved by relevance: they count as
        # relevance 0 and are lifted by the boost only.
        relevance = r.get("_relevance_score") or 0.0

        # Final score: relevance (blended with decay) plus the graph boost,
        # always inside the documented 0..1 range.
        r["_final_score"] = _clamp01(
            _blend_scores(relevance, decay_norm) + graph_boost
        )

    # Sort by final score and limit
    all_results.sort(key=lambda x: -x.get("_final_score", 0))
    return all_results[:limit]
