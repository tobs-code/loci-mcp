import json
import logging
from typing import Any

from mcp_server.config import FTS_WEIGHT, RRF_K
from mcp_server.db import get_conn
from mcp_server.scopes import _ag_str, _retrieval_tables, _scope_label
from mcp_server.scoring import (
    _blend_scores,
    _decay_norm,
    _decay_score,
    _fts_query,
)

logger = logging.getLogger(__name__)

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
) -> list[dict]:
    """
    Reciprocal Rank Fusion over vector + FTS results.

    Retrieval runs against the project table (if any) and the global table;
    project rows win when both contain the same content. Scores are normalized
    so that decay can reorder results without drowning out relevance.
    """
    tables = tables if tables is not None else _retrieval_tables(cur, project)
    if not tables:
        return []

    max_distance = None if min_similarity is None else 1.0 - min_similarity
    fts_query = _fts_query(query_text)

    # (score, id, table) tuples, merged across tables before ranking
    vec_candidates: list[tuple[float, str, str]] = []
    text_candidates: list[tuple[float, str, str]] = []

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
        where = " AND ".join(conditions)

        # 1. Vector Search (Top 50 per table)
        if query_embedding is not None:
            sql = (
                f"SELECT id, embedding <=> %s::vector AS distance FROM {table} "
                f"WHERE {where} AND embedding IS NOT NULL"
            )
            vec_params: list[Any] = [query_embedding] + params
            if max_distance is not None:
                sql += " AND embedding <=> %s::vector <= %s"
                vec_params += [query_embedding, max_distance]
            sql += " ORDER BY distance ASC LIMIT 50"
            cur.execute(sql, vec_params)
            vec_candidates.extend(
                (float(dist), row_id, table) for row_id, dist in cur.fetchall()
            )

        # 2. FTS Search (Top 50 per table), OR-joined terms
        if fts_query:
            sql = (
                f"SELECT id, ts_rank(to_tsvector('german', content), "
                f"websearch_to_tsquery('german', %s)) AS rank FROM {table} "
                f"WHERE to_tsvector('german', content) @@ "
                f"websearch_to_tsquery('german', %s)"
            )
            fts_params: list[Any] = [fts_query, fts_query]
            if max_distance is not None and query_embedding is not None:
                sql += " AND embedding IS NOT NULL AND embedding <=> %s::vector <= %s"
                fts_params += [query_embedding, max_distance]
            sql += f" AND {where} ORDER BY rank DESC LIMIT 50"
            cur.execute(sql, fts_params + params)
            text_candidates.extend(
                (float(rank), row_id, table) for row_id, rank in cur.fetchall()
            )

    vec_candidates.sort(key=lambda c: c[0])
    text_candidates.sort(key=lambda c: -c[0])

    # 3. RRF Fusion
    vector_results: dict[str, float] = {}
    text_results: dict[str, float] = {}
    similarities: dict[str, float] = {}
    table_of: dict[str, str] = {}

    for rank, (distance, row_id, table) in enumerate(vec_candidates[:50], 1):
        vector_results[row_id] = 1.0 / (k + rank)
        similarities[row_id] = 1.0 - distance
        table_of.setdefault(row_id, table)
    for rank, (_, row_id, table) in enumerate(text_candidates[:50], 1):
        text_results[row_id] = FTS_WEIGHT / (k + rank)
        table_of.setdefault(row_id, table)

    fused_scores = {
        row_id: vector_results.get(row_id, 0.0) + text_results.get(row_id, 0.0)
        for row_id in set(vector_results) | set(text_results)
    }
    if not fused_scores:
        return []

    max_rrf = max(fused_scores.values()) or 1.0
    # Do not cut before cross-table dedup: many duplicate contents could
    # otherwise hide unique rows that would fit into the limit.
    top_ids = sorted(fused_scores, key=lambda x: -fused_scores[x])

    # 4. Load details per table (project first), dropping cross-table duplicates
    rows: list[dict] = []
    seen_hashes: set[str] = set()
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
            c_hash = d.get("content_hash") or ""
            if c_hash and c_hash in seen_hashes:
                continue
            if c_hash:
                seen_hashes.add(c_hash)
            d["_rrf_score"] = fused_scores.get(row_id, 0.0)
            d["_rrf_norm"] = d["_rrf_score"] / max_rrf
            if row_id in similarities:
                d["_similarity"] = round(similarities[row_id], 4)
                d["_distance"] = round(1.0 - similarities[row_id], 4)
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

    # 5. Re-rank
    if include_decay:
        for r in rows:
            r["_final_score"] = _blend_scores(r["_rrf_norm"], r["_decay_norm"])
        rows.sort(key=lambda x: -x.get("_final_score", 0))
    else:
        rows.sort(key=lambda x: -x.get("_rrf_score", 0))

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
) -> list[dict]:
    """
    Wrapper for search functions. Uses RRF when hybrid=True, otherwise legacy search.

    `tables` overrides the table set (default: project table + global).
    """
    if hybrid and (query_embedding is not None or query_text):
        return _search_memories_rrf(
            cur, query_embedding, query_text, mem_type, entity, limit, project,
            include_decay, include_embedding, include_superseded, min_similarity,
            tables=tables,
        )

    # Legacy fallback for individual retrievers
    tables = tables if tables is not None else _retrieval_tables(cur, project)
    if not tables:
        return []

    max_distance = None if min_similarity is None else 1.0 - min_similarity
    has_vec = query_embedding is not None
    fts_query = _fts_query(query_text)

    rows: list[dict] = []
    seen_hashes: set[str] = set()
    for table in tables:
        conditions: list[str] = ["TRUE"]
        params: list[Any] = []
        params_vec: list[Any] = []

        if has_vec:
            col = "*, embedding <=> %s::vector AS distance"
            conditions.append("embedding IS NOT NULL")
            order = "ORDER BY distance ASC"
            params_vec.append(query_embedding)
            if max_distance is not None:
                conditions.append("embedding <=> %s::vector <= %s")
                params_vec.extend([query_embedding, max_distance])
        else:
            col = "*"
            order = "ORDER BY updated_at DESC"

        if fts_query:
            conditions.append(
                "to_tsvector('german', content) @@ websearch_to_tsquery('german', %s)"
            )
            params.append(fts_query)
        if mem_type:
            conditions.append("mem_type = %s")
            params.append(mem_type)
        if entity:
            conditions.append("entity = %s")
            params.append(entity)

        if include_superseded:
            conditions.append("deleted_at IS NULL")
        else:
            conditions.append("superseded_by IS NULL AND deleted_at IS NULL")

        where = " AND ".join(conditions)
        cur.execute(
            f"SELECT {col}, created_at, updated_at, access_count, importance "
            f"FROM {table} WHERE {where} {order} LIMIT %s",
            params_vec + params + [limit],
        )

        colnames = [desc[0] for desc in cur.description] if cur.description else []
        for row in cur.fetchall():
            d = dict(zip(colnames, row))
            if "distance" in d and d["distance"] is not None:
                d["_distance"] = float(d.pop("distance"))
                d["_similarity"] = round(1.0 - d["_distance"], 4)
            else:
                d.pop("distance", None)
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
    else:
        rows.sort(key=lambda x: x.get("updated_at") or _now(), reverse=True)

    if include_decay:
        for r in rows:
            if has_vec and "_similarity" in r:
                r["_final_score"] = _blend_scores(r["_similarity"], r["_decay_norm"])
            else:
                # Nothing relevant to blend with: recency/frequency only.
                r["_final_score"] = r["_decay_norm"]
        rows.sort(key=lambda x: -x.get("_final_score", 0))

    return rows[:limit]


def _search_memories_graph_boost(
    existing_results: list[dict],
    entity: str,
    limit: int,
    project: str | None,
    include_decay: bool = False,
    graph_boost_weight: float = 0.3,  # 30% boost for graph candidates
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
        existing_ids = {r["id"] for r in existing_results}
        for entity_name, weight in graph_entities:
            for table in tables:
                cur.execute(
                    f"SELECT * FROM {table} "
                    f"WHERE entity = %s AND superseded_by IS NULL AND deleted_at IS NULL "
                    f"LIMIT 5",
                    (entity_name,)
                )
                for row in cur.fetchall():
                    d = dict(zip([desc[0] for desc in cur.description], row))
                    d.pop("embedding", None)
                    if d["id"] in existing_ids:
                        continue
                    d["_graph_weight"] = weight
                    d["_graph_source"] = entity_name
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

    if not graph_results:
        return existing_results

    # 3. Fuse with existing results
    # Keep the boost relative to the best RRF score so graph candidates
    # cannot jump to the top on weight alone.
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

        # Final score: relevance (RRF) + graph boost + at most 30% decay
        r["_final_score"] = _blend_scores(rrf_norm, decay_norm) + graph_boost

    # Sort by final score and limit
    all_results.sort(key=lambda x: -x.get("_final_score", 0))
    return all_results[:limit * 2]
