import hashlib
import json
import re

from mcp_server.config import FTS_LANGUAGE, _IDENT_RE
from mcp_server.db import get_conn

def _safe_ident(value: str, kind: str = "identifier") -> str:
    if not _IDENT_RE.match(value or ""):
        raise ValueError(
            f"invalid {kind} {value!r}: only letters, digits and '_' allowed "
            f"(must not start with a digit)"
        )
    return value


def _norm_project(project: str | None) -> str | None:
    """Canonical project name: trimmed, or None for the global scope.

    "global" is the reserved name of the shared scope, so it can never be a
    project of its own (previously `project="global"` created an isolated
    `Project_global` graph that was invisible to global traversal).
    """
    if project is None:
        return None
    name = str(project).strip()
    if not name or name.lower() == "global":
        return None
    return name


def _project_slug(project: str) -> str:
    """Stable, SQL-safe suffix for a project name.

    Plain lower-case identifiers up to 40 chars are used verbatim (backwards
    compatible). Everything else is lower-cased, truncated and suffixed with a
    hash of the *original* name, so case variants ("Team"/"team"), long names
    and names with special characters never share a table or index name.
    """
    project = _norm_project(project)
    if not project:
        raise ValueError("project name must not be empty")
    if _IDENT_RE.match(project) and project == project.lower() and len(project) <= 40:
        return project
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", project).strip("_").lower()
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"p_{cleaned}"
    digest = hashlib.sha1(project.encode()).hexdigest()[:8]
    return f"{cleaned[:40]}_{digest}"


def _project_suffix(project: str) -> str:
    """Stable, SQL-safe, length-bounded table suffix for a project name."""
    return _project_slug(project)


def _scope_key(project: str | None) -> str:
    """Scope value stored on entity rows and graph nodes."""
    return _norm_project(project) or "global"


def _scope_label(project: str | None) -> str:
    """Apache AGE node label for one scope (project or global)."""
    name = _norm_project(project)
    return f"Project_{_project_suffix(name)}" if name else "Entity"


def _table_exists(cur, table: str) -> bool:
    cur.execute(
        "SELECT EXISTS (SELECT FROM information_schema.tables "
        "WHERE table_schema = 'public' AND table_name = %s)",
        (table,),
    )
    return bool(cur.fetchone()[0])


def _ag_str(value) -> str:
    """Unwrap an AGE agtype string ('"foo"' -> 'foo')."""
    text = str(value)
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text[1:-1]
    return text


def _project_tables(cur) -> list[str]:
    """All project-scoped memory tables (mem_<project>)."""
    cur.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'public' AND table_name ~ '^mem_[A-Za-z0-9_]+$' "
        "ORDER BY table_name"
    )
    return [r[0] for r in cur.fetchall()]


def _all_mem_tables(cur) -> list[str]:
    """All project tables first, then the global one.

    The order matters: when the same content exists in a project and in the
    global scope, the project row wins - in retrieval, in a browse and when
    a memory id is updated without a project. The global table comes last
    so that it is the one that is dropped by the cross-scope dedup.
    """
    return _project_tables(cur) + ["memories"]


def _retrieval_tables(cur, project: str | None) -> list[str]:
    """Tables consulted by recall: the project table (if any) plus the global one.

    Project hits win over global duplicates, but global facts remain visible
    from inside a project (and vice versa).
    """
    tables: list[str] = []
    project = _norm_project(project)
    if project:
        safe = _project_suffix(project)
        if _table_exists(cur, f"mem_{safe}"):
            tables.append(f"mem_{safe}")
    tables.append("memories")
    return tables


def _ensure_fts_index(cur, table: str, index_name: str) -> None:
    """Make sure the full-text index of `table` uses the configured dictionary.

    An index built with a different dictionary cannot be used by the query
    side, so it is dropped and rebuilt.
    """
    cur.execute(
        "SELECT indexname, indexdef FROM pg_indexes WHERE schemaname = 'public' "
        "AND tablename = %s AND indexdef LIKE '%%to_tsvector%%'",
        (table,),
    )
    for name, indexdef in cur.fetchall():
        if f"to_tsvector('{FTS_LANGUAGE}'" not in indexdef:
            cur.execute(f'DROP INDEX IF EXISTS "{name}"')
    cur.execute(
        "SELECT 1 FROM pg_indexes WHERE schemaname = 'public' AND tablename = %s "
        "AND indexdef LIKE '%%to_tsvector%%'",
        (table,),
    )
    if not cur.fetchone():
        cur.execute(
            f"CREATE INDEX {index_name} ON {table} USING GIN "
            f"(to_tsvector('{FTS_LANGUAGE}', content))"
        )


def _migrate_fts_indexes() -> int:
    """Rebuild full-text indexes that use a different dictionary.

    Returns the number of rebuilt indexes.
    """
    conn = get_conn()
    rebuilt = 0
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_name LIKE 'mem\\_%'"
        )
        for (table,) in cur.fetchall():
            cur.execute(
                "SELECT indexdef FROM pg_indexes WHERE schemaname = 'public' "
                "AND tablename = %s AND indexdef LIKE '%%to_tsvector%%'",
                (table,),
            )
            stale = [d for (d,) in cur.fetchall()
                     if f"to_tsvector('{FTS_LANGUAGE}'" not in d]
            if not stale:
                continue
            safe = table[len("mem_"):]
            idx_digest = hashlib.sha1(safe.encode()).hexdigest()[:10]
            _ensure_fts_index(cur, table, f"idx_{idx_digest}_tsv")
            rebuilt += 1
        conn.commit()
    finally:
        conn.close()
    return rebuilt


def _ensure_project_table(cur, project: str) -> None:
    project = _norm_project(project)
    if not project:
        raise ValueError("project name must not be empty")
    safe = _project_suffix(project)
    # Short, hash-based index names: long project names would otherwise
    # truncate `idx_mem_<name>_embedding` and `..._hash` to the same 63-byte
    # identifier and fail with "relation already exists".
    idx_digest = hashlib.sha1(safe.encode()).hexdigest()[:10]
    emb_idx = f"idx_{idx_digest}_emb"
    hash_idx = f"idx_{idx_digest}_hash"
    tsv_idx = f"idx_{idx_digest}_tsv"
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS mem_{safe} (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            content_hash TEXT NOT NULL DEFAULT '',
            embedding vector(768),
            mem_type TEXT NOT NULL DEFAULT 'fact',
            scope TEXT NOT NULL DEFAULT 'project',
            entity TEXT,
            importance FLOAT DEFAULT 1.0,
            metadata JSONB DEFAULT '{{}}',
            access_count INT DEFAULT 0,
            superseded_by TEXT,
            deleted_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    cur.execute(f"ALTER TABLE mem_{safe} ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ")
    # Look indexes up by table + definition: long project names get truncated
    # by PostgreSQL to 63 characters, so the exact index name is unreliable.
    cur.execute(
        "SELECT 1 FROM pg_indexes WHERE schemaname = 'public' AND tablename = %s "
        "AND indexdef LIKE '%%ivfflat%%'",
        (f"mem_{safe}",),
    )
    if not cur.fetchone():
        # Dynamic lists based on expected scaling
        # For 768d: 100 up to ~50k, 200-400 up to ~200k records
        cur.execute(f"SELECT COUNT(*) FROM mem_{safe}")
        count = cur.fetchone()[0] if cur.rowcount > 0 else 0
        
        if count < 50000:
            lists = 100
        elif count < 200000:
            lists = 200
        else:
            lists = 400
        
        cur.execute(f"""
            CREATE INDEX {emb_idx}
            ON mem_{safe} USING ivfflat (embedding vector_cosine_ops) WITH (lists = {lists})
        """)
    cur.execute(
        "SELECT 1 FROM pg_indexes WHERE schemaname = 'public' AND tablename = %s "
        "AND indexdef LIKE '%%content_hash%%'",
        (f"mem_{safe}",),
    )
    if not cur.fetchone():
        cur.execute(f"CREATE INDEX {hash_idx} ON mem_{safe}(content_hash)")
    _ensure_fts_index(cur, f"mem_{safe}", tsv_idx)
    cur.execute(
        "DELETE FROM projects WHERE name = %s AND suffix <> %s",
        (project, safe),
    )
    cur.execute(
        "INSERT INTO projects (suffix, name) VALUES (%s, %s) "
        "ON CONFLICT (suffix) DO UPDATE SET name = EXCLUDED.name",
        (safe, project),
    )


def _project_names(cur) -> dict[str, str]:
    """Map table suffix -> project name from the registry."""
    try:
        cur.execute("SELECT suffix, name FROM projects")
        return {row[0]: row[1] for row in cur.fetchall()}
    except Exception:
        return {}


def _mem_table(project: str | None) -> str:
    project = _norm_project(project)
    if project:
        return f"mem_{_project_suffix(project)}"
    return "memories"


def _count_entities() -> int:
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM entities")
        return cur.fetchone()[0]
    except Exception:
        return 0
    finally:
        conn.close()
