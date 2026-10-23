"""Re-embed every stored memory with the document prompt.

Needed once after switching to prompt-aware embeddings (query vs document);
older rows were embedded without the prompt and are not comparable to new
query embeddings any more.

Usage:
    python -m mcp_server.reembed
"""

import logging
import sys

from mcp_server.db import get_conn
from mcp_server.inference_gateway import embed, health

logger = logging.getLogger(__name__)

BATCH_SIZE = 32


def _memory_tables(cur) -> list[str]:
    cur.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'public' AND table_name ~ '^mem_[A-Za-z0-9_]+$' "
        "ORDER BY table_name"
    )
    return ["memories"] + [r[0] for r in cur.fetchall()]


def reembed(batch_size: int = BATCH_SIZE) -> dict:
    if not health():
        raise SystemExit("Inference API not reachable - cannot re-embed")

    conn = get_conn()
    summary: dict[str, int] = {}
    try:
        cur = conn.cursor()
        for table in _memory_tables(cur):
            cur.execute(f"SELECT id, content FROM {table} ORDER BY id")
            rows = cur.fetchall()
            done = 0
            for start in range(0, len(rows), batch_size):
                chunk = rows[start:start + batch_size]
                vectors = embed([row[1] for row in chunk], mode="document")
                cur.executemany(
                    f"UPDATE {table} SET embedding = %s::vector WHERE id = %s",
                    [(vec, row[0]) for vec, row in zip(vectors, chunk)],
                )
                done += len(chunk)
                logger.info("%s: %s/%s", table, done, len(rows))
            conn.commit()
            summary[table] = done
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(reembed())
    sys.exit(0)
