import hashlib
import json
import math
import os
import re
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import psycopg2
import pytest
from psycopg2 import sql

DIM = 768
DEFAULT_TEST_DB = "postgresql://memory:geheim@localhost:5432/loci_test"


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _embedding(text: str) -> list[float]:
    vec = [0.0] * DIM
    for word in _tokens(text):
        digest = hashlib.sha256(word.encode()).digest()
        vec[int.from_bytes(digest[:4], "big") % DIM] += 1.0
        vec[int.from_bytes(digest[4:8], "big") % DIM] += 0.5
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return [1.0 / math.sqrt(DIM)] * DIM
    return [v / norm for v in vec]


def _contradiction(query: str, document: str) -> float:
    q = set(_tokens(query))
    d = set(_tokens(document))
    if ("not" in q) != ("not" in d) and (q | d) - {"not"}:
        return 0.95
    return 0.05


class _FakeInference(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send({"status": "ok"})
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        payload = json.loads(raw or b"{}")
        if self.path == "/embed":
            texts = payload.get("texts", [])
            self._send({"embeddings": [_embedding(t) for t in texts]})
        elif self.path == "/nli":
            query = payload.get("query", "")
            documents = payload.get("documents", [])
            self._send({"contradiction_scores": [
                _contradiction(query, d) for d in documents
            ]})
        else:
            self._send({"error": "not found"}, 404)


_fake_inference = ThreadingHTTPServer(("127.0.0.1", 0), _FakeInference)
threading.Thread(target=_fake_inference.serve_forever, daemon=True).start()
os.environ["MEMORY_INFERENCE_URL"] = (
    f"http://127.0.0.1:{_fake_inference.server_address[1]}"
)

TEST_DATABASE_URL = os.environ.get("MEMORY_TEST_DATABASE_URL", DEFAULT_TEST_DB)
_parsed = urllib.parse.urlparse(TEST_DATABASE_URL)
_admin_url = _parsed._replace(path="/postgres").geturl()
_test_dbname = _parsed.path.lstrip("/")

_admin = psycopg2.connect(_admin_url)
_admin.autocommit = True
with _admin.cursor() as _cur:
    _cur.execute(
        "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
        "WHERE datname = %s AND pid <> pg_backend_pid()",
        (_test_dbname,),
    )
    _cur.execute(sql.SQL("DROP DATABASE IF EXISTS {}").format(
        sql.Identifier(_test_dbname)))
    _cur.execute(sql.SQL("CREATE DATABASE {}").format(
        sql.Identifier(_test_dbname)))
_admin.close()

os.environ["MEMORY_DATABASE_URL"] = TEST_DATABASE_URL

from mcp_server.db import ensure_schema, get_conn  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _schema():
    ensure_schema()


@pytest.fixture(autouse=True)
def clean_db():
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("SET search_path TO ag_catalog, public")
        cur.execute(
            "SELECT * FROM cypher('memory_graph', $$ "
            "MATCH (n) DETACH DELETE n $$) AS (x agtype)"
        )
        cur.execute("SET search_path TO public")
        cur.execute(
            "SELECT tablename FROM pg_tables "
            "WHERE schemaname = 'public' AND tablename LIKE 'mem\\_%'"
        )
        for (table,) in cur.fetchall():
            cur.execute(sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(
                sql.Identifier(table)))
        cur.execute(
            "TRUNCATE memories, entities, relations, projects "
            "RESTART IDENTITY CASCADE"
        )
        conn.commit()
    finally:
        conn.close()
    yield
