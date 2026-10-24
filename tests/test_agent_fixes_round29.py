"""Regression tests for the round-29 live-audit fixes (blocks A and B)."""

import asyncio
import logging
import time

import pytest

from mcp_server import inference_client, search
from mcp_server.app import EnvelopeTool
from mcp_server.backup import import_memories
from mcp_server.db import get_conn
from mcp_server.graph import list_relations, relate
from mcp_server.scope_catalog import _mem_table, _norm_entity
from mcp_server.storage import clear_dangling_supersede
from mcp_server.store import MemoryStore
from mcp_server.tools.maintenance import contradictions_report, server_health
from mcp_server.tools.memory import (
    forget,
    remember,
    remember_batch,
    update_batch,
)
from mcp_server.tools.read import get_entity, list_memories, recall


def test_entity_name_with_dollar_quote_is_refused():
    malicious = "R29$$) AS (n agtype); DROP TABLE memories; --"
    with pytest.raises(ValueError, match=r"\$"):
        _norm_entity(malicious)
    refused = relate(malicious, "R29Safe", "USES")
    assert refused["ok"] is False
    assert "invalid entity" in refused["error"]

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass('memories')")
            assert cur.fetchone()[0] == "memories"
    finally:
        conn.rollback()
        conn.close()


def test_mark_superseded_only_touches_the_resolved_table():
    class Cursor:
        def __init__(self):
            self.executed = []
            self.rowcount = 0

        def execute(self, sql, params=()):
            self.executed.append((sql, params))
            return self

    class Conn:
        def __init__(self):
            self.cursor_instance = Cursor()

        def cursor(self):
            return self.cursor_instance

    store = MemoryStore(conn=Conn())
    store.mark_superseded("old", "new", project="api", target="global")
    statements = [sql for sql, _ in store.conn.cursor_instance.executed]
    assert statements == [
        ("UPDATE memories SET superseded_by = %s, updated_at = NOW() "
         "WHERE id = %s")
    ]

    store = MemoryStore(conn=Conn())
    store.mark_superseded("old", "new", project="api", target="project")
    statements = [sql for sql, _ in store.conn.cursor_instance.executed]
    assert statements == [
        ("UPDATE mem_api SET superseded_by = %s, updated_at = NOW() "
         "WHERE id = %s")
    ]


def test_clear_dangling_keeps_valid_cross_scope_link():
    project = "round29-dangling"
    global_row = remember("R29 shared fact", importance=0.9)
    project_row = remember("R29 project fact", project=project)
    assert global_row["ok"] and project_row["ok"]

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE {_mem_table(project)} SET superseded_by = %s "
                f"WHERE id = %s",
                (global_row["id"], project_row["id"]),
            )
            repaired = clear_dangling_supersede(
                cur, [_mem_table(project)])
            assert repaired == 0
            cur.execute(
                f"SELECT superseded_by FROM {_mem_table(project)} "
                f"WHERE id = %s",
                (project_row["id"],),
            )
            assert cur.fetchone()[0] == global_row["id"]
    finally:
        conn.rollback()
        conn.close()


def test_rerank_keeps_history_penalty_and_exact_floor(monkeypatch):
    monkeypatch.setattr(search, "RERANK_ENABLED", True)
    monkeypatch.setattr(search, "RERANK_ALWAYS", True)
    monkeypatch.setattr(
        search, "rerank", lambda query, documents: [3.0, 0.0, 1.0])

    # The first stage already applied the history penalty and the exact-match
    # floor. Reranking must not undo either.
    history = {
        "id": "old", "content": "old wording", "superseded_by": "new",
        "_relevance_score": 0.25, "_rrf_score": 1.0, "_project": "global",
        "_history_penalty": 0.5,
    }
    current = {
        "id": "new", "content": "new wording", "superseded_by": None,
        "_relevance_score": 0.5, "_rrf_score": 0.9, "_project": "global",
    }
    exact = {
        "id": "exact", "content": "ZEBRAFISH_TOKEN", "superseded_by": None,
        "_relevance_score": 0.95, "_rrf_score": 0.8, "_project": "global",
        "_exact_match": "content",
    }
    stats = {}
    rows = search._rerank_rows([history, current, exact], "q", stats, False)

    assert history["_history_penalty"] == 0.5
    assert history["_relevance_score"] == 0.25
    assert exact["_relevance_score"] >= 0.95
    assert stats["reranked"] == 3
    assert rows[0] is exact
    assert rows.index(current) < rows.index(history)


def test_envelope_tool_offloads_sync_calls_and_keeps_elicitation():
    import threading

    from mcp.shared.exceptions import UrlElicitationRequiredError

    loop_thread = threading.get_ident()

    def worker_tool() -> dict:
        assert threading.get_ident() != loop_thread
        return {"ok": True}

    tool = EnvelopeTool.from_function(worker_tool)
    assert asyncio.run(tool.run({})) == {"ok": True}

    def elicit() -> dict:
        raise UrlElicitationRequiredError([], "needs url")

    tool = EnvelopeTool.from_function(elicit)
    with pytest.raises(UrlElicitationRequiredError):
        asyncio.run(tool.run({}))

def test_recall_limit_above_50_uses_a_large_enough_candidate_pool():
    project = "round29-b1"
    facts = [{"content": f"R29B1 token item {i}"} for i in range(60)]
    stored = remember_batch(facts=facts, project=project)
    assert stored["ok"] and stored["succeeded"] == 60

    found = recall(query="R29B1 token", project=project, limit=60)
    assert found["ok"]
    assert len(found["results"]) == 60, found.get("warnings")


def test_remember_batch_validates_per_fact_threshold_and_dedupe():
    project = "round29-b2"
    result = remember_batch(facts=[
        {"content": "R29B2 numeric string threshold",
         "fuzzy_threshold": "0.85"},
        {"content": "R29B2 bad dedupe", "dedupe": "bogus"},
        {"content": "R29B2 bad supersede flag", "supersede_last": "false"},
    ], project=project)
    assert result["ok"]
    by_content = {entry["content"]: entry for entry in result["results"]}
    assert by_content["R29B2 numeric string threshold"]["status"] == "stored"
    assert by_content["R29B2 bad dedupe"]["status"] == "error"
    assert "dedupe" in by_content["R29B2 bad dedupe"]["error"]
    assert by_content["R29B2 bad supersede flag"]["status"] == "error"
    assert "boolean" in by_content["R29B2 bad supersede flag"]["error"]


def test_update_batch_rejects_string_booleans_instead_of_truthiness():
    project = "round29-b13"
    stored = remember("R29B13 ttl fact", project=project,
                      expires_at="2099-01-01")
    assert stored["ok"]

    result = update_batch(updates=[
        {"memory_id": stored["id"], "clear_expires_at": "false"},
        {"memory_id": stored["id"], "touch": "false"},
        {"memory_id": stored["id"], "importance": 0.6},
    ], project=project)
    assert result["ok"] is True
    assert result["status"] == "partial"
    errors = [entry for entry in result["results"]
              if entry["status"] == "error"]
    assert len(errors) == 2
    assert all("boolean" in entry["error"] for entry in errors)

    current = list_memories(memory_id=stored["id"], project=project)
    assert current["ok"]
    assert current["memory"]["expires_at"] is not None


def test_project_supersede_content_falls_back_to_global_without_entity():
    global_row = remember("R29B3 shared fact", entity="R29B3Old")
    assert global_row["ok"]
    project = "round29-b3"
    replaced = remember("R29B3 replacement", project=project,
                        entity="R29B3New",
                        supersede_content="R29B3 shared fact")
    assert replaced["ok"]
    assert global_row["id"] in replaced.get("superseded", [])
    assert not any("supersede_content skipped" in warning
                   for warning in replaced.get("warnings", []))


def test_semantic_dedupe_ignores_relation_context_rows():
    project = "round29-b4"
    edge = relate("R29B4A", "R29B4B", "R29B4REL",
                  context="R29B4 edge sentence", project=project)
    assert edge["ok"], edge
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE {_mem_table(project)} SET content_hash = %s "
                f"WHERE content = %s",
                ("r29b4-forced-different-hash", "R29B4 edge sentence"),
            )
            assert cur.rowcount == 1
        conn.commit()
    finally:
        conn.close()

    stored = remember("R29B4 edge sentence", project=project,
                      dedupe="semantic", fuzzy_threshold=0.99)
    assert stored["ok"]
    assert stored["status"] == "stored", stored


def test_recall_include_deleted_returns_deleted_at():
    project = "round29-b5"
    stored = remember("R29B5 hidden fact", project=project)
    assert stored["ok"]
    assert forget(memory_id=stored["id"], project=project)["ok"]

    found = recall(query="R29B5 hidden fact", project=project,
                   include_deleted=True)
    assert found["ok"] and found["results"]
    row = found["results"][0]
    assert row["forgotten"] is True
    assert row["deleted_at"] is not None


def test_list_relations_global_does_not_leak_project_edges():
    project = "round29-b6"
    assert relate("R29B6GlobalA", "R29B6GlobalB", "R29B6REL")["ok"]
    assert relate("R29B6ProjA", "R29B6ProjB", "R29B6REL",
                  project=project)["ok"]

    page = list_relations(project="global", relation_type="R29B6REL")
    assert page["ok"]
    assert page["total"] == 1
    assert page["results"][0]["from_entity"] == "R29B6GlobalA"
    assert page["results"][0]["project"] == "global"


def test_contradiction_report_limit_does_not_cap_the_seed_scan():
    project = "round29-b7"
    facts = [{"content": f"R29B7 fact number {i}", "entity": "R29B7"}
             for i in range(6)]
    assert remember_batch(facts=facts, project=project)["succeeded"] == 6

    report = contradictions_report(project=project, entity="R29B7",
                                   limit=2, budget_seconds=20)
    assert report["ok"]
    assert report["checked_pairs"] == 15, report


def test_import_survives_a_broken_graph_record():
    payload = {
        "records": [{"content": "R29B8 memory survives",
                     "project": "round29-b8"}],
        "graph": {
            "entities": [{"name": "R29B8Bad$Name",
                          "project": "round29-b8"}],
            "relations": [{"from_entity": "R29B8A", "to_entity": "R29B8B",
                           "relation_type": "R29B8REL", "weight": "nope",
                           "project": "round29-b8"}],
        },
    }
    result = import_memories(data=payload, format="json")
    assert result["ok"], result
    assert result["imported"] == 1
    assert result.get("errors")


def test_memory_ids_bulk_lookup_uses_one_connection(monkeypatch):
    project = "round29-b9"
    ids = []
    for i in range(3):
        stored = remember(f"R29B9 fact {i}", project=project)
        assert stored["ok"]
        ids.append(stored["id"])

    from mcp_server.tools import read as read_tools
    real_store = read_tools.MemoryStore
    opened = []

    class CountingStore(real_store):
        def __init__(self, *args, **kwargs):
            opened.append(1)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(read_tools, "MemoryStore", CountingStore)
    found = list_memories(memory_ids=ids, project=project)
    assert found["ok"]
    assert len(found["results"]) == 3
    assert len(opened) == 1


def test_server_health_logs_a_failed_postgres_ping(monkeypatch, caplog):
    from mcp_server.tools import maintenance

    class BrokenStore:
        def __init__(self):
            raise RuntimeError("postgres down")

    monkeypatch.setattr(maintenance, "MemoryStore", BrokenStore)
    with caplog.at_level(logging.WARNING):
        result = server_health()
    assert result["ok"]
    assert result["postgres"] is False
    assert any("postgres ping failed" in record.message
               for record in caplog.records)


def test_nli_retry_sleep_respects_the_deadline(monkeypatch):
    sleeps = []

    class Response:
        def __init__(self, bad):
            self.bad = bad

        def raise_for_status(self):
            return None

        def json(self):
            if self.bad:
                raise ValueError("invalid json")
            return {"contradiction_scores": [0.0]}

    class Session:
        def __init__(self):
            self.calls = 0

        def post(self, *args, **kwargs):
            self.calls += 1
            return Response(bad=self.calls == 1)

    session = Session()
    monkeypatch.setattr(inference_client, "_get_session", lambda: session)
    monkeypatch.setattr(inference_client.time, "sleep",
                        lambda delay: sleeps.append(delay))

    deadline = time.monotonic() + 0.05
    scores, _partial, _failed = inference_client.nli_detailed(
        "q", ["d"], deadline=deadline, timeout=5)
    assert scores == [0.0]
    assert sleeps and sleeps[0] <= 0.06


def test_entity_total_does_not_hide_an_expired_copy_as_duplicate():
    project = "round29-b12"
    global_row = remember("R29B12 same fact", entity="R29B12")
    project_row = remember("R29B12 same fact", project=project,
                           entity="R29B12")
    assert global_row["ok"] and project_row["ok"]

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE {_mem_table(project)} "
                f"SET expires_at = NOW() - INTERVAL '1 day' WHERE id = %s",
                (project_row["id"],),
            )
        conn.commit()
    finally:
        conn.close()

    details = get_entity("R29B12", project=project, include_expired=True)
    assert details["ok"]
    assert details["total"] == 2, details
    assert len(details["memories"]) == 2

def test_list_projects_counts_a_graph_only_scope():
    project = "round29-c2"
    from mcp_server.tools.scopes import drop_project, list_projects
    drop_project(project)
    assert relate("R29C2A", "R29C2B", "R29C2REL", project=project)["ok"]
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE {_mem_table(project)}")
            cur.execute("DELETE FROM projects WHERE name = %s", (project,))
        conn.commit()
    finally:
        conn.close()

    page = list_projects(include_counts=True)
    assert page["ok"]
    row = next(item for item in page["results"]
               if item["project"] == project)
    assert row["has_table"] is False
    assert row["entities"] == 2
    assert row["relations"] == 1


def test_ensure_entity_serializes_creation():
    from mcp_server.graph_store import _ensure_entity

    class FakeCursor:
        def __init__(self):
            self.statements = []
            self.connection = self

        def execute(self, sql, params=None):
            self.statements.append(str(sql))

        def fetchone(self):
            return (True,)

        def cursor(self):
            return self

        def close(self):
            return None

    cur = FakeCursor()
    assert _ensure_entity(cur, "R29C3", None) is True
    joined = "\n".join(cur.statements)
    assert "pg_advisory_xact_lock" in joined
    assert "MERGE" in joined


def test_age_json_keeps_type_suffixes_inside_values():
    from mcp_server.graph_store import _age_json

    raw = ('[{"id": 1, "properties": {"name": "A::vertexB"}}::vertex, '
           '{"id": 2, "properties": {"name": "C::edgeD"}}::edge]')
    parsed = _age_json(raw)
    assert [item["properties"]["name"] for item in parsed] == [
        "A::vertexB", "C::edgeD"]


def test_edge_context_is_scoped_to_its_endpoints_and_ttl():
    project = "round29-c5"
    assert relate("R29C5X", "R29C5Y_R29C5REL_R29C5Z", "R29C5REL",
                  context="R29C5 first", project=project)["ok"]
    assert relate("R29C5X_R29C5REL_R29C5Y", "R29C5Z", "R29C5REL",
                  context="R29C5 second", project=project)["ok"]

    from mcp_server.graph import find_connections
    edges = find_connections("R29C5X", project=project)
    assert edges["ok"]
    edge = next(item for item in edges["results"]
                if item["entity"] == "R29C5Y_R29C5REL_R29C5Z")
    assert edge["context"] == "R29C5 first"

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE {_mem_table(project)} "
                f"SET expires_at = NOW() - INTERVAL '1 day' "
                f"WHERE metadata->>'from' = %s AND metadata->>'to' = %s",
                ("R29C5X", "R29C5Y_R29C5REL_R29C5Z"),
            )
            assert cur.rowcount == 1
        conn.commit()
    finally:
        conn.close()

    edges = find_connections("R29C5X", project=project)
    edge = next(item for item in edges["results"]
                if item["entity"] == "R29C5Y_R29C5REL_R29C5Z")
    assert "context" not in edge

def test_recall_limit_with_entity_filter_keeps_parameter_order():
    project = "round29-b1-entity"
    facts = [{"content": f"R29B1E token item {i}", "entity": "R29B1E"}
             for i in range(60)]
    stored = remember_batch(facts=facts, project=project)
    assert stored["ok"] and stored["succeeded"] == 60

    found = recall(query="R29B1E token", project=project,
                   entity="R29B1E", limit=60)
    assert found["ok"]
    assert len(found["results"]) == 60, found.get("warnings")

def test_invalid_fts_language_is_reported_not_silently_ignored(monkeypatch):
    import importlib

    from mcp_server import config
    from mcp_server.startup_status import (
        clear_startup_warnings,
        startup_warnings,
    )

    clear_startup_warnings()
    monkeypatch.setenv("FTS_LANGUAGE", "english:bad")
    try:
        reloaded = importlib.reload(config)
        assert reloaded.FTS_LANGUAGE == "english"
        assert any("FTS_LANGUAGE" in warning
                   for warning in startup_warnings())
    finally:
        monkeypatch.delenv("FTS_LANGUAGE", raising=False)
        importlib.reload(config)
        clear_startup_warnings()
