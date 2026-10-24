"""Regression tests for the thirtieth round of fixes (agent audit).

Covers:
- `prune_memories` clears dangling supersede links in every scope, not only
  in the tables it just pruned;
- the NLI client retries a transport failure but not a 400;
- `server_health` closes its store even when the ping fails;
- the contradiction scan counts every processed seed (also seeds without
  neighbours), reports the seeds the cap skipped, and accepts a score of
  exactly the contradiction threshold;
- a blank project in an imported graph record is reported per record instead
  of silently writing the entity to the global scope;
- a hidden exact match respects the non-visibility filters of the query;
- the best-dropped hint reports the project name and keeps a row whose id is
  only visible in another scope;
- the topic tokenizer keeps accented words intact and drops German
  stopwords;
- the seed cap is reachable in a real scope (the loader reads one row past
  the cap) and `relate_batch` takes its per-scope locks in sorted order so
  two batches cannot deadlock;
- `contradictions_report` refuses a non-positive `limit`;
- `update_batch` releases every savepoint it rolls back;
- an unknown FTS dictionary falls back to `english` at startup;
- `ensure_entity` reports a conflict update as not-created and still updates
  the stored type.
"""

import requests
from pydantic import ValidationError, validate_call

import mcp_server.contradiction_scan as scan_module
import mcp_server.inference_client as inference_client
import mcp_server.schema_migrator as schema_migrator
import mcp_server.tools.maintenance as maintenance
import mcp_server.tools.memory as memory
from mcp_server import config
from mcp_server.contradiction_scan import (
    CandidateGroup,
    ContradictionScan,
    Judgement,
    Neighbour,
    ScanReport,
    ScanRequest,
    report_from_judgement,
)
from mcp_server.db import get_conn
from mcp_server.duplicates import topic_tokens
from mcp_server.graph_store import GraphStore
from mcp_server.search import _decorate_best_dropped
from mcp_server.server import (
    contradictions_report,
    import_memories,
    list_memories,
    prune_memories,
    recall,
    relate_batch,
    remember,
    server_health,
    update_batch,
)
from mcp_server.store import MemoryStore


def test_prune_clears_dangling_supersede_links_in_other_scopes():
    old = remember("round30 old fact", project="round30-prune-a")
    replacement = remember("round30 new fact", project="round30-prune-a",
                           supersede=old["id"])
    remember("round30 other scope fact", project="round30-prune-b")

    # Hard-delete the replacement behind the tool's back: `forget(soft=False)`
    # would already clear the link, so the prune call must be the one that
    # repairs it.
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'public' "
                    "AND table_name LIKE 'mem\\_%' AND table_name <> 'memories'")
        table = [t for (t,) in cur.fetchall() if t != "memories"][0]
        cur.execute(f'DELETE FROM "{table}" WHERE id = %s',
                    (replacement["id"],))
        conn.commit()
    finally:
        conn.close()

    assert list_memories(memory_id=old["id"],
                         project="round30-prune-a")["ok"] is False

    result = prune_memories(project="round30-prune-b", older_than_days=0,
                            dry_run=False)

    assert result["ok"] is True
    restored = list_memories(memory_id=old["id"], project="round30-prune-a")
    assert restored["ok"] is True
    assert restored["memory"]["superseded_by"] is None


class _Response:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)

    def json(self):
        return self._payload


class _Session:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def post(self, *args, **kwargs):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_nli_retries_a_transport_failure(monkeypatch):
    session = _Session([
        requests.ConnectionError("refused"),
        _Response(200, {"contradiction_scores": [0.0]}),
    ])
    monkeypatch.setattr(inference_client, "_get_session", lambda: session)
    monkeypatch.setattr(inference_client.time, "sleep", lambda _: None)

    scores, partial, failed = inference_client.nli_detailed(
        "query", ["document"], timeout=5.0)

    assert session.calls == 2
    assert scores == [0.0]
    assert partial is None
    assert failed == []


def test_nli_does_not_retry_a_400(monkeypatch):
    session = _Session([_Response(400, {"error": "bad request"})])
    monkeypatch.setattr(inference_client, "_get_session", lambda: session)

    try:
        inference_client.nli_detailed("query", ["document"], timeout=5.0)
    except requests.HTTPError:
        pass
    else:  # pragma: no cover - the call must fail
        raise AssertionError("a 400 must be surfaced, not swallowed")

    assert session.calls == 1


def test_server_health_closes_the_store_after_a_failed_ping(monkeypatch):
    closed: list[bool] = []

    class _BrokenStore:
        def ping(self):
            raise RuntimeError("postgres is down")

        def commit(self):  # pragma: no cover - never reached
            raise AssertionError("commit must not run after a failed ping")

        def close(self):
            closed.append(True)

    monkeypatch.setattr(maintenance, "MemoryStore", _BrokenStore)

    result = server_health()

    assert result["postgres"] is False
    assert closed == [True]


def test_seed_cap_counts_seeds_without_neighbours(monkeypatch):
    monkeypatch.setattr(scan_module, "MAX_SCAN_SEEDS", 2)

    class _DB:
        cur = None

        def nearest(self, **kwargs):
            return []

    facts = [{"id": f"fact-{i}", "content": f"fact {i}",
              "embedding": [0.1] * 8} for i in range(5)]
    scan = ContradictionScan(lambda *a, **k: ([0.0], None, []))

    loaded = scan._load_candidates(ScanRequest(db=_DB()), facts)

    assert loaded.skipped_seeds == 3
    assert loaded.skipped_seed_ids == ["fact-2", "fact-3", "fact-4"]

    report = scan._note_seed_limit(ScanReport(), loaded)
    assert any("candidate seed" in warning for warning in report.warnings)
    assert [failure["fact_id"] for failure in report.failures] == [
        "fact-2", "fact-3", "fact-4"]
    assert all(failure["kind"] == "seed_limit"
               for failure in report.failures)


def test_relate_batch_acquires_scope_locks_in_a_stable_order(monkeypatch):
    # `_ensure_entity` locks the scope; a batch that covers two scopes in
    # insertion order could take the locks in the opposite order of a
    # concurrent batch and deadlock. The tool must sort its scope groups.
    visited: list[str | None] = []

    def fake_relate(self, edges, project=None):
        visited.append(project)
        return [{"status": "created"} for _ in edges]

    monkeypatch.setattr(GraphStore, "relate", fake_relate)

    result = relate_batch(edges=[
        {"from_entity": "A", "to_entity": "B", "relation_type": "R",
         "project": "zeta"},
        {"from_entity": "C", "to_entity": "D", "relation_type": "R",
         "project": "alpha"},
    ])

    assert result["ok"] is True
    assert visited == ["alpha", "zeta"]


def test_seed_cap_warning_is_reachable_in_a_real_scope(monkeypatch):
    # `_load` must load one row beyond the cap; otherwise the loop never sees
    # a leftover seed and this warning can never fire (the round-30 bug).
    monkeypatch.setattr(scan_module, "MAX_SCAN_SEEDS", 2)
    for i in range(3):
        remember(f"Seed cap probe {i}: worker alpha listens on port {9000 + i}")

    result = contradictions_report(limit=5)

    assert any("candidate seed" in warning for warning in result["warnings"])
    assert any(failure["kind"] == "seed_limit"
               for failure in result["nli_failures"])


def test_a_score_at_the_threshold_is_a_contradiction():
    group = CandidateGroup(
        fact={"id": "a", "content": "The port is 8000"},
        neighbours=[Neighbour(id="b", content="The port is 8001",
                              entity=None, distance=0.1)],
    )
    judgement = Judgement(
        scores_by_fact={"a": ([scan_module.CONTRADICTION_SCORE_THRESHOLD],
                              None, [])},
        budget=40.0,
    )

    report = report_from_judgement([group], judgement, entity=None, limit=20,
                                   timestamps={})

    assert len(report.pairs) == 1
    assert report.pairs[0]["confidence"] == scan_module.CONTRADICTION_SCORE_THRESHOLD


def test_import_reports_a_blank_graph_project_per_record():
    result = import_memories(data={
        "version": "0.3",
        "records": [],
        "graph": {
            "entities": [{"name": "Round30BlankScope", "project": "   "}],
            "relations": [],
        },
    })

    assert result["ok"] is True
    assert any("graph entity" in error for error in result["errors"])
    assert result["graph_entities"] == 0


def test_hidden_exact_match_respects_the_mem_type_filter():
    project = "round30-hidden-exact"
    remember("Round30 ZEBRAFISH_R30 lives here",
             project=project, mem_type="note",
             expires_at="2020-01-01T00:00:00Z")

    wrong_type = recall("ZEBRAFISH_R30", project=project, mem_type="fact")
    assert wrong_type.get("hidden_exact_match") is None

    right_type = recall("ZEBRAFISH_R30", project=project, mem_type="note")
    assert right_type["hidden_exact_match"]["count"] == 1
    assert "include_expired" in right_type["hidden_exact_match"]["flags"]


class _ProjectsCursor:
    def __init__(self):
        self._rows = [("round30suffix", "round30-name")]

    def execute(self, *args, **kwargs):
        return None

    def fetchall(self):
        return self._rows


def test_best_dropped_reports_the_project_name():
    dropped = {"_table": "mem_round30suffix", "id": "x", "similarity": 0.29}

    _decorate_best_dropped(_ProjectsCursor(), dropped)

    assert dropped["scope"] == "round30-name"


def test_best_dropped_keeps_a_row_visible_only_in_another_scope():
    dropped = {"_table": "mem_round30suffix", "id": "x", "similarity": 0.29}

    # The same id exists in the global scope, but this hint is about the
    # project row: it is still a real near-miss.
    _decorate_best_dropped(_ProjectsCursor(), dropped, {("global", "x")})

    assert dropped["scope"] == "round30-name"

    visible = {"_table": "mem_round30suffix", "id": "x", "similarity": 0.29}
    _decorate_best_dropped(_ProjectsCursor(), visible,
                           {("round30-name", "x")})

    assert visible == {}


def test_topic_tokens_keep_accented_words():
    assert topic_tokens("Preis für Betrieb", None) == {"preis", "betrieb"}
    assert topic_tokens("Prüfsumme der Größe", None) == {"prüfsumme", "größe"}


def test_contradictions_report_rejects_a_non_positive_limit():
    for limit in (0, -1):
        try:
            validate_call(contradictions_report)(limit=limit)
        except ValidationError:
            continue
        raise AssertionError(f"limit={limit} must be rejected")


def test_update_batch_releases_every_rolled_back_savepoint(monkeypatch):
    class _CountingStore(MemoryStore):
        rollbacks = 0
        releases = 0

        def rollback_to(self, name):
            type(self).rollbacks += 1
            return super().rollback_to(name)

        def release(self, name):
            type(self).releases += 1
            return super().release(name)

    monkeypatch.setattr(memory, "MemoryStore", _CountingStore)

    result = update_batch(updates=[
        {"memory_id": f"round30-missing-{i}", "importance": 0.6}
        for i in range(20)])

    assert result["failed"] == 20
    assert _CountingStore.rollbacks == 20
    assert _CountingStore.releases == 20


def test_unknown_fts_language_falls_back_to_english(monkeypatch):
    recorded: list[str] = []
    monkeypatch.setattr(config, "FTS_LANGUAGE", "not_a_real_dictionary")
    monkeypatch.setattr(schema_migrator, "record_warning", recorded.append)

    schema_migrator.ensure_fts_language()

    assert config.FTS_LANGUAGE == "english"
    assert recorded and "falling back" in recorded[0]


def test_known_fts_language_is_kept(monkeypatch):
    recorded: list[str] = []
    monkeypatch.setattr(config, "FTS_LANGUAGE", "english")
    monkeypatch.setattr(schema_migrator, "record_warning", recorded.append)

    schema_migrator.ensure_fts_language()

    assert config.FTS_LANGUAGE == "english"
    assert recorded == []


def test_ensure_entity_reports_a_conflict_as_not_created():
    project = "round30-entity"
    store = MemoryStore()
    try:
        graph = GraphStore(db=store)
        assert graph.ensure_entity("Round30Node", project, "FirstType") is True
        assert graph.ensure_entity("Round30Node", project, "SecondType") is False
        store.cur.execute(
            "SELECT type FROM entities WHERE name = %s AND project = %s",
            ("Round30Node", project))
        assert store.cur.fetchone()[0] == "SecondType"
        store.commit()
    finally:
        store.close()
