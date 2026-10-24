"""write_one owns the order; the tools only render its outcome."""

from contextlib import contextmanager

import mcp_server.write_pipeline as wp
from mcp_server.tools import memory


class FakeCur:
    def __init__(self):
        self.executed: list[tuple[str, object]] = []

    def execute(self, sql, params=()):
        self.executed.append((sql, params))

    def fetchone(self):
        return None


class FakeStore:
    def __init__(self):
        self.cur = FakeCur()
        self.calls: list[tuple] = []

    def ensure_scope(self, project):
        self.calls.append(("ensure_scope", project))
        return False

    def insert(self, content, embedding_vec, mem_type, entity, importance,
               metadata, *, project=None, scope=None, expires_at=None):
        self.calls.append(("insert", content, mem_type, entity,
                           dict(metadata), project, scope))
        return "new-id"

    def mark_superseded(self, memory_id, superseded_by, *, project=None,
                        target=None):
        self.calls.append(("mark_superseded", memory_id, superseded_by,
                           project, target))
        return 1


def _quiet_pipeline(monkeypatch, **overrides):
    defaults = {
        "_deduplicate": lambda *a, **k: None,
        "_find_superseded_match": lambda *a, **k: None,
        "_find_near_duplicate": lambda *a, **k: None,
        "_check_contradictions": lambda *a, **k: [],
        "_supersede_targets": lambda *a, **k: ([], []),
        "_resolve_mem_type": lambda *a, **k: "fact",
        "_cross_scope_duplicates": lambda *a, **k: [],
        "_apply_remember_links": lambda *a, **k: ([], []),
        "_read_metadata": lambda *a, **k: {},
        "_scope_of_id": lambda *a, **k: "global",
    }
    defaults.update(overrides)
    for name, value in defaults.items():
        monkeypatch.setattr(wp, name, value)


def test_write_one_inserts_once_in_order(monkeypatch):
    _quiet_pipeline(monkeypatch)
    store = FakeStore()

    outcome = wp.write_one(store, wp.WriteIntent(
        content="API uses Postgres", project="api", embedding=[0.0]))

    assert outcome.status == "stored"
    assert outcome.id == "new-id"
    assert [call[0] for call in store.calls] == ["ensure_scope", "insert"]


def test_duplicate_short_circuits_before_insert(monkeypatch):
    _quiet_pipeline(
        monkeypatch,
        _deduplicate=lambda *a, **k: ("dup-1", "semantic", 0.93))
    store = FakeStore()

    outcome = wp.write_one(store, wp.WriteIntent(
        content="same fact", embedding=[0.0]))

    assert outcome.status == "duplicate"
    assert outcome.id == "dup-1"
    assert outcome.duplicate_mode == "semantic"
    assert outcome.similarity == 0.93
    assert outcome.persist is True
    assert [call[0] for call in store.calls] == ["ensure_scope"]


def test_supersede_marks_after_insert_and_carries_metadata(monkeypatch):
    _quiet_pipeline(
        monkeypatch,
        _supersede_targets=lambda *a, **k: (["old-1"], []),
        _read_metadata=lambda *a, **k: {"ticket": "T-1"})
    store = FakeStore()

    outcome = wp.write_one(store, wp.WriteIntent(
        content="new fact", embedding=[0.0], metadata={"env": "prod"}))

    assert [call[0] for call in store.calls] == [
        "ensure_scope", "insert", "mark_superseded"]
    inserted = store.calls[1]
    assert inserted[4] == {"env": "prod"}
    assert outcome.superseded == ["old-1"]
    assert outcome.metadata_carried == ["ticket"]
    sql, params = store.cur.executed[-1]
    assert "UPDATE memories SET metadata" in sql
    assert '"env": "prod"' in params[0]
    assert '"ticket": "T-1"' in params[0]


def test_link_failure_is_a_warning_not_a_lost_fact(monkeypatch):
    _quiet_pipeline(
        monkeypatch,
        _apply_remember_links=lambda *a, **k: ([], ["links failed: boom"]))
    store = FakeStore()

    outcome = wp.write_one(store, wp.WriteIntent(
        content="fact with a broken edge",
        links=[{"to_entity": "Postgres", "relation_type": "USES"}],
        embedding=[0.0]))

    assert outcome.status == "stored"
    assert outcome.id == "new-id"
    assert "links failed: boom" in outcome.warnings
    assert any(call[0] == "insert" for call in store.calls)


@contextmanager
def _attempt(name="entry"):
    yield


def test_remember_batch_reports_one_failed_entry(monkeypatch):
    class FakeBatchStore(FakeStore):
        def attempt(self, name="entry"):
            self.calls.append(("savepoint", name))
            return _attempt(name)

        def commit(self):
            self.calls.append(("commit",))

        def rollback(self):
            self.calls.append(("rollback",))

        def close(self):
            self.calls.append(("close",))

    store = FakeBatchStore()

    def fake_write_one(db, intent):
        if intent.content == "bad":
            raise RuntimeError("boom")
        return wp.WriteOutcome(
            status="stored", id="id-" + intent.content,
            scope=intent.scope, project=intent.project or "global",
            content=intent.content)

    monkeypatch.setattr(memory, "MemoryStore", lambda: store)
    monkeypatch.setattr(memory, "write_one", fake_write_one)
    monkeypatch.setattr(memory, "embed",
                        lambda texts: [[0.0] for _ in texts])

    result = memory.remember_batch(facts=[
        {"content": "good"}, {"content": "bad"}])

    assert result["status"] == "partial"
    assert result["succeeded"] == 1
    assert result["failed"] == 1
    assert result["results"][0]["status"] == "stored"
    assert result["results"][1] == {
        "id": None, "content": "bad", "status": "error",
        "error": "boom", "scope": "global", "project": "global",
    }
    savepoints = [c for c in store.calls if c[0] == "savepoint"]
    assert savepoints == [("savepoint", "fact_0"), ("savepoint", "fact_1")]
    assert ("close",) in store.calls
