"""Regression tests for the seventh round of fixes.

Every test locks in a concrete audit finding: a restore must not resurrect
a superseded fact, the forgotten sentinel must not overwrite a real insert,
`include_deleted` and `include_superseded` are orthogonal, `recall` can
hide the relation-context sentences, an explicitly requested scope is no
warning, and entity rows say which kind they are.
"""

from mcp_server.server import (
    forget,
    get_entity,
    list_entities,
    list_memories,
    recall,
    relate,
    remember,
    restore_memory,
)
from mcp_server.config import SENTINEL_FORGOTTEN


# --- restore must not resurrect a superseded fact -------------------------

def test_restore_keeps_a_real_supersede_intact():
    old_id = remember("Revival contract: the old fact",
                      project="revive-scope")["id"]
    new_id = remember("Revival contract: the new fact",
                      project="revive-scope", supersede=old_id)["id"]

    forget(memory_id=old_id, project="revive-scope")  # soft
    result = restore_memory(old_id, project="revive-scope")
    assert result["ok"] is True

    listed = list_memories(project="revive-scope", memory_id=old_id,
                           include_superseded=True,
                           fields=["id", "deleted_at", "superseded_by"])
    row = listed["results"][0]
    # The restore clears the forgotten sentinel, never the invalidation.
    assert row["deleted_at"] is None
    assert row["superseded_by"] == new_id
    assert row["superseded_by"] != SENTINEL_FORGOTTEN

    still_active = recall("revival contract", project="revive-scope", limit=10)
    hits = {r["id"]: r for r in still_active["results"]}
    assert new_id in hits
    assert old_id not in hits


def test_forget_does_not_overwrite_a_real_supersede_with_the_sentinel():
    old_id = remember("Sentinel contract: the old fact",
                      project="sentinel-scope")["id"]
    new_id = remember("Sentinel contract: the new fact",
                      project="sentinel-scope", supersede=old_id)["id"]

    forget(memory_id=old_id, project="sentinel-scope")

    listed = list_memories(project="sentinel-scope", memory_id=old_id,
                           include_deleted=True, include_superseded=True)
    row = listed["results"][0]
    assert row["deleted_at"] is not None
    assert row["superseded_by"] == new_id
    assert row["superseded_by"] != SENTINEL_FORGOTTEN


def test_a_plain_forgotten_memory_carries_the_forgotten_supersede_marker():
    stored = remember("Sentinel contract: a plain fact",
                      project="sentinel-scope")["id"]
    forget(memory_id=stored, project="sentinel-scope")

    listed = list_memories(project="sentinel-scope", memory_id=stored,
                           include_deleted=True, include_superseded=True,
                           fields=["id", "superseded_by", "forgotten"])
    row = listed["results"][0]
    # The sentinel is hidden from `superseded_by` in the projection, so the
    # row reports the `forgotten` flag instead.
    assert row["forgotten"] is True


# --- include_deleted and include_superseded are orthogonal ---------------

def test_include_deleted_does_not_leak_superseded_history():
    old_id = remember("Orthogonal contract: the old fact",
                      project="orthogonal-scope")["id"]
    remember("Orthogonal contract: the new fact", project="orthogonal-scope",
             supersede=old_id)

    deleted = list_memories(project="orthogonal-scope",
                            contains="Orthogonal contract",
                            include_deleted=True)
    ids = {r["id"] for r in deleted["results"]}
    assert old_id not in ids
    assert deleted["count"] == len(deleted["results"])

    with_history = list_memories(project="orthogonal-scope",
                                 contains="Orthogonal contract",
                                 include_deleted=True,
                                 include_superseded=True)
    ids = {r["id"] for r in with_history["results"]}
    assert old_id in ids


# --- recall can hide the relation-context sentences -----------------------

def test_recall_can_hide_the_relation_context_sentences():
    relate("RecCtxA", "RecCtxB", "USES",
           context="RecCtxA uses RecCtxB for the night run", project="recctx")

    default = recall("RecCtxB", project="recctx", limit=10)
    assert any(m["entity"] and "_USES_" in m["entity"]
               for m in default["results"])

    hidden = recall("RecCtxB", project="recctx", limit=10,
                    include_relation_context=False)
    assert not any(m["entity"] and "_USES_" in m["entity"]
                   for m in hidden["results"])


# --- an explicitly requested scope is no other-scopes warning -------------

def test_explicit_global_scope_raises_no_other_scopes_warning():
    remember("Explicit scope contract: a project fact", project="beta")
    result = recall("explicit scope contract", project="global", limit=10)

    assert result["ok"] is True
    assert not any("other scopes exist" in w for w in result["warnings"])


def test_explicit_project_scope_raises_no_other_scopes_warning():
    remember("Explicit scope contract: a project fact", project="beta")
    remember("Explicit scope contract: a global fact")

    result = recall("explicit scope contract", project="beta", limit=10)

    assert result["ok"] is True
    assert not any("other scopes exist" in w for w in result["warnings"])


def test_the_project_scope_filter_is_explicit_in_the_answer():
    remember("Explicit scope contract: a project fact", project="beta")
    remember("Explicit scope contract: a global fact")

    assert recall("explicit scope contract", project="beta",
                  limit=10)["scopes_searched"] == ["beta", "global"]
    assert recall("explicit scope contract", project="global",
                  limit=10)["scopes_searched"] == ["global"]


# --- entity rows say which kind they are ---------------------------------

def test_get_entity_reports_the_graph_type_with_its_label():
    relate("KindA", "KindB", "KNOWS", project="kinds")
    project = get_entity(entity="KindA", project="kinds")

    assert project["details"]["type"] == "graph"
    assert project["details"]["label"]

    global_relate = relate("KindC", "KindD", "KNOWS")
    assert global_relate["ok"] is True
    detail = get_entity(entity="KindC")
    assert detail["details"]["type"] == "graph"
    assert detail["details"]["label"] == "Entity"


def test_get_entity_reports_memory_only_entities():
    remember("Kind contract: memory-only fact", entity="KindMemOnly",
             project="kinds")

    detail = get_entity(entity="KindMemOnly", project="kinds")
    assert detail["details"]["type"] == "memory-only"


def test_list_entities_rows_type_is_graph_or_memory_only_only():
    remember("Kind contract: memory-only fact", entity="KindMemOnly",
             project="kinds")
    relate("KindGraphA", "KindGraphB", "KNOWS", project="kinds")

    rows = list_entities(project="kinds", include_global=False)["results"]
    assert rows
    for row in rows:
        assert row["type"] in ("graph", "memory-only")
        if row["graph_node"]:
            assert row["type"] == "graph" and row["label"]
        else:
            assert row["type"] == "memory-only" and row["label"] is None


def test_list_entities_graph_filter_selects_only_graph_nodes():
    relate("KindGraphA", "KindGraphB", "KNOWS", project="kinds")

    rows = list_entities(entity_type="graph", project="kinds",
                         include_global=False)["results"]
    assert rows
    assert all(r["type"] == "graph" and r["graph_node"] for r in rows)


# --- forget(entity='*') does not claim more than it did -------------------

def test_soft_star_forget_says_the_rows_stay_recoverable():
    remember("Star scope contract: a fact", project="star-scope")

    result = forget(entity="*", project="star-scope")

    assert result["status"] == "soft_forgotten"
    assert "restore_memory" in result["message"]

    recovered = list_memories(project="star-scope", contains="Star scope",
                              include_deleted=True)
    assert recovered["count"] == 1


# --- the NLI provider retry tolerates one bad JSON response ---------------


def test_nli_retries_invalid_json_with_the_same_format(monkeypatch):
    import requests
    from inference import llm as llm

    config = llm.LLMConfig(provider="groq", model="openai/gpt-oss-20b",
                           api_key="k", base_url="https://api.groq.com/openai/v1",
                           style="openai")

    responses = iter([
        {'choices': [{'message': {'content': 'not json at all'}}]},
        {'choices': [{'message': {'content': '{"label": "contradiction", "score": 0.9}'}}]},
    ])

    def fake_post(url, headers=None, json=None, timeout=None):
        class Resp:
            status_code = 200
            def json(self):
                return next(responses)
        return Resp()

    monkeypatch.setattr(requests, 'post', fake_post)
    assert llm.classify(config, 'A', 'B') == 0.9


def test_nli_fails_when_every_attempt_is_invalid_json(monkeypatch):
    import requests
    from inference import llm as llm

    config = llm.LLMConfig(provider="groq", model="openai/gpt-oss-20b",
                           api_key="k", base_url="https://api.groq.com/openai/v1",
                           style="openai")

    def fake_post(url, headers=None, json=None, timeout=None):
        class Resp:
            status_code = 200
            def json(self):
                return {'choices': [{'message': {'content': 'not json at all'}}]}
        return Resp()

    monkeypatch.setattr(requests, 'post', fake_post)
    import pytest
    with pytest.raises(RuntimeError, match='not valid NLI JSON'):
        llm.classify(config, 'A', 'B')

